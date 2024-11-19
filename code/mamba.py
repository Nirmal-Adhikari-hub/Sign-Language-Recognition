import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math
import os

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from transformers import MambaForCausalLM, AutoTokenizer, GPT2Model, GPT2Config

class Block(nn.Module):
    def __init__(
            self, dim: int, mixer_cls: Mamba, norm_cls=nn.LayerNorm, fused_add_norm: bool=False,
            residual_in_fp32: bool=False, drop_path: float=0.
        ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim) # Mamba block
        self.norm = norm_cls(dim)
        self.drop_path =  DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "If you use fused_add_norm, you need RMSNorm layer"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), "Normalization layer must be LayerNorm or RMSNorm"
        
    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor]=None, inference_params=None,
            use_checkpoint: bool = False
        ):
        # Block마다 residual과 norm을 같이 계산하는지 여부
        if not self.fused_add_norm: # 같이 계산하지 않고, add -> norm
            residual = (residual + self.drop_path(hidden_states) if residual is not None else hidden_states)
            hidden_states = self.norm(residual.to(self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else: # fused_add_norm 사용
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states) if residual is not None else hidden_states,
                self.norm.weight, self.norm.bias, residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps
            )

        # Recomputation
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params, use_reentrant=True)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
        d_model: int , ssm_cfg: Optional[dict]=None, norm_epsilon: float=1e-5, drop_path: float=0., rms_norm: bool=True,
        residual_in_fp32: bool=True, fused_add_norm: bool=True, layer_idx: Optional[int]=None,
        device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None
    ):
    factory_kwargs = {'device': device, 'dtype': dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx,**ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, norm_cls, fused_add_norm, residual_in_fp32, drop_path)
    block.layer_idx = layer_idx
    
    return block

# Init linear and embedding layer
def _init_params(
        module: nn.Module, n_layer: int, initializer_range: float=.02, rescale_prenorm_residual: bool=True, n_residuals_per_layer: int=1
    ):
    # Init Linear layer
    if isinstance(module, nn.Linear):
        if module.bias is not None: # bias 존재 여부
            if not getattr(module.bias, '_no_reinit', False): # bias가 _no_reinit property를 갖는지 확인(_no_reinit이 없는 것만 초기화)
                nn.init.zeros_(module.bias)
    # Embedding layer            
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range) # 정규 분포로 weight 초기화
    
    # Block 내의 연산 전에 Normalization 
    if rescale_prenorm_residual:
        for name, param in module.named_parameters():
            if name in ['in_proj.weight', 'out_proj.weight', 'fc2.weight']:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                with torch.no_grad():
                    param /= math.sqrt(n_residuals_per_layer * n_layer)

# Init head
def segm_init_params(m):
    if isinstance(m, nn.Linear): # For linear layer
        trunc_normal_(m.weight, std=.02) # Init to normal dist with trunc
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) # Init bias
    elif isinstance(m, nn.LayerNorm): # For LayerNorm
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0)

class MambaBlocks(nn.Module):
    def __init__(
            self, d_model, depth, vocab_size, max_len, n_classes: int=22, fused_add_norm: bool=True,
            residual_in_fp32: bool=True, rms_norm: bool=True, drop_rate: float=0., drop_path_rate: float=.1,
            fc_drop_rate: float=0., ssm_cfg: Optional[dict]=None, norm_epsilon: float=1e-5,
            initializer_cfg=None, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None,
            use_checkpoint: bool=False, checkpoint_num: int=0, no_head=False,
        ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.no_head = no_head
        print(f"Use chcekpoint: {use_checkpoint}")
        print(f"Checkpoint number: {checkpoint_num}")        
    
        self.n_classes = n_classes
        self.d_model = self.n_features = self.embed_dim = d_model # Model's dimension

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = nn.Linear(d_model, n_classes)
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0. else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # Stochastic depth decay rule
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Mamba layer with Depth decay rule
        self.layers = nn.ModuleList([
            create_block(
                d_model, ssm_cfg, norm_epsilon, dpr[i], rms_norm, residual_in_fp32,
                fused_add_norm, layer_idx=i, **factory_kwargs
            )
            for i in range(depth)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)

        # Params init
        self.apply(segm_init_params)
        self.apply(partial(_init_params, n_layer=depth, **(initializer_cfg if initializer_cfg is not None else {}))) # For Mamba blocks

    # Inference 시 미리 메모리 할당
    def allocate_inference_cahce(self, batch_size: int, max_seqlen: int, dtype: Optional[torch.dtype]=None, **kwargs):
        return {
            i: layer.allocate_inference_cahce(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def get_n_layers(self):
        return len(self.layers)

    def forward(self, x, inference_params=None):
        """
        x: (B, L, D)
        """
        x = self.embeddings(x) # (B, L, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Mamba implement
        residual = None
        hidden_states = x

        # Mamba blocks
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

            if idx == len(self.layers) - 1:
                # Normalization
                if not self.fused_add_norm:
                    if residual is None:
                        residual = hidden_states
                    else:
                        residual = residual + self.drop_path(hidden_states)
                    hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
                else:
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states = fused_add_norm_fn(
                        self.drop_path(hidden_states),
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32
                    )

        cls = hidden_states[:, 0, :]
        
        if self.no_head:
            return cls
        else:
            cls = self.head(self.head_drop(cls))
        
        # Return cls token
        return cls
    
def load_state_dict(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    pretrained_state_dict = {k[9:]: v for k, v in pretrained_state_dict.items() if k[9:] in model_state_dict} # Remove 'backbone.'

    for k, v in pretrained_state_dict.items():
        if v.shape != model_state_dict[k].shape:
            print(f"Skipping weight {k} due to mismatch in shape: {v.shape} vs {model_state_dict[k].shape}")
    
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

def mamba_130m(tokenizer, pretrained=False, **kwargs):
    model = MambaBlocks(
        d_model=768,
        depth=24,
        vocab_size=len(tokenizer),
        **kwargs
    )

    if pretrained:
        print("Load pretrained model")
        pretrained_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        pretrained_model.resize_token_embeddings(len(tokenizer))
        load_state_dict(model, pretrained_model.state_dict())

    return model
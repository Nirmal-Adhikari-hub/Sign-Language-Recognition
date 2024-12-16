# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
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
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
def create_block(
        d_model: int , ssm_cfg: Optional[dict]=None, norm_epsilon: float=1e-5, drop_path: float=0., rms_norm: bool=True,
        residual_in_fp32: bool=True, fused_add_norm: bool=True, layer_idx: Optional[int]=None, bimamba: bool=True,
        device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None
    ):
    factory_kwargs = {'device': device, 'dtype': dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
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

class PatchEmbed(nn.Module):
    """
    Image to Patch
    """
    def __init__(self, width: int=224, height: int=224, patch_size: list=[16, 16], kernel_size: int=1, in_chans: int=3, embed_dim: int=768):
        super().__init__()
        assert (width % patch_size[0] == 0) or (height % patch_size[1] == 0), 'Width and Height should be multiple of patch size'
        patch_size = to_2tuple(patch_size)
        n_patches = (width // patch_size[0]) * (height // patch_size[1])
        self.img_size = (height, width)
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.kernel_size = kernel_size

        # Input shape: (B, C, T, H, W)
        # 동영상은 W, H, T의 3개의 차원을 가지므로 Conv3D을 사용해 projection
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )
    
    def forward(self, x):
        return self.proj(x) # (B, D, T, H / p, W / p)

class VisionMamba(nn.Module):
    def __init__(
            self, width: int=224, height: int=224, patch_size: list=[16, 16], depth: int=24, embed_dim: int=192, channels: int=3,
            n_classes: int=22, drop_rate: float=0., drop_path_rate: float=.1, ssm_cfg: Optional[dict]=None, norm_epsilon: float=1e-5,
            initializer_cfg=None, fused_add_norm: bool=True, rms_norm: bool=True, residual_in_fp32: bool=True, bimamba: bool=True,
            kernel_size: int=1, n_frames: int=16, fc_drop_rate: float=0., device: Optional[torch.device]=None,
            dtype: Optional[torch.dtype]=None, use_checkpoint: bool=False, checkpoint_num: int=0, no_head: bool=False
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.no_head = no_head
        print(f"Use checkpoint: {use_checkpoint}")
        print(f"Checkpoint number: {checkpoint_num}")

        self.n_classes = n_classes
        self.d_model = self.n_features = self.embed_dim = embed_dim # Model's dimension

        self.patch_embed = PatchEmbed(
            width=width, height=height, patch_size=patch_size, kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # (1, 1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, self.embed_dim)) # (1, #p + 1, D) -> #p + [cls]
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, n_frames // kernel_size, embed_dim)) # (1, #f, D)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if no_head is False:
            self.head = nn.Linear(self.n_features, n_classes)
            self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0. else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # Stochastic depth decay rule
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Mamba layer with Depth decay rule
        self.layers = nn.ModuleList([
            create_block(
                embed_dim, ssm_cfg, norm_epsilon, dpr[i], rms_norm, residual_in_fp32,
                fused_add_norm, layer_idx=i, bimamba=bimamba, **factory_kwargs
            )
            for i in range(depth)
        ])

        # Output Norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # Params init
        self.apply(segm_init_params)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(partial(_init_params, n_layer=depth, **(initializer_cfg if initializer_cfg is not None else {}))) # For Mamba blocks
    
    # Inference 시 미리 메모리 할당
    def allocate_inference_cahce(self, batch_size: int, max_seqlen: int, dtype: Optional[torch.dtype]=None, **kwargs):
        return {
            i: layer.allocate_inference_cahce(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'temporal_pos_embedding'}
    
    def get_n_layers(self):
        return len(self.layers)
    
    def forward_features(self, x: Tensor, inference_params=None):
        """
        1. Patch embedding
        2. Add cls token
        3. Postional embedding(+Temporal)
        4. Dropout
        5. Mamba blocks x depth
        6. Normalization
        7. Extract cls token
        """
        x = self.patch_embed(x) # (B, D, T, n_PH, n_PW)
        B, D, T, n_PH, n_PW = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B*T, n_PH*n_PW, D) # (BT, n_PH * n_PW, D)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # Batch size만큼 복제: (1, 1, D) -> (BT, 1, D)
        x = torch.cat((cls_token, x), dim=1) # 맨 앞에 추가 -> (BT, n_PH * n_PW + 1, D) = (BT, #p + 1, D)
        x = x + self.pos_embed # Add positional embedding

        # Add temporal pos embeddding
        # 배치마다 맨 앞의 frame에 대한 cls token
        cls_token = x[:B, :1, :] # (B, 1, D)
        # cls token을 제외 한 나머지
        x = x[:, 1:, :] # (BT, #p, D)
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T) # (B * #p, T, D)
        x = x + self.temporal_pos_embedding # (B * #p, T, D)
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T) # (B, T * #p, D)
        x = torch.cat((cls_token, x), dim=1) # (B, T * #p + 1, D)

        x = self.pos_drop(x)

        # Mamba implement
        residual = x
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
        
        return hidden_states[:, 0, :]
    
    def forward(self, x: Tensor, inference_params=None):
        x = self.forward_features(x, inference_params)
        if self.no_head:
            return x
        else:
            return self.head(self.head_drop(x))

def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d

# def load_state_dict(model, state_dict, center=True):
#     state_dict_3d = model.state_dict()
#     for k in state_dict.keys():
#         if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
#             if 'pos_embed' in k:
#                 print(f"Interpolating {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}")
#                 state_dict[k] = state_dict[k].permute(0, 2, 1) # (1, 768, 197)
#                 state_dict[k] = torch.nn.functional.interpolate(
#                     state_dict[k], size=state_dict_3d[k].shape[1], mode='linear', align_corners=False
#                 )
#                 state_dict[k] = state_dict[k].permute(0, 2, 1)
#             elif 'patch_embed.proj.weight' in k:
#                 print(f"Interpolating {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}")
#                 state_dict[k] = torch.nn.functional.interpolate(
#                     state_dict[k], size=state_dict_3d[k].shape[-2:], mode='bilinear', align_corners=False
#                 )
#                 state_dict[k] = state_dict[k]
#                 state_dict[k] = inflate_weight(state_dict[k], state_dict_3d[k].shape[2], center=center)
#                 print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')

#             if len(state_dict_3d[k].shape) <= 3 and 'pos_embed' not in k:
#                 print(f'Ignore: {k}')
    
#     del state_dict['head.weight']
#     del state_dict['head.bias']
#     msg = model.load_state_dict(state_dict, strict=False)
#     print(msg)

def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        embed_dim=384,
        depth=24,
        # rms_norm=False,
        # fused_add_norm=False,
        **kwargs
    )
    
    return model

def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        embed_dim=576,
        depth=32,
        # rms_norm=False,
        # fused_add_norm=False,
        **kwargs
    )

    if pretrained:
        print("Load pretrained params")
        state_dict = torch.load('/home/kks/workspace/SSM-Based-Movie-Genre-Classification/data/ckpt/videomamba_m16_in1k_res224.pth', map_location='cpu')
        load_state_dict(model, state_dict['model'])
    
    return model

def videomamba_base(pretrained=False, **kwargs):
    model = VisionMamba(
        embed_dim=768,
        depth=24,
        # rms_norm=False,
        # fused_add_norm=False,
        **kwargs
    )
    
    if pretrained:
        print("Load pretrained params")
        state_dict = torch.load('../../data/ckpt/videomamba_b16_in1k_res224.pth', map_location='cpu')
        load_state_dict(model, state_dict['model'])
    
    return model

if __name__ == '__main__':
    import numpy as np

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # model = videomamba_small(n_frames=32, kernel_size=1, width=854, height=480).cuda()
    # model = videomamba_middle(n_frames=64, kernel_size=1, width=854, height=480).cuda()
    model = videomamba_middle(pretrained=True, n_frames=1, kernel_size=1, width=224, height=224).cuda()
    
    # import time
    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import flop_count_table

    # flops = FlopCountAnalysis(model, torch.rand(1, 3, 32, 720, 1280).cuda())
    # s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time() - s)
    print(model(torch.rand(1, 3, 1, 224, 224).cuda()).shape)
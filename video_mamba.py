# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from functools import partial
from itertools import groupby
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

def ctc_decode_func(tf_gloss_logits, input_lengths, beam_size):
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits, 
        sequence_length=input_lengths.cpu().detach().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences

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
            drop_rate: float=0., drop_path_rate: float=.1, ssm_cfg: Optional[dict]=None, norm_epsilon: float=1e-5,
            initializer_cfg=None, fused_add_norm: bool=True, rms_norm: bool=True, residual_in_fp32: bool=True, bimamba: bool=True,
            kernel_size: int=1, n_frames: int=16, fc_drop_rate: float=0., device: Optional[torch.device]=None,
            dtype: Optional[torch.dtype]=None, use_checkpoint: bool=False, checkpoint_num: int=0
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f"Use checkpoint: {use_checkpoint}")
        print(f"Checkpoint number: {checkpoint_num}")

        self.d_model = self.n_features = self.embed_dim = embed_dim # Model's dimension

        self.patch_embed = PatchEmbed(
            width=width, height=height, patch_size=patch_size, kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # (1, 1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, self.embed_dim)) # (1, #p + 1, D) -> #p
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, n_frames // kernel_size, embed_dim)) # (1, #f, D)
        self.pos_drop = nn.Dropout(p=drop_rate)

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
        return {'pos_embed', 'temporal_pos_embedding'}
    
    def get_n_layers(self):
        return len(self.layers)
    
    def forward(self, x: Tensor, sgn_lengths, inference_params=None):
        """
        1. Patch embedding
        2. Flatten + pos embedding
        3. Mamba blocks x depth
        4. Normalization
        5. (추가) 프레임별로 묶어서 (B, T, D) 형태도 함께 얻기
        """
        # 1) Patch embed
        x = self.patch_embed(x)  # (B, D, T, n_PH, n_PW)

        B, D, T, n_PH, n_PW = x.shape

        # 2) Flatten + positional embedding
        x = x.permute(0, 2, 3, 4, 1)                # (B, T, n_PH, n_PW, D)
        x = x.reshape(B*T, n_PH*n_PW, D)            # (B*T, n_PH*n_PW, D)
        x = x + self.pos_embed

        # temporal pos
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)  # (B*n_PH*n_PW, T, D)
        x = x + self.temporal_pos_embedding                  # (B*n_PH*n_PW, T, D)
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)  # (B, T*n_PH*n_PW, D)

        x = self.pos_drop(x)  # (B, L, D),  여기서 L = T*n_PH*n_PW

        residual = None
        hidden_states = x

        # 3) Mamba blocks
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

            # 마지막 레이어면 normalization
            if idx == len(self.layers) - 1:
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

        sgn_mask_lst, valid_len_out_lst = [], []

        sgn = hidden_states.view(B, T, n_PH*n_PW, self.embed_dim)  # (B, T, nPatch, D)
        sgn = sgn.mean(dim=2) # (B, T, D)
        sgn_mask = torch.zeros([B, 1, T], dtype=torch.bool, device=sgn.device)
        valid_len_out = torch.floor(sgn_lengths / self.patch_embed.kernel_size).long() #B,
        for bi in range(B):
            sgn_mask[bi, :, :valid_len_out[bi]] = True
        sgn_mask_lst.append(sgn_mask)
        valid_len_out_lst.append(valid_len_out)

        return {
            'sgn_mask': sgn_mask_lst, 'valid_len_out': valid_len_out_lst, 'sgn':sgn
        }
    
    def decode(self, gloss_logits, beam_size, input_lengths):
        gloss_logits = gloss_logits.permute(1, 0, 2) #T,B,V
        gloss_logits = gloss_logits.cpu().detach().numpy()
        tf_gloss_logits = np.concatenate(
            (gloss_logits[:, :, 1:], gloss_logits[:, :, 0, None]),
            axis=-1,
        )
        decoded_gloss_sequences = ctc_decode_func(
            tf_gloss_logits=tf_gloss_logits,
            input_lengths=input_lengths,
            beam_size=beam_size
        )
        return decoded_gloss_sequences


# def load_temporal_pos_embedding(model, state_dict, new_frames):
#     # pretrained_temp: shape [1, 64, 192]
#     pretrained_temp = state_dict['temporal_pos_embedding']  # (1, T_old, D)
#     T_old = pretrained_temp.shape[1]

#     if T_old == new_frames:
#         # 프레임 수가 동일하다면 그냥 할당
#         model.temporal_pos_embedding.data = pretrained_temp
#         return model

#     # (1, T_old, D) -> (1, D, T_old)
#     pretrained_temp = pretrained_temp.permute(0, 2, 1)

#     # (1, D, T_old) -> 보간 -> (1, D, new_frames)
#     pretrained_temp = torch.nn.functional.interpolate(
#         pretrained_temp,
#         size=new_frames,
#         mode='linear',
#         align_corners=False
#     )
#     # 다시 (1, new_frames, D)로 변환
#     pretrained_temp = pretrained_temp.permute(0, 2, 1)

#     pretrained_temp = pretrained_temp.contiguous()

#     # 최종 할당
#     model.temporal_pos_embedding.data = pretrained_temp

#     return model


# def inflate_weight(weight_2d, time_dim, center=True):
#     print(f'Init center: {center}')
#     if center:
#         weight_3d = torch.zeros(*weight_2d.shape)
#         weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
#         middle_idx = time_dim // 2
#         weight_3d[:, :, middle_idx, :, :] = weight_2d
#     else:
#         weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
#         weight_3d = weight_3d / time_dim
#     return weight_3d

def inflate_weight(weight_2d, time_dim, center=True):
    # shape: [C_out, C_in, 1, kH, kW]
    # 원래 한 장(1)에 해당하는 차원을 time_dim으로 늘려 주기
    if center:
        # (C_out, C_in, time_dim, kH, kW) 형태의 0 텐서 만들기
        weight_3d = torch.zeros(
            weight_2d.size(0),
            weight_2d.size(1),
            time_dim,
            weight_2d.size(3),
            weight_2d.size(4),
            device=weight_2d.device,
            dtype=weight_2d.dtype
        )
        middle_idx = time_dim // 2
        # 기존 weight_2d는 시간 차원이 1뿐이므로, 여기에 그대로 할당
        weight_3d[:, :, middle_idx, :, :] = weight_2d[:, :, 0, :, :]
    else:
        # 단순 반복: (C_out, C_in, time_dim, kH, kW)
        weight_3d = weight_2d.repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    return weight_3d.contiguous()


# def load_state_dict(model, state_dict, center=True):
#     state_dict_3d = model.state_dict()
#     for k in state_dict.keys():
#         if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
#             if len(state_dict_3d[k].shape) <= 3:
#                 print(f'Ignore: {k}')
#                 continue
#             print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
#             time_dim = state_dict_3d[k].shape[2]
#             state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
#     del state_dict['head.weight']
#     del state_dict['head.bias']
#     del state_dict['pos_embed']
#     del state_dict['temporal_pos_embedding']
#     msg = model.load_state_dict(state_dict, strict=False)
#     print(msg)

# def load_state_dict(model, state_dict, center=True, n_frames=32):
#     # ----------------------------------------------------------------
#     # (1) pos_embed & temporal_pos_embedding 처리
#     # ----------------------------------------------------------------
#     if 'temporal_pos_embedding' in state_dict:
#         load_temporal_pos_embedding(model, state_dict, new_frames=n_frames)
#         del state_dict['temporal_pos_embedding']

#     # ----------------------------------------------------------------
#     # (2) patch_embed.proj.weight inflate (예: 2D -> 3D)
#     # ----------------------------------------------------------------
#     # 2-1) 모델이 가진 weight와 프리트레인 state_dict의 weight shape 확인
#     key = 'patch_embed.proj.weight'
#     if key in state_dict:
#         pretrained_w = state_dict[key]  # shape 예: [192, 3, 1, 16, 16]
#         model_w = model.state_dict()[key]  # shape 예: [192, 3, 4, 16, 16]

#         if pretrained_w.shape != model_w.shape:
#             # time_dim = model_w.shape[2], 현재 모델에서 필요로 하는 시간 차원
#             tdim = model_w.shape[2]
#             print(f'Inflate: {key}, {pretrained_w.shape} => {model_w.shape}')
#             new_weight = inflate_weight(pretrained_w, time_dim=tdim, center=center)
#             state_dict[key] = new_weight

#     # ----------------------------------------------------------------
#     # (3) head.weight, head.bias 등 필요 없다면 삭제
#     # ----------------------------------------------------------------
#     if 'head.weight' in state_dict:
#         del state_dict['head.weight']
#     if 'head.bias' in state_dict:
#         del state_dict['head.bias']

#     # ----------------------------------------------------------------
#     # (4) 최종 load
#     # ----------------------------------------------------------------
#     msg = model.load_state_dict(state_dict, strict=False)
#     print(msg)

def load_state_dict(model, state_dict, center=True):
    # ----------------------------------------------------------------
    # (1) pos_embed & temporal_pos_embedding 처리
    # ----------------------------------------------------------------
    key = 'temporal_pos_embedding'
    if key in state_dict:
        pretrained_temp = state_dict[key]  # (1, T_old, D)
        model_temp = model.state_dict()[key]  # (1, T_new, D)
        T_old, T_new = pretrained_temp.shape[1], model_temp.shape[1]

        if T_old != T_new:
            print(f'Interpolating {key}: {pretrained_temp.shape} => {model_temp.shape}')
            pretrained_temp = pretrained_temp.permute(0, 2, 1)  # (1, D, T_old)
            pretrained_temp = torch.nn.functional.interpolate(
                pretrained_temp, size=T_new, mode='linear', align_corners=False
            )
            pretrained_temp = pretrained_temp.permute(0, 2, 1)  # (1, T_new, D)
            state_dict[key] = pretrained_temp

    # ----------------------------------------------------------------
    # (2) patch_embed.proj.weight inflate (예: 2D -> 3D)
    # ----------------------------------------------------------------
    key = 'patch_embed.proj.weight'
    if key in state_dict:
        pretrained_w = state_dict[key]  # shape 예: [192, 3, 1, 16, 16]
        model_w = model.state_dict()[key]  # shape 예: [192, 3, 4, 16, 16]

        if pretrained_w.shape != model_w.shape:
            # time_dim = model_w.shape[2], 현재 모델에서 필요로 하는 시간 차원
            tdim = model_w.shape[2]
            print(f'Inflating {key}: {pretrained_w.shape} => {model_w.shape}')
            new_weight = inflate_weight(pretrained_w, time_dim=tdim, center=center)
            state_dict[key] = new_weight

    # ----------------------------------------------------------------
    # (3) head.weight, head.bias 등 필요 없다면 삭제
    # ----------------------------------------------------------------
    for key in ['head.weight', 'head.bias']:
        if key in state_dict:
            del state_dict[key]

    # ----------------------------------------------------------------
    # (4) 최종 load
    # ----------------------------------------------------------------
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        embed_dim=192,
        depth=24,
        # rms_norm=False,
        # fused_add_norm=False,
        **kwargs
    )

    if pretrained:
        print("Load pretrained params")
        state_dict = torch.load('/home/ubuntu/workspace/slr/data/pretrained/videomamba_t16_k400_f64_res224_sliced.pth', map_location='cpu')
        load_state_dict(model, state_dict)
    
    return model
    
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
        state_dict = torch.load('//home/ubuntu/workspace/slr/data/pretrained/videomamba_m16_in1k_res224.pth', map_location='cpu')
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
    model = videomamba_tiny(pretrained=True, n_frames=256, kernel_size=1, width=224, height=224).cuda()
    
    # import time
    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import flop_count_table

    # flops = FlopCountAnalysis(model, torch.rand(1, 3, 32, 720, 1280).cuda())
    # s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time() - s)
    print(model(torch.rand(1, 3, 256, 224, 224).cuda()).shape)
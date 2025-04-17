from functools import partial
import tensorflow as tf
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from dataclasses import dataclass
from alphaction.modeling.poolers import make_3d_pooler
from itertools import groupby


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


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                    num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        # B, C, T, H, W = x.shape
        x = self.proj(x)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=80,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches  # 8x14x14
        self.use_checkpoint = use_checkpoint
        self.grid_size = [img_size//patch_size, img_size//patch_size]  # [14,14]

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        # print(f"===========Number of Patches: {num_patches}\n \
        #       ===========Postional Embedding Shape: {self.pos_embed.shape} \n \
        #       =========== Grid Size: [{img_size}//{patch_size}, {img_size}//{patch_size}]")
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)  
        self.fc_norm = None

        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)

        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(f"Shape after patch embedding: {x.shape}")
        B, width, t, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)
        # print(f"Shape after flatten and transpose: {x.shape}")

        if self.pos_embed is not None:  
            # print(f"Position Embedding Shape: {self.pos_embed.shape}")
            pos_embed = self.pos_embed.reshape(t, -1, width)
            pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [h, w], 0).reshape(1, -1, width)
            x = x + pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_checkpoint:
            # print("Using gradient checkpointing.")
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, use_reentrant=True)
        else:   
            # print("Not using gradient checkpointing.")
            for blk in self.blocks:
                # print(f"[DEBUG]: Input shape: {x.shape}")
                x = blk(x)

        x = self.norm(x)  # [b thw=8x14x14 c=768]
        return x

    def forward(self, x, sgn_lengths):
        x = self.forward_features(x) # The output of the model (B, THW, C)
        # print(f" Before pooling and head Shape: {x.shape}")

        B, patches_per_video, C = x.shape
        T = patches_per_video // (self.grid_size[0] * self.grid_size[1]) # Recover frames count

        # (B, T, patches_per_frame, C)
        x = x.view(B,T,-1,C)

        # Global average pooling the patches per frame
        sgn = x.mean(dim=2)

        sgn_mask_lst, valid_len_out_lst = [], []

        # Calculate valid_len_out and mask
        valid_len_out = torch.floor(sgn_lengths / self.patch_embed.tubelet_size).long() # B,
        # print(f"======[DEBUG]===== valid_len_out: {valid_len_out}")
        
        sgn_mask = torch.zeros([B, 1, T], dtype=torch.bool, device=x.device)
        for bi in range(B):
            sgn_mask[bi, :, :valid_len_out[bi]] = True
        sgn_mask_lst.append(sgn_mask)
        valid_len_out_lst.append(valid_len_out)
        # x = x.mean(dim=1, keepdim=False)  # [b num_classes] -> Global Pooling
        # # print(f"Head input: {x.detach().cpu().numpy()}")
        # x = self.head(x)  # Global average pooling
        # print(f" After head Shape: {x.shape}")
        # print(f"======[DEBUG]===== valid_len_out: {valid_len_out}, sgn_mask: {sgn_mask.shape}")
        # Return outputs
        # print(f"====================Processed Frames (T) in forward: {T}")  # Number of frames
        return {
            'sgn': sgn,  # Features (B, T, D)
            'sgn_mask': sgn_mask_lst,  # Mask (B, 1, T)
            'valid_len_out': valid_len_out_lst # Valid lengths (B,)
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


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def interpolate_pos_embed_online(
    pos_embed, 
    orig_size: Tuple[int], 
    new_size: Tuple[int], 
    num_extra_tokens: int):

    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]
    embedding_size = pos_tokens.shape[-1]
    # print(f"Position Embedding Shape Before: {pos_tokens.shape}, {extra_tokens.shape}, {embedding_size}") 
    pos_tokens = pos_tokens.reshape(
        -1, orig_size[0], orig_size[1], embedding_size
    ).permute(0, 3, 1, 2)
    # print(f"Position Embedding Shape: {pos_tokens.shape}")
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=new_size, mode="bicubic", align_corners=False,
    )
    # print(f"Interpolated Position Embedding Shape: {pos_tokens.shape}")
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    # print(f"Flattened Position Embedding Shape: {pos_tokens.shape}")
    # print(f"Extra Tokens Shape: {extra_tokens.shape}")
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    return new_pos_embed


if __name__ == '__main__':
    from run import get_args
    import utils
    from collections import OrderedDict


    args, _ = get_args()
    model = vit_small_patch16_224(
                    num_classes=args.n_classes,            # Number of classes for the classification head
                    all_frames=args.num_frames,  # Total number of frames
                    # all_frames=args.n_frames * args.num_segments,  # Total number of frames
                    tubelet_size=args.tubelet_size,         # Tubelet size
                    drop_path_rate=args.drop_path_rate,          # Stochastic depth rate
                    use_checkpoint=args.use_checkpoint,     # Gradient checkpointing
                    use_mean_pooling=args.use_mean_pooling, # Use mean pooling or CLS token
    )
    # B, C, T, H, W
    # print(f"Output Shape: {model(torch.randn(2, 3, 128, 224, 224), 128).shape}")

    if args.pretrained_model_checkpoint_path:
        if args.pretrained_model_checkpoint_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained_model_checkpoint_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained_model_checkpoint_path, map_location='cpu')

        print("Load ckpt from %s" % args.pretrained_model_checkpoint_path)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:  #
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # for keys, param in checkpoint_model.items():
        #     print(keys, param.shape)


        # print(f"checkpoint_model: {checkpoint_model}")
        
    # utils.load_state_dict(model.video_backbone, checkpoint_model, prefix='')
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        print("Initializing the classification head weights.")
        nn.init.xavier_uniform_(model.head.weight)
        nn.init.constant_(model.head.bias, 0)



    # # Check if the pretrained checkpoint model has the necessary attributes
    # print("\n==== Pretrained Model Configuration ====")

    # # Check the embedding dimension
    # if "patch_embed.proj.weight" in checkpoint_model:
    #     embed_dim = checkpoint_model["patch_embed.proj.weight"].shape[0]  # First dim = embedding size
    #     print(f"Embedding Dimension: {embed_dim}")

    # # Check the number of attention heads (using block 0 as reference)
    # for key in checkpoint_model.keys():
    #     if ".attn.qkv.weight" in key:  
    #         attn_dim = checkpoint_model[key].shape[0]  # This should be `num_heads * head_dim * 3`
    #         num_heads = attn_dim // 3 // embed_dim  # Divide by 3 for q, k, v, and head_dim
    #         print(f"Number of Attention Heads: {num_heads}")
    #         break  # We only need to check this once

    # # Dropout rates (not always stored explicitly in checkpoint, may need default assumption)
    # drop_rate = 0.1  # Default, unless explicitly stored
    # drop_path_rate = 0.1  # Default, but can be searched for
    # if "blocks.0.attn.proj.weight" in checkpoint_model:
    #     print(f"Drop Rate: {drop_rate} (Check model definition for exact value)")
    #     print(f"Drop Path Rate: {drop_path_rate} (Check model definition for exact value)")

    # # MLP ratio (determined from shape of MLP layers)
    # for key in checkpoint_model.keys():
    #     if ".mlp.fc1.weight" in key:  
    #         mlp_ratio = checkpoint_model[key].shape[0] // embed_dim  # fc1 has shape (hidden_dim, embed_dim)
    #         print(f"MLP Ratio: {mlp_ratio}")
    #         break

    # # Number of Transformer blocks (depth of the model)
    # num_blocks = sum(1 for key in checkpoint_model if "blocks." in key and ".attn.qkv.weight" in key)
    # print(f"Number of Transformer Blocks (Depth): {num_blocks}")

    # # Normalization layers
    # if "norm.weight" in checkpoint_model:
    #     print("Layer Normalization Applied Before Final Projection")

    # print("========================================\n")

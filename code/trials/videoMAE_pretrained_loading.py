import torch
import sys
import os
from collections import OrderedDict

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from videoMae_small import VisionTransformer

# Simulate a new input configuration
new_img_size = 384  # Change spatial dimension
new_num_frames = 256  # Change the number of frames


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

# Initialize the fine-tuning model
model = VisionTransformer(
    img_size=384,
    patch_size=16,
    num_classes=100,  # Adjust as per your dataset
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.,
    qkv_bias=True,
    use_checkpoint=False,
    all_frames=256
)

# Load pre-trained weights
checkpoint_path = "/nas//Dataset/Phoenix/pretrained/Nirmal/videoMAE_vit-s_16x224x224_checkpoint_all_configs.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Strip 'module.' prefix if needed
# model_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["module"].items()}
checkpoint_model = None
for model_key in 'model|module'.split('|'):
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


# interpolate position embedding
if 'pos_embed' in checkpoint_model:
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
    num_patches = model.patch_embed.num_patches  #
    print(f"============ Number of patches: {num_patches}")
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

    # height (== width) for the checkpoint position embedding 
    orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                256 // model.patch_embed.tubelet_size)) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int((num_patches // (256 // model.patch_embed.tubelet_size)) ** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, 256 // model.patch_embed.tubelet_size, orig_size,
                                        orig_size, embedding_size)
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1,
                                                            256 // model.patch_embed.tubelet_size,
                                                            new_size, new_size, embedding_size)
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

load_state_dict(model, checkpoint_model, prefix='')


# Remove classification head weights
if 'head.weight' in checkpoint_model:
    del checkpoint_model['head.weight']
if 'head.bias' in checkpoint_model:
    del checkpoint_model['head.bias']

# Load into model
model.load_state_dict(checkpoint_model, strict=False)
# print("Missing keys:", missing_keys)
# print("Unexpected keys:", unexpected_keys)

# Fine-tune model
model.to("cuda")
print("Interpolated positional embedding shape:", model.pos_embed.shape)

# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}\n")
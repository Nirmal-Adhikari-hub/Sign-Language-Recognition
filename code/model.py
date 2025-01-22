import torch
import torch.nn as nn
from head import VisualHead
from videoMae_small import vit_small_patch16_224


class RGBModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.video_backbone = vit_small_patch16_224(
                    num_classes=args.n_classes,            # Number of classes for the classification head
                    all_frames=args.num_frames,  # Total number of frames
                    # all_frames=args.n_frames * args.num_segments,  # Total number of frames
                    tubelet_size=args.tubelet_size,         # Tubelet size
                    drop_path_rate=args.drop_path_rate,          # Stochastic depth rate
                    use_checkpoint=args.use_checkpoint,     # Gradient checkpointing
                    use_mean_pooling=args.use_mean_pooling, # Use mean pooling or CLS token
                    )
        self.visual_head = VisualHead(
                        cls_num=args.n_classes,
                        input_size=args.embed_dim,
                        hidden_size=512,
                        ff_size=2048,
                        pe=False,
                        head_drop_rate=args.head_drop_rate,
                        ff_kernelsize=3,
                        pretrained_ckpt=None,
                        is_empty=False,
                        frozen=False,
                        plus_conv_cfg={},
                        ssl_projection_cfg={}
        )

    def forward(self, x, sgn_lenghts):
        """
        x: Input tensor of shape: (B, C, T, H, W)
        sgn_lengths: Frame counts in each videos of a batch
        """

        # Backbone Output
        backbone_out = self.video_backbone(x, sgn_lenghts)

        # Extract features and valid lengths
        # sgn = backbone_output['sgn'].permute(0, 2, 1)
        sgn = backbone_out['sgn']
        sgn_mask = backbone_out['sgn_mask'][-1]
        valid_len_in = backbone_out['valid_len_out'][-1]

        # Head processing
        head_output = self.visual_head(sgn, sgn_mask, valid_len_in)

        return {
            'gloss_logits': head_output['gloss_logits'],
            'gloss_probabilities_log': head_output['gloss_probabilities_log'],
            'valid_len_out': head_output['valid_len_out']
        }




def build_model(args):
    """
    Builds the VideoMAE model based on the arguments provided.
    Args:
        args: Argument parser object containing model configurations.
    Returns:
        model: An instance of the VideoMAE model.
    """

    model = RGBModel(args)

    return model


# def load_pretrained_weights(model, checkpoint_path, detach_head=True):
#     """
#     Load pretrained weights into the model and initialize the classification head.
    
#     Args:
#         model (torch.nn.Module): The model into which the weights will be loaded.
#         checkpoint_path (str): Path to the pretrained checkpoint.
#         detach_head (bool): Whether to remove the classification head weights from the checkpoint.
    
#     Returns:
#         torch.nn.Module: The model with the pretrained weights loaded.
#     """
#     print(f"Loading pretrained weights from: {checkpoint_path}")

#     # Load checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')

#     # Extract model state dict, strip 'module.' prefix if needed
#     checkpoint_model = checkpoint
#     if "module" in checkpoint_model:
#         checkpoint_model = checkpoint_model["module"]

#     # Remove classification head weights if required
#     if detach_head:
#         for key in ["head.weight", "head.bias"]:
#             if key in checkpoint_model:
#                 del checkpoint_model[key]
#                 print(f"Removed key {key} from pretrained checkpoint.")

#     # Load the state dict into the model
#     missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
#     if missing_keys:
#         print(f"Missing keys when loading pretrained weights: {missing_keys}")
#     if unexpected_keys:
#         print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

#     # # Initialize classification head
#     # if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
#     #     print("Initializing the classification head weights.")
#     #     nn.init.xavier_uniform_(model.head.weight)
#     #     nn.init.constant_(model.head.bias, 0)

#     print("Successfully loaded pretrained weights.")
        
#     return model
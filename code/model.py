import torch
import torch.nn as nn
from head import VisualHead
from videoMae_small import vit_small_patch16_224
from argparse import Namespace


class RGBModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.video_backbone = vit_small_patch16_224(
                    embed_dim=args.embed_dim,            # Embedding dimension

                    num_classes=args.n_classes,            # Number of classes for the classification head
                    all_frames=args.num_frames,  # Total number of frames
                    # all_frames=args.n_frames * args.num_segments,  # Total number of frames
                    tubelet_size=args.tubelet_size,         # Tubelet size
                    drop_path_rate=args.drop_path_rate,          # Stochastic depth rate
                    use_checkpoint=args.use_checkpoint,     # Gradient checkpointing
                    use_mean_pooling=args.use_mean_pooling, # Use mean pooling or CLS token
                    pretrained=False,
                    drop_rate=args.drop_rate,
                    attn_drop_rate=args.attn_drop_rate,
                    # num_classes=args.nb_classes,
                    # all_frames=args.num_frames * args.num_segments,
                    # tubelet_size=args.tubelet_size,
                    # drop_rate=args.drop,
                    # drop_path_rate=args.drop_path,
                    # attn_drop_rate=args.attn_drop_rate,
                    # drop_block_rate=None,
                    # use_checkpoint=args.use_checkpoint,
                    # use_mean_pooling=args.use_mean_pooling,
                    # init_scale=args.init_scale,
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

if __name__ == '__main__':
    def test_full_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a dummy args object with minimal parameters required by both the backbone and head.
        # Adjust these parameters if needed.
        dummy_args = Namespace(
            embed_dim=192,          # For video_backbone (e.g., embedding dimension)
            n_classes=1235,         # Number of output classes
            num_frames=64,          # Total number of frames per video
            tubelet_size=2,         # Tubelet size
            drop_rate=0.2,          # Dropout rate for the backbone
            attn_drop_rate=0.1,     # Attention dropout rate
            drop_path_rate=0.1,     # Stochastic depth rate
            use_checkpoint=False,   # Whether to use gradient checkpointing
            use_mean_pooling=True,  # Whether to use mean pooling instead of a CLS token
            patch_size=[16, 16],    # Patch size for the backbone
            in_chans=3,             # Number of input channels
            
            # Head-specific parameters:
            head_drop_rate=0.1,     # Dropout for the head
            ff_size=2048,           # Feed-forward intermediate dimension
            ff_kernelsize=3,        # Kernel size for the position-wise feedforward
            pe=False,               # Whether to use positional encoding
            pretrained_model_checkpoint_path="",  # No pretrained checkpoint for testing
            
            # Optional configurations for plus_conv and ssl_projection can be left as their default {}
            # If you have additional parameters expected by build_model, include them here.
        )
        
        # Build the full model using your build_model function.
        model = build_model(dummy_args)
        model.to(device)
        
        # Create dummy input data:
        # A batch of 2 videos, 3 channels, 64 frames, 224x224 spatial resolution.
        dummy_input = torch.randn(2, 3, 64, 224, 224, requires_grad=True).to(device)
        
        # Create a dummy lengths tensor (one length per sample in the batch)
        dummy_lengths = torch.tensor([64, 64]).to(device)
        
        # Run the forward pass through the full model.
        # The model's forward signature is: forward(self, x, sgn_lenghts)
        output = model(dummy_input, dummy_lengths)
        # The output dictionary contains keys such as 'gloss_logits', 'gloss_probabilities_log', and 'valid_len_out'.
        
        # Compute a simple scalar loss from the output; for example, sum all elements in 'gloss_logits'.
        loss = output['gloss_logits'].sum()
        
        # Backward pass to compute gradients.
        loss.backward()
        
        # Print out the gradient norm for each parameter in the full model.
        print("Gradients for full model parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm() if param.grad is not None else None
                print(f"{name}: grad norm = {grad_norm}")

    print("Testing full model (video_backbone + head) gradient flow:")
    test_full_model()
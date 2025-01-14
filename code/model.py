import torch
import torch.nn as nn
from videoMae_small import vit_small_patch16_224


def build_model(args):
    """
    Builds the VideoMAE model based on the arguments provided.
    Args:
        args: Argument parser object containing model configurations.
    Returns:
        model: An instance of the VideoMAE model.
    """

    model = vit_small_patch16_224(
    num_classes=args.n_classes,            # Number of classes for the classification head
    all_frames=args.num_frames,  # Total number of frames
    # all_frames=args.n_frames * args.num_segments,  # Total number of frames
    tubelet_size=args.tubelet_size,         # Tubelet size
    drop_path_rate=args.drop_path_rate,          # Stochastic depth rate
    use_checkpoint=args.use_checkpoint,     # Gradient checkpointing
    use_mean_pooling=args.use_mean_pooling, # Use mean pooling or CLS token
    )

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
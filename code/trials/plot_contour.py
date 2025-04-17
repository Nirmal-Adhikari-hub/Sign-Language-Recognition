# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import argparse
# from model import build_model
# from datasets.dataset import build_dataset
# from utils import load_state_dict
# from run import get_args
# from datasets.tokenizer import GlossTokenizer_S2G
# from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
# import utils
# from functools import partial
# import torch.nn as nn
# from torchmetrics.text import WordErrorRate
# from datasets.phoenix_cleanup import clean_phoenix_2014




# def compute_loss(model, gloss_tokenizer, criterion, dataloader, device):
#     """
#     Computes the loss for the model on the dataset

#     Args:
#     model (torch.nn.Moduel): Trained model
#     criterion (torch.nn.Module): Loss function
#     dataloader (torch.utils.data.DataLoader): Validation dataloader
#     device (torch.device): Device to run the model on

#     Returns:
#     float: Average loss over the dataset.
#     """
#     model.eval()
#     total_loss = 0.0
#     num_batches = 0
#     wer = WordErrorRate(sync_on_compute=False)

#     with torch.no_grad():
#         for frames, frames_length, targets, targets_length, name in dataloader:
#             frames, frames_length = frames.to(device), frames_length.to(device)
#             targets, targets_length = targets.to(device), targets_length.to(device)
#             outputs = model(frames, frames_length)
#             loss = criterion(
#                 log_probs=outputs['gloss_probabilities_log'].permute(1, 0, 2),
#                 targets=targets,
#                 input_lengths=outputs['valid_len_out'],
#                 target_lengths=targets_length
#             )
#             decoded_gloss_ids = model.video_backbone.decode(
#             gloss_logits=outputs['gloss_logits'], beam_size=1, input_lengths=outputs['valid_len_out'])
            
#             decoded_glosses = [
#             clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(decoded_gloss_id)))
#             for decoded_gloss_id in decoded_gloss_ids
#             ]
#             target_glosses = [
#                 clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(target[:tgt_len].tolist())))
#                 for target, tgt_len in zip(targets, targets_length)
#             ]

#             wer.update(decoded_glosses, target_glosses)

#             # total_loss += loss.item()
#             # num_batches += 1

#         return wer.compute().item()
    

# def extract_multiple_weight_snapshots(model, num_snapshots=5, perturb_scale=0.02):
#     """
#     Extract multiple weight snapshots by perturbing model parameters.

#     Args:
#         model (torch.nn.Module): Trained model.
#         num_snapshots (int): Number of weight snapshots to generate.
#         perturb_scale (float): Scale of perturbations applied to weights.

#     Returns:
#         np.ndarray: Collection of weight snapshots.
#     """
#     all_weights = []
    
#     # Get original model weights as a vector
#     base_weights = []
#     for param in model.parameters():
#         base_weights.append(param.data.view(-1))
#     base_weights = torch.cat(base_weights).cpu().numpy()

#     all_weights.append(base_weights)  # Include original weights

#     # Generate slightly perturbed versions of weights
#     for _ in range(num_snapshots - 1):  # We already have 1 original
#         perturbed_weights = base_weights + np.random.normal(0, perturb_scale, base_weights.shape)
#         all_weights.append(perturbed_weights)

#     return np.array(all_weights)

    

# def plot_3d_loss_surface(model, gloss_tokenizer, criterion, dataloader, device, output_path="3d_plot.png"):
#     """
#      Generates and saves a 3D loss surface plot by perturbing model weights in 2D space.

#     Args:
#         model (torch.nn.Module): The trained model.
#         criterion (torch.nn.Module): Loss function used during training.
#         dataloader (torch.utils.data.DataLoader): A dataloader for evaluation.
#         device (str): The device (CPU/GPU) for evaluation.
#         output_path (str): The path to save the contour plot.   
#     """

#     model.eval()

#     # Extract real weight variations
#     weights_snapshots = extract_multiple_weight_snapshots(model, num_snapshots=10)

#     # Apply PCA to reduce weight space to 2D
#     pca = PCA(n_components=2)
#     weights_projections = pca.fit_transform(weights_snapshots)

#     # Define a 2D grid for weight perturbations
#     x_vals = np.linspace(-1,1,20)
#     y_vals = np.linspace(-1,1,20)
#     X, Y = np.meshgrid(x_vals, y_vals)

#     # Save original weights before modification
#     original_weights = {name: param.clone() for name, param in model.named_parameters()}

#     # Compute loss at each grid point
#     loss_values = np.zeros_like(X)

#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             perturbed_weights = weights_projections[0] + np.array([X[i,j], Y[i,j]])
#             perturbed_weights = pca.inverse_transform(perturbed_weights)

#             # Update the model weights
#             index = 0
#             with torch.no_grad():
#                 for name, param in model.named_parameters():
#                     param_size = param.numel()
#                     param.copy_(torch.tensor(perturbed_weights[index:index+param_size]).view(param.shape))
#                     index += param_size

#             # Compute loss on validation set
#             loss_values[i, j] = compute_loss(model, gloss_tokenizer, criterion, dataloader, device)

#     # Restore original weights
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             param.copy_(original_weights[name])

#     # Plot the 3D loss surface
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, loss_values, cmap='coolwarm')

#     # Labels
#     ax.set_xlabel("Weight direction 1")
#     ax.set_ylabel("Weight direction 2")
#     ax.set_zlabel("Loss")
#     plt.title("3D Loss Landscape")

#     # Save the plot
#     plt.savefig(output_path)
#     plt.close()
#     print(f"3D Loss surface plot saved at: {output_path}")


# def main():
#     args, _ = get_args()

#     # Load model
#     device = torch.device(args.device)
#     model = build_model(args)
#     model.to(device)

#     # Load checkpoint
#     checkpoint = torch.load("/nas/Nirmal/workspace/slr_results/2videomae_192_embed_dim_64_num_frames/checkpoint-latest/mp_rank_00_model_states.pt", map_location=device)
    
#     # Find the correct key in checkpoint
#     checkpoint_key = 'model' if 'model' in checkpoint else 'module' if 'module' in checkpoint else None

#     if checkpoint_key is None:
#         raise KeyError("❌ Error: Could not find 'model' or 'module' in checkpoint. Available keys: {}".format(checkpoint.keys()))

#     print(f"✅ Loaded weights from checkpoint key: {checkpoint_key}")
#     load_state_dict(model, checkpoint[checkpoint_key])

#     # Setting the tokenizers for dataloader
#     tokenizer_cfg = {
#             'gloss2id_file': args.gloss_to_id_path,
#         }
#     gloss_tokenizer = GlossTokenizer_S2G(tokenizer_cfg)

#     # Load dataset
#     n_tasks = utils.get_world_size()
#     global_rank = utils.get_rank()
#     collate = partial(utils.collate_fn, gloss_tokenizer=gloss_tokenizer)
#     val_dataset = build_dataset(modal=args.modal,
#                                     gloss_tokenizer=gloss_tokenizer,
#                                     is_train=False,
#                                     is_test=False,
#                                     args=args)
    
#     if args.dist_eval: # Evaluation using DDP
#         if len(val_dataset) % n_tasks != 0:
#             print('Warning: Enabling distributed evaluation with an eval dataset not divisible by\
#                 process number.') if utils.is_main_process() else None
#         val_sampler = DistributedSampler(val_dataset,n_tasks, global_rank, shuffle=False)

#     else: # Sampling sequentially starting from index 0
#         val_sampler = SequentialSampler(val_dataset)
    
#     val_dataloader = DataLoader(
#                 val_dataset, 
#                 sampler=val_sampler, 
#                 batch_size=args.batch_size, 
#                 num_workers=args.num_workers,
#                 pin_memory=args.pin_mem, 
#                 drop_last=False, 
#                 collate_fn=collate,
#                 prefetch_factor=2,  # Similar prefetching strategy for validation
#                 persistent_workers=False  # Set False for consistency
#             )
    
#     # Define loss function
#     criterion = nn.CTCLoss(blank=gloss_tokenizer.silence_id, zero_infinity=True)

#     # Generate and save the 3D loss surface plot
#     plot_3d_loss_surface(model, gloss_tokenizer, criterion, val_dataloader, device, output_path="3d_plot.png")


# if __name__ == '__main__':
#     main()




import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
from model import build_model
from datasets.dataset import build_dataset
from utils import load_state_dict
from run import get_args
from datasets.tokenizer import GlossTokenizer_S2G
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
import utils
from functools import partial
import torch.nn as nn
from torchmetrics.text import WordErrorRate
from datasets.phoenix_cleanup import clean_phoenix_2014


def compute_loss(model, gloss_tokenizer, criterion, dataloader, device):
    """
    Computes the Word Error Rate (WER) for the model on the dataset.
    """
    print("🔄 Starting WER computation...")
    model.eval()
    wer = WordErrorRate(sync_on_compute=False)

    with torch.no_grad():
        for batch_idx, (frames, frames_length, targets, targets_length, name) in enumerate(dataloader):
            print(f"   ▶ Processing batch {batch_idx + 1}/{len(dataloader)}...")
            frames, frames_length = frames.to(device), frames_length.to(device)
            targets, targets_length = targets.to(device), targets_length.to(device)
            outputs = model(frames, frames_length)

            decoded_gloss_ids = model.video_backbone.decode(
                gloss_logits=outputs['gloss_logits'], beam_size=1, input_lengths=outputs['valid_len_out']
            )
            
            decoded_glosses = [
                clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(decoded_gloss_id)))
                for decoded_gloss_id in decoded_gloss_ids
            ]
            target_glosses = [
                clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(target[:tgt_len].tolist())))
                for target, tgt_len in zip(targets, targets_length)
            ]

            wer.update(decoded_glosses, target_glosses)

    final_wer = wer.compute().item()
    print(f"✅ Finished WER computation. Final WER: {final_wer:.4f}")
    return final_wer


def extract_multiple_weight_snapshots(model, num_snapshots=5, perturb_scale=0.02):
    """
    Extract multiple weight snapshots by perturbing model parameters.
    """
    print(f"🔄 Extracting {num_snapshots} weight snapshots with perturbation scale {perturb_scale}...")
    all_weights = []
    
    base_weights = []
    for param in model.parameters():
        base_weights.append(param.data.view(-1))
    base_weights = torch.cat(base_weights).cpu().numpy()

    all_weights.append(base_weights)

    for i in range(num_snapshots - 1):
        perturbed_weights = base_weights + np.random.normal(0, perturb_scale, base_weights.shape)
        all_weights.append(perturbed_weights)
        print(f"   ▶ Snapshot {i + 2}/{num_snapshots} generated.")

    print("✅ Weight snapshots extraction complete.")
    return np.array(all_weights)


def plot_3d_loss_surface(model, gloss_tokenizer, criterion, dataloader, device, output_path="3d_plot.png"):
    """
    Generates and saves a 3D WER surface plot.
    """
    print("🚀 Starting 3D loss surface generation...")
    model.eval()

    # Extract weight snapshots
    weights_snapshots = extract_multiple_weight_snapshots(model, num_snapshots=10)
    
    # Apply PCA
    print("🔄 Applying PCA to reduce weight space to 2D...")
    pca = PCA(n_components=2)
    weights_projections = pca.fit_transform(weights_snapshots)
    print("✅ PCA transformation complete.")

    # Define grid
    x_vals = np.linspace(-1, 1, 20)
    y_vals = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Save original weights before modification
    print("🔄 Saving original model weights...")
    original_weights = {name: param.clone() for name, param in model.named_parameters()}
    print("✅ Original weights saved.")

    # Compute WER at each grid point
    wer_values = np.zeros_like(X)

    print("🔄 Computing WER values for perturbed weight configurations...")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            perturbed_weights = weights_projections[0] + np.array([X[i, j], Y[i, j]])
            perturbed_weights = pca.inverse_transform(perturbed_weights)

            index = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param_size = param.numel()
                    param.copy_(torch.tensor(perturbed_weights[index:index+param_size]).view(param.shape))
                    index += param_size

            wer_values[i, j] = compute_loss(model, gloss_tokenizer, criterion, dataloader, device)
            print(f"   ▶ Grid ({i+1}/{X.shape[0]}, {j+1}/{X.shape[1]}): WER = {wer_values[i, j]:.4f}")

    # Restore original weights
    print("🔄 Restoring original model weights...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(original_weights[name])
    print("✅ Original model weights restored.")

    # Plot the 3D WER surface
    print("🔄 Generating 3D plot...")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, wer_values, cmap='coolwarm')

    ax.set_xlabel("Weight direction 1")
    ax.set_ylabel("Weight direction 2")
    ax.set_zlabel("WER")
    plt.title("3D WER Landscape")

    plt.savefig(output_path)
    plt.close()
    print(f"✅ 3D Loss surface plot saved at: {output_path}")


def main():
    print("🚀 Starting script execution...")
    args, _ = get_args()

    # Load model
    print("🔄 Loading model...")
    device = torch.device('cpu')
    model = build_model(args)
    model.to(device)
    print("✅ Model loaded successfully.")

    # Load checkpoint
    print("🔄 Loading checkpoint...")
    checkpoint = torch.load("/nas/Nirmal/workspace/slr_results/2videomae_192_embed_dim_128_num_frames/checkpoint-best/mp_rank_00_model_states.pt", map_location=device)
    
    # Find the correct key in checkpoint
    checkpoint_key = 'model' if 'model' in checkpoint else 'module' if 'module' in checkpoint else None
    if checkpoint_key is None:
        raise KeyError(f"❌ Error: Could not find 'model' or 'module' in checkpoint. Available keys: {checkpoint.keys()}")

    print(f"✅ Loaded weights from checkpoint key: {checkpoint_key}")
    load_state_dict(model, checkpoint[checkpoint_key])

    # Setting the tokenizers for dataloader
    print("🔄 Initializing gloss tokenizer...")
    tokenizer_cfg = {'gloss2id_file': args.gloss_to_id_path}
    gloss_tokenizer = GlossTokenizer_S2G(tokenizer_cfg)
    print("✅ Gloss tokenizer initialized.")

    # Load dataset
    print("🔄 Loading dataset...")
    n_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    collate = partial(utils.collate_fn, gloss_tokenizer=gloss_tokenizer)
    
    val_dataset = build_dataset(modal=args.modal, gloss_tokenizer=gloss_tokenizer, is_train=False, is_test=False, args=args)
    val_sampler = DistributedSampler(val_dataset, n_tasks, global_rank, shuffle=False) if args.dist_eval else SequentialSampler(val_dataset)

    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False, collate_fn=collate, prefetch_factor=2, persistent_workers=False
    )
    print("✅ Dataset loaded successfully.")

    # Define loss function
    criterion = nn.CTCLoss(blank=gloss_tokenizer.silence_id, zero_infinity=True)

    # Generate and save the 3D loss surface plot
    plot_3d_loss_surface(model, gloss_tokenizer, criterion, val_dataloader, device, output_path="3d_plot.png")


if __name__ == '__main__':
    main()

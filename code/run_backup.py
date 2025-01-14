import argparse
import datetime
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import torch.utils
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from functools import partial
from pathlib import Path
from transformers import AutoTokenizer

from timm.utils import ModelEma
from optim_factory import get_parameter_groups, LayerDecayValueAssigner

# from datasets.dataset import build_dataset
from datasets.tokenizer import GlossTokenizer_S2G
from engines.engine import train_one_epoch, validation_one_epoch, final_test
# from videomamba import videomamba_base
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import collate_fn
from model import build_model, load_pretrained_weights
from collections import OrderedDict
import utils
import contextlib


def get_args():
    parser = argparse.ArgumentParser('Visual Bacbone Training ad Evaluation for Continuous SLR.')
    parser.add_argument('--task', default='s2g', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1,type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model Params
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--patch_size', default=[2,4,4], type=int, nargs='+')
    parser.add_argument('--in_chans', default=3, type=int)
    parser.add_argument('--embed_dim', default=96, type=int)
    parser.add_argument('--depths', default=[3,3,3,3], type=int, nargs='+')
    parser.add_argument('--num_heads', default=[12,12,12,12], type=int, nargs='+')
    parser.add_argument('--tubelet_size', type=int, default=2)
    # parser.add_argument('--window_size')
    parser.add_argument('--drop_rate', default=0.1, type=float)
    parser.add_argument('--attn_drop_rate', default=0.1, type=float)
    parser.add_argument('--head_drop_rate', default=0.1, type=float)
    parser.add_argument('--drop_path_rate', default=0.1, type=float)
    parser.add_argument('--disable_eval_during_training', default=False, action='store_true')
    parser.add_argument('--model_ema', default=False, action='store_true')
    parser.add_argument('--model_ema_decay', default=.9999, type=float)
    parser.add_argument('--model_ema_force_cpu', default=False, action='store_true')

    # Optimizer Params
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+')
    parser.add_argument('--clip_grad', default=None, type=float) # Max Norm
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=.05, type=float)
    parser.add_argument('--weight_decay_start', default=None, type=float)
    parser.add_argument('--weight_decay_end', default=None, type=float)
    parser.add_argument('--lr', default=1e-3, type=float) # Maximum LR
    # parser.add_argument('--layer_decay', default=.75, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float) # Minimum LR
    parser.add_argument('--warmup_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--T_0', default=50, type=int)
    parser.add_argument('--T_mul', default=1, type=int)
    parser.add_argument('--lr_gamma', default=0.5, type=float)

    # Augmentation Params
    parser.add_argument('--aug_size', default=[210,210], type=int, nargs='+')
    parser.add_argument('--t_min', default=.5, type=float)
    parser.add_argument('--t_max', default=1.5, type=float)
    parser.add_argument('--color_jitter', default=.4, type=float)
    parser.add_argument('--aa', default=None) # Auto Augment Policy
    parser.add_argument('--train_interpolation', default='bicubic', type=str)

    # Random Erasing Params
    parser.add_argument('--reprob', default=0., type=float)
    parser.add_argument('--remode', default='pixel', type=str)
    parser.add_argument('--recount', default=1, type=int)

    # Dataset Params
    parser.add_argument('--modal', default='video', type=str)
    parser.add_argument('--video_size', default=[224,224], type=int, nargs='+')
    parser.add_argument('--metadata_path', default='', type=str)
    parser.add_argument('--gloss_to_id_path', default='/nas/Dataset/Phoenix/gloss2ids.pkl', type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--keypoint_path', default='', type=str)
    parser.add_argument('--split', default=',', type=str)
    parser.add_argument('--n_classes', default=1235, type=int)
    parser.add_argument('--num_frames', default=256, type=int)
    parser.add_argument('--dataset', default='phoenix-2014', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--log_dir', default='', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=str)
    parser.add_argument('--resume', default='', type=str) # Weight Path during re-learning
    parser.add_argument('--save_ckpt', default=False, action='store_true')
    parser.add_argument('--auto_resume', default=False, action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--test_best', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--dist_eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--no_amp', default=False, type=bool)

    # DDP Params
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--enable_deepspeed', default=True)
    parser.add_argument('--bf16', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true') # distributed training on interactive platform
    parser.add_argument('--dist_url', default='env://')

    # Finetuning
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--use_pretrained', default=True, type=bool)
    parser.add_argument('--pretrained_model_checkpoint_path', default='/nas//Dataset/Phoenix/pretrained/Nirmal/videoMAE_vit-s_16x224x224_checkpoint.pth', type=str)
    parser.add_argument('--use_mean_pooling', action='store_true') # cls_token
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed

            parser = deepspeed.add_config_arguments(parser) # Initialize deepspeed environment and wrap with deepspeed parser
            ds_init = deepspeed.initialize
        except:
            print("Please install deepspeed")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    torch.cuda.empty_cache()

    utils.init_distributed_mode(args)
    # print("==========A============")
    # Deepspeed config.json  generation
    if ds_init != None:
        utils.create_ds_config(args)
    # print("==========B============")

    print(args)

    device = torch.device(args.device)

    # Seed for each process for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed) # for PyTorch
    np.random.seed(seed) # for Numpy

    cudnn.benchmark = True #  https://medium.com/@adhikareen/why-and-when-to-use-cudnn-benchmark-true-in-pytorch-training-f2700bf34289

    if args.task == 's2g':
        tokenizer_cfg = {
            'gloss2id_file': args.gloss_to_id_path,
        }
        gloss_tokenizer = GlossTokenizer_S2G(tokenizer_cfg)

    assert args.n_classes == len(gloss_tokenizer), f"{args.n_classes, len(gloss_tokenizer)}"

    # # Train dataset
    # train_dataset = build_dataset(modal=args.modal, 
    #                               gloss_tokenizer=gloss_tokenizer,
    #                               is_train=True,
    #                               is_test=False,
    #                               args=args)
    
    # # Using validation during training or not
    # if args.disable_eval_during_training:
    #     val_dataset = None
    # else:
    #     val_dataset = build_dataset(modal=args.modal,
    #                                 gloss_tokenizer=gloss_tokenizer,
    #                                 is_train=False,
    #                                 is_test=False,
    #                                 args=args)
    # test_dataset = build_dataset(modal=args.modal,
    #                              gloss_tokenizer=gloss_tokenizer,
    #                              is_train=False,
    #                              is_test=True,
    #                              args=args)
    
    n_tasks = utils.get_world_size() # Total GPU processes
    global_rank = utils.get_rank() # process idx

    print(f"n_tasks: {n_tasks}, global_rank: {global_rank}")
    # train_sampler = DistributedSampler(train_dataset, n_tasks, global_rank, True)

    # if args.dist_eval: # Evaluation using DDP
    #     if len(val_dataset) % n_tasks != 0:
    #         print('Warning: Enabling distributed evaluation with an eval dataset not divisible by\
    #               process number.')
    #     val_sampler = DistributedSampler(val_dataset,n_tasks, global_rank, shuffle=False)
    #     test_sampler = DistributedSampler(test_dataset, n_tasks, global_rank, shuffle=False)
    # else: # Sampling sequentially starting from index 0
    #     val_sampler = SequentialSampler(val_dataset)
    #     test_sampler = SequentialSampler(test_dataset)

    # # Logging by master process only
    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    # # If the num of aug is greater than 1, use multiple_samples_collate with collate_fn
    # # Separate video tensor, label, video_id etc
    # '''Always remember that collate does all the organizations for a batch like merging, padding
    # making samples of consistent sizes, transformations, preprocessing , etc'''
    # collate = partial(collate_fn, gloss_tokenizer=gloss_tokenizer)

    # train_loader = DataLoader(
    #     train_dataset, sampler=train_sampler,
    #     batch_size=args.batch_size, num_workers=args.num_workers,
    #     pin_memory=args.pin_mem, drop_last=True, collate_fn=collate,
    #     persistent_workers=True
    # )

    # if val_dataset != None:
    #     val_loader = DataLoader(
    #         val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers,
    #         pin_memory=args.pin_mem, drop_last=False, collate_fn=collate, persistent_workers=True
    #     )
    # else:
    #     val_loader = None
    
    # if test_dataset != None:
    #     test_loader = DataLoader(
    #         test_dataset, sampler=test_sampler, batch_size=args.batch_size,
    #         num_workers=args.num_workers, pin_memory=args.pin_mem,
    #         drop_last=False, collate_fn=collate, persistent_workers=True
    #     )
    # else:
    #     test_loader = None

    model_class = {
        'video': build_model,
        'keypoint': ...
    }.get(args.modal)

    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model}")
    
    if args.modal == 'video':
        model = model_class(args)
    elif args.modal == 'keypoint':
        pass

    patch_size = model.patch_embed.patch_size  # 16
    print("Patch size = %s" % str(patch_size))

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
        # for name in checkpoint_model.keys():
        #     print(f"=======Key: {name}")
        # print(f"_________________Shape of Pos Embed: {checkpoint_model['pos_embed'].shape}")

        # interpolate position embedding
        # if 'pos_embed' in checkpoint_model:
        #     pos_embed_checkpoint = checkpoint_model['pos_embed']
        #     embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
        #     num_patches = model.patch_embed.num_patches  #
        #     num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

        #     # height (== width) for the checkpoint position embedding 
        #     orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
        #                 args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        #     # height (== width) for the new position embedding
        #     new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        #     # class_token and dist_token are kept unchanged
        #     if orig_size != new_size:
        #         print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        #         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        #         # only the position tokens are interpolated
        #         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        #         # B, L, C -> BT, H, W, C -> BT, C, H, W
        #         pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size,
        #                                         orig_size, embedding_size)
        #         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        #         pos_tokens = torch.nn.functional.interpolate(
        #             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        #         # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        #         pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1,
        #                                                             args.num_frames // model.patch_embed.tubelet_size,
        #                                                             new_size, new_size, embedding_size)
        #         pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        #         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        #         checkpoint_model['pos_embed'] = new_pos_embed
        
    utils.load_state_dict(model, checkpoint_model, prefix='')
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        print("Initializing the classification head weights.")
        nn.init.xavier_uniform_(model.head.weight)
        nn.init.constant_(model.head.bias, 0)



    # if args.use_pretrained:
    #     if os.path.exists(args.pretrained_model_checkpoint_path):
    #         model = load_pretrained_weights(model=model, 
    #                                         checkpoint_path=args.pretrained_model_checkpoint_path, 
    #                                         detach_head=True)
    #     else:
    #         raise ValueError(f"Checkpoint not found at {args.pretrained_model_checkpoint_path}")

    model.to(device)
    return model

    # # ModelEMA
    # # Using the weighted moving averages of past parameters fro the current epoch
    # # Smoothening
    # # EMA(t + 1) = EMA(t) * decay + current_value * (1 - decay)
    # model_ema = None

    # if args.model_ema:
    #     model_ema = ModelEma(model,
    #                         decay=args.model_ema_decay,
    #                         device='cpu' if args.model_ema_force_cpu else '',
    #                         resume='')
    #     print(f"Using EMA with decay: {args.model_ema_decay}")

    # model_without_ddp = model
    # n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    # print(f"Model: {model_without_ddp}")
    # print(f"NUmber of params: {n_params}")
    
    # # Effective Batch Size
    # total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    # n_training_steps_per_epoch = len(train_dataset) // total_batch_size
    # # Adjust LR considering batch size and data augmentation during update
    # args.lr = args.lr * total_batch_size / 256
    # args.min_lr = args.min_lr * total_batch_size / 256 
    # args.warmup_lr = args.warmup_lr * total_batch_size / 256
    # print(f"LR: {args.lr:.8f}")
    # print(f"Batch size: {total_batch_size}")
    # print(f"Update frequent: {args.update_freq}")
    # print(f"Number of training examples: {len(train_dataset)}")
    # print(f"Number of training steps per epoch: {n_training_steps_per_epoch}")

    # amp_autocast = contextlib.nullcontext() # Deepspeed automatically handles AMP
    # loss_scaler = 'none' # Handling loss scaling during the underflow and overflow 

    # if args.enable_deepspeed:
    #     loss_scaler = None # deepspeed automatically handles loss scaling
    #     # initialize 
    #     model, optimizer, _, _ = ds_init(
    #         args=args, model=model, model_parameters=model.parameter(),
    #         dist_init_required=not args.distributed
    #     )

    # if args.weight_decay_start is None:
    #     args.weight_decay_start = args.weight_decay
    # if args.weight_decay_end is None:
    #     args.weight_decay_end = args.weight_decay

    # # Weight decay Scheduler
    # wd_schedule_values = utils.wd_scheduler(
    #     args.weight_decay, args.weight_decay_end, args.start_epoch, args.epochs,
    #     args.warmup_epochs, args.weight_decay_start
    # )

    # print(f"Max WD: {max(wd_schedule_values):.6f}, Min WD: {min(wd_schedule_values):.6f}, Warmup lr: {float(args.warmup_lr)}")

    # # --------------- Handle This ------------------------------------
    # criterion = nn.CTCLoss(blank=gloss_tokenizer.silence_id, zero_infinity=True)

    # # Call the latest model
    # utils.auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema)

    # # Evaluation
    # if args.eval:
    #     pred_files = 


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    model = main(opts, ds_init)

    # Creating a dummy batch with varying frame counts
    batch_size = 2
    max_frames = 256  # The maximum number of frames in the batch
    video_shape = (batch_size, 3, max_frames, 224, 224)  # Batch, Channels, Frames, Height, Width
    dummy_input = torch.randn(video_shape).to(opts.device)

    print(f"Dummy input shape: {dummy_input.shape}")

    # Pass the dummy input through the model
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    print(f"Model output: {output[:3]}")

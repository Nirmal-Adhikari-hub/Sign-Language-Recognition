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
from datasets.dataset import build_dataset
from datasets.tokenizer import GlossTokenizer_S2G
from engines.engine import train_one_epoch, validation_one_epoch, final_test
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import collate_fn
from model import build_model
from collections import OrderedDict
import utils
import contextlib
import builtins

# Override print function
original_print = builtins.print

def print(*args, **kwargs):
    if utils.is_main_process():
        original_print(*args, **kwargs)

builtins.print = print

def get_args():
    parser = argparse.ArgumentParser('Visual Backbone Training and Evaluation for Continuous SLR.')
    parser.add_argument('--task', default='s2g', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1,type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model Params
    parser.add_argument('--pretrained', default=True, action='store_true')
    parser.add_argument('--patch_size', default=[16,16], type=int, nargs='+')
    parser.add_argument('--in_chans', default=3, type=int)
    parser.add_argument('--embed_dim', default=384, type=int)
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
    parser.add_argument('--aa', default='rand-m9-mstd0.5') # Auto Augment Policy
    parser.add_argument('--train_interpolation', default='bicubic', type=str)

    # Random Erasing Params
    parser.add_argument('--reprob', default=0., type=float)
    parser.add_argument('--remode', default='pixel', type=str)
    parser.add_argument('--recount', default=1, type=int)

    # Dataset Params
    parser.add_argument('--modal', default='video', type=str)
    parser.add_argument('--video_size', default=[224,224], type=int, nargs='+')
    parser.add_argument('--metadata_path', default='/nas/Dataset/Phoenix/', type=str)
    parser.add_argument('--gloss_to_id_path', default='/nas/Dataset/Phoenix/gloss2ids.pkl', type=str)
    parser.add_argument('--video_path', default='/nas/Dataset/Phoenix/phoenix-2014-videos.zip', type=str)
    parser.add_argument('--keypoint_path', default='', type=str)
    parser.add_argument('--split', default=',', type=str)
    parser.add_argument('--n_classes', default=1235, type=int)
    parser.add_argument('--num_frames', default=256, type=int)
    parser.add_argument('--dataset', default='phoenix-2014', type=str)
    parser.add_argument('--output_dir', default='/nas/Nirmal/workspace/slr_results/videomae', type=str)
    parser.add_argument('--log_dir', default='/nas/Nirmal/workspace/slr_results/videomae', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=str)
    parser.add_argument('--resume', default='/nas/Nirmal/workspace/slr_results/videomae', type=str) # Weight Path during re-learning
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
    parser.add_argument('--zero_stage', default=0, type=int, help='DeepSpeed ZeRO optimization stage (0, 1, 2, 3)')
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--enable_deepspeed', default=True)
    parser.add_argument('--bf16', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true') # distributed training on interactive platform
    parser.add_argument('--dist_url', default='env://')

    # Finetuning
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--use_pretrained', default=True, type=bool)
    parser.add_argument('--pretrained_model_checkpoint_path', default='/nas/Dataset/Phoenix/pretrained/Nirmal/videoMAE_vit-s_16x224x224_checkpoint.pth', type=str)
    parser.add_argument('--use_mean_pooling', action='store_true') # cls_token
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=True)

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

    # Set PYTORCH_CUDA_ALLOC_CONF for memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    utils.init_distributed_mode(args)
    # Deepspeed config.json  generation
    if ds_init != None:
        ds_config = utils.create_ds_config(args)
        print(f"[DEBUG]DeepSpeed Stage: {ds_config['zero_optimization']['stage']}")
        print(f"[DEBUG]bf16 Stage: {ds_config['bf16']}")

    print(args)

    device = torch.device(args.device)

    if utils.is_main_process():
        print(f"Number of GPUS:{utils.get_world_size()}")

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

    # Train dataset
    train_dataset = build_dataset(modal=args.modal, 
                                  gloss_tokenizer=gloss_tokenizer,
                                  is_train=True,
                                  is_test=False,
                                  args=args)
    
    # Using validation during training or not
    if args.disable_eval_during_training:
        val_dataset = None
    else:
        val_dataset = build_dataset(modal=args.modal,
                                    gloss_tokenizer=gloss_tokenizer,
                                    is_train=False,
                                    is_test=False,
                                    args=args)
    test_dataset = build_dataset(modal=args.modal,
                                 gloss_tokenizer=gloss_tokenizer,
                                 is_train=False,
                                 is_test=True,
                                 args=args)
    
    n_tasks = utils.get_world_size() # Total GPU processes
    global_rank = utils.get_rank() # process idx

    # print(f"n_tasks: {n_tasks}, global_rank: {global_rank}")
    train_sampler = DistributedSampler(train_dataset, n_tasks, global_rank, True)

    if args.dist_eval: # Evaluation using DDP
        if len(val_dataset) % n_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by\
                  process number.') if utils.is_main_process() else None
        val_sampler = DistributedSampler(val_dataset,n_tasks, global_rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, n_tasks, global_rank, shuffle=False)
    else: # Sampling sequentially starting from index 0
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)

    # Logging by master process only
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # If the num of aug is greater than 1, use multiple_samples_collate with collate_fn
    # Separate video tensor, label, video_id etc
    '''Always remember that collate does all the organizations for a batch like merging, padding
    making samples of consistent sizes, transformations, preprocessing , etc'''
    collate = partial(collate_fn, gloss_tokenizer=gloss_tokenizer)

    # train_loader = DataLoader(
    #     train_dataset, sampler=train_sampler,
    #     batch_size=args.batch_size, num_workers=args.num_workers,
    #     pin_memory=args.pin_mem, drop_last=True, collate_fn=collate,
    #     persistent_workers=False
    # )

    train_loader = DataLoader(
                train_dataset, 
                sampler=train_sampler,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                pin_memory=args.pin_mem, 
                drop_last=True, 
                collate_fn=collate,
                prefetch_factor=2,  # Optimize data loading
                persistent_workers=False  # Avoid pinned memory issues with gradient checkpointing
                )

    if val_dataset != None:
        # val_loader = DataLoader(
        #     val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers,
        #     pin_memory=args.pin_mem, drop_last=False, collate_fn=collate, persistent_workers=True
        # )

        val_loader = DataLoader(
                val_dataset, 
                sampler=val_sampler, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                pin_memory=args.pin_mem, 
                drop_last=False, 
                collate_fn=collate,
                prefetch_factor=2,  # Similar prefetching strategy for validation
                persistent_workers=False  # Set False for consistency
            )
    else:
        val_loader = None
    
    if test_dataset != None:
        # test_loader = DataLoader(
        #     test_dataset, sampler=test_sampler, batch_size=args.batch_size,
        #     num_workers=args.num_workers, pin_memory=args.pin_mem,
        #     drop_last=False, collate_fn=collate, persistent_workers=True
        # )

        test_loader = DataLoader(
                test_dataset, 
                sampler=test_sampler, 
                batch_size=args.batch_size,
                num_workers=args.num_workers, 
                pin_memory=args.pin_mem,
                drop_last=False, 
                collate_fn=collate,
                prefetch_factor=2,  # Similar prefetching for testing
                persistent_workers=False
            )
    else:
        test_loader = None

    print("Before Model Initialization:")
    print(torch.cuda.memory_summary())


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

    patch_size = model.video_backbone.patch_embed.patch_size  # 16
    # print("Patch size = %s" % str(patch_size))

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
        state_dict = model.video_backbone.state_dict()
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
        
    utils.load_state_dict(model.video_backbone, checkpoint_model, prefix='')
    if hasattr(model.video_backbone, 'head') and isinstance(model.video_backbone.head, nn.Linear):
        print("Initializing the classification head weights.")
        nn.init.xavier_uniform_(model.video_backbone.head.weight)
        nn.init.constant_(model.video_backbone.head.bias, 0)


    model.to(device)

    print("After Model Initialization:")
    print(torch.cuda.memory_summary())


    # ModelEMA
    # Using the weighted moving averages of past parameters fro the current epoch
    # Smoothening
    # EMA(t + 1) = EMA(t) * decay + current_value * (1 - decay)
    model_ema = None

    if args.model_ema:
        model_ema = ModelEma(model,
                            decay=args.model_ema_decay,
                            device='cpu' if args.model_ema_force_cpu else '',
                            resume='')
        print(f"Using EMA with decay: {args.model_ema_decay}")

    model_without_ddp = model
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(f"Model: {model_without_ddp}")
    print(f"NUmber of params: {n_params}")
    
    # Effective Batch Size
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    n_training_steps_per_epoch = len(train_dataset) // total_batch_size
    # Adjust LR considering batch size and data augmentation during update
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256 
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print(f"LR: {args.lr:.8f}")
    print(f"Batch size: {total_batch_size}")
    print(f"Update frequent: {args.update_freq}")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of training steps per epoch: {n_training_steps_per_epoch}")

    # n_layers = model_without_ddp.get_n_layers() # Mamba block의 수

    # Layer decay
    # if args.layer_decay < 1.:
    #     # Head + 가중치 감쇠가 적용되지 않는 layer 고려
    #     assigner = LayerDecayValueAssigner(list(args.layer_decay ** i for i in range(n_layers + 1)))
    # else:
    #     assigner = None

    # if assigner is not None:
    #     print(f"Assinger values: {assigner.values}")

    amp_autocast = contextlib.nullcontext() # Deepspeed automatically handles AMP
    loss_scaler = 'none' # Handling loss scaling during the underflow and overflow 

    if args.enable_deepspeed:
        loss_scaler = None # deepspeed는 자동으로 loss scaling을 처리
        # Layer에 따른 parameter 호출
        # optimizer_params = get_parameter_groups(
        #     model, args.weight_decay, skip_weight_decay_list,
        #     assigner.get_layer_id if assigner != None else None, assigner.get_scale if assigner != None else None
        # )
        # initialize
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=model.parameters(), dist_init_required=not args.distributed
        )

    if args.weight_decay_start is None:
        args.weight_decay_start = args.weight_decay
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    # Weight decay Scheduler
    wd_schedule_values = utils.wd_scheduler(
        args.weight_decay, args.weight_decay_end, args.start_epoch, args.epochs,
        args.warmup_epochs, args.weight_decay_start
    )
    print(f"Max WD: {max(wd_schedule_values):.6f}, Min WD: {min(wd_schedule_values):.6f}, Warmup lr: {float(args.warmup_lr)}")

    # lr scheduler
    lr_schedule_scheduler = utils.CosineAnnealingWarmUpRestarts(
        base_lr=args.warmup_lr, max_lr=args.lr, T_0=args.T_0, T_mul=args.T_mul, T_up=args.warmup_epochs,
        n_iters_per_epoch=n_training_steps_per_epoch, gamma=args.lr_gamma
    )
    lr_schedule_values = lr_schedule_scheduler.get_lrs(args.start_epoch - 1, args.epochs)

    print(f"Max lr: {max(lr_schedule_values):.8f}, Min lr: {min(lr_schedule_values):.8f}, Warmup lr: {float(args.warmup_lr)}")


    # Test one batch through the dataloader and model
    # for i, batch in enumerate(train_loader):
    #     frames, frame_lens, gloss_ids, gloss_len, video_ids = batch
    #     # Move tensors to the device
    #     frames = frames.to(device)
    #     gloss_ids = gloss_ids.to(device)

    #     # Print the shapes of the batch elements
    #     print(f"Batch {i + 1}:")
    #     print(f"  Frames shape: {frames.shape}")
    #     print(f"  Frame lengths: {frame_lens}")
    #     print(f"  Gloss IDs shape: {gloss_ids.shape}")
    #     print(f"  Gloss lengths: {gloss_len}")
    #     print(f"  Video IDs: {video_ids}")

    #     # Perform a forward pass through the model
    #     print("Before Forward Pass:")
    #     # print(torch.cuda.memory_summary())

    #     outputs = model(frames, frame_lens)
    #     print(f"Model Output Shape: {outputs['gloss_logits'].shape, outputs['gloss_probabilities_log'].shape, outputs['valid_len_out']}")

    #     print("After Forward Pass:")
    #     # print(torch.cuda.memory_summary())


    #     # Limit to a few iterations for debugging
    #     if i >= 2:  # Process only 3 batches
    #         break

    # # Verify Validation and Test Datasets
    # if val_loader is not None:
    #     print("\nValidating Validation Set:")
    #     for i, batch in enumerate(val_loader):
    #         frames, frame_lens, gloss_ids, gloss_len, video_ids = batch
    #         # Move tensors to the device
    #         frames = frames.to(device)
    #         gloss_ids = gloss_ids.to(device)

    #         # Print the shapes of the batch elements
    #         print(f"Validation Batch {i + 1}:")
    #         print(f"  Frames shape: {frames.shape}")
    #         print(f"  Frame lengths: {frame_lens}")
    #         print(f"  Gloss IDs shape: {gloss_ids.shape}")
    #         print(f"  Gloss lengths: {gloss_len}")
    #         print(f"  Video IDs: {video_ids}")

    #         # Perform a forward pass through the model
    #         print("Before Validation Forward Pass:")
    #         # print(torch.cuda.memory_summary())

    #         with torch.no_grad():
    #             outputs = model(frames, frame_lens)
    #         print(f"Model Output Shape: {outputs['gloss_logits'].shape, outputs['gloss_probabilities_log'].shape, outputs['valid_len_out']}")

    #         print("After Validation Forward Pass:")
    #         # print(torch.cuda.memory_summary())

    #         # Limit to a few iterations for debugging
    #         if i >= 2:  # Process only 3 batches
    #             break

    # if test_loader is not None:
    #     print("\nValidating Test Set:")
    #     for i, batch in enumerate(test_loader):
    #         frames, frame_lens, gloss_ids, gloss_len, video_ids = batch
    #         # Move tensors to the device
    #         frames = frames.to(device)
    #         gloss_ids = gloss_ids.to(device)

    #         # Print the shapes of the batch elements
    #         print(f"Test Batch {i + 1}:")
    #         print(f"  Frames shape: {frames.shape}")
    #         print(f"  Frame lengths: {frame_lens}")
    #         print(f"  Gloss IDs shape: {gloss_ids.shape}")
    #         print(f"  Gloss lengths: {gloss_len}")
    #         print(f"  Video IDs: {video_ids}")

    #         # Perform a forward pass through the model
    #         print("Before Test Forward Pass:")
    #         # print(torch.cuda.memory_summary())

    #         with torch.no_grad():
    #             outputs = model(frames,frame_lens)
    #         print(f"Model Output Shape: {outputs['gloss_logits'].shape, outputs['gloss_probabilities_log'].shape, outputs['valid_len_out']}")

    #         print("After Test Forward Pass:")
    #         # print(torch.cuda.memory_summary())

    #         # Limit to a few iterations for debugging
    #         if i >= 2:  # Process only 3 batches
    #             break

    

    # # --------------- Handle This ------------------------------------
    criterion = nn.CTCLoss(blank=gloss_tokenizer.silence_id, zero_infinity=True)

    # Call the latest model
    utils.auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema)

    # Evaluation
    if args.eval:
        pred_files = os.path.join(args.output_dir, str(global_rank) + '.txt') # Path to save prediction result for each process
        # Evaluation Criteria
        test_stats = final_test(
            test_loader, model, device, pred_files, amp_autocast, ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
            n_classes=args.n_classes, cls_cnt_list=cls_cnt_list, pos_weights=pos_weights, args=args
        )
        torch.distributed.barrier()

        # Master process
        if global_rank == 0:
            log_stats = {
                **{f'test_{k}': v for k, v in test_stats.items()},
                'n_parameters': n_params
            }

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats) + "\n") # Written in json format with key and value
        
        exit(0) # Evaluation end


    # Training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_wer = 100.
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if log_writer != None:
            log_writer.set_step(epoch * n_training_steps_per_epoch * args.update_freq) # TensorBoardLogger의 step을 현재 epoch의 시작 step으로 설정
        
        # 1 epoch 학습 후, 지표들의 global 평균값을 dictionary 형태로 반환
        
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, loss_scaler, amp_autocast,
            args.clip_grad, model_ema, log_writer, epoch * n_training_steps_per_epoch,
            lr_schedule_values, wd_schedule_values, n_training_steps_per_epoch, args.update_freq,
            no_amp=args.no_amp, bf16=args.bf16, gloss_tokenizer=gloss_tokenizer
        )

        if log_writer is not None:
            log_writer.update(train_avg_loss=train_stats['loss'], head='loss', step=epoch)
            log_writer.update(train_avg_wer=train_stats['wer'], head='perf', step=epoch)

        # latest 이름으로 모델 저장
        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_name='latest', model_ema=model_ema
            )
        
        # Validation 수행 후, 지표들의 global 평균값을 dictionary 형태로 반환
        if val_loader != None:
            test_stats = validation_one_epoch(
                val_loader, model, device, amp_autocast, args.enable_deepspeed,
                args.no_amp, args.bf16, gloss_tokenizer
            )
            timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{timestep}] WER of the network on the {len(val_dataset)} val videos: {test_stats['wer'] * 100:.2f}%")

            if best_wer > test_stats['wer']:
                best_wer = test_stats['wer']

                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_name='best', model_ema=model_ema
                    )
            
            print(f"Best wer: {best_wer * 100:.2f}")

            if log_writer is not None:
                log_writer.update(val_loss=test_stats['loss'], head='loss', step=epoch)
                log_writer.update(val_wer=test_stats['wer'], head='perf', step=epoch)
            
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_params
            }
        else:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'n_parameters': n_params
            }

        if args.output_dir and utils.is_main_process():
            if log_writer != None:
                # log_writer에 기록된 현재까지의 모든 log을 디스크에 저장
                log_writer.flush()
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f: # mode: a -> 파일에 쓰기를 목적으로 열지만, 만약 파일이 있으면 파일의 끝에 추가해 작성
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    pred_files = os.path.join(args.output_dir, str(global_rank) + '.txt')

    if args.test_best:
        print("Auto testing the best model")
        args.eval = True
        # load checkpoint-best.pth
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema
        )
    
    # 각 process별 test 성능 지표 반환
    test_stats = final_test(
        test_loader, model, device, pred_files, amp_autocast, args.enable_deepspeed, args.no_amp, args.bf16, args.n_classes, cls_cnt_list, pos_weights, args
    )
    torch.distributed.barrier()

    # Master process인 경우
    if global_rank == 0:
        log_stats = {
            **{f'test_{k}': v for k, v in test_stats.items()},
            'n_parameters': n_params
        }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + "\n") # key, value 형태의 json 형식으로 작성



if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
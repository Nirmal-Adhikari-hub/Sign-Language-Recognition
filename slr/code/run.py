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
from engines.engine import train_one_epoch, validation_one_epoch, final_test
# from videomamba import videomamba_base
# from mamba import mamba_130m
# from audiomamba import audiomamba_base
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import collate_fn
import utils
import contextlib

def get_args():
    parser = argparse.ArgumentParser('Single modal training and evaluation script for movie genre classification')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model params
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--tubelet_size', default=1, type=int)
    parser.add_argument('--video_size', default=[224, 224], type=int, nargs='+')
    parser.add_argument('--audio_size', default=[32, 18000], type=int, nargs='+')
    parser.add_argument('--fc_drop_rate', default=.1, type=float)
    parser.add_argument('--drop', default=.1, type=float)
    parser.add_argument('--drop_path', default=.1, type=float)
    parser.add_argument('--head_drop', default=.1, type=float)
    parser.add_argument('--disable_eval_during_training', default=False, action='store_true')
    parser.add_argument('--model_ema', default=False, action='store_true')
    parser.add_argument('--model_ema_decay', default=.9999, type=float)
    parser.add_argument('--model_ema_force_cpu', default=False, action='store_true')

    # Optimizer params
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+')
    parser.add_argument('--clip_grad', default=None, type=float) # Max norm
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--weight_decay', default=.05, type=float)
    parser.add_argument('--weight_decay_start', default=None, type=float)
    parser.add_argument('--weight_decay_end', default=None, type=float)
    parser.add_argument('--lr', default=1e-3, type=float) # 최대 학습률
    parser.add_argument('--layer_decay', default=.75, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float) # 최소 학습률
    parser.add_argument('--warmup_lr', default=1e-6, type=float) # warmup 시작 학습률
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--T_0', default=50, type=int)
    parser.add_argument('--T_mul', default=1, type=int)
    parser.add_argument('--lr_gamma', default=0.5, type=float)

    # Augmentation params
    parser.add_argument('--aug_size', default=[224, 224], type=int, nargs='+')
    parser.add_argument('--t_min', default=1., type=float)
    parser.add_argument('--t_max', default=1., type=float)
    parser.add_argument('--color_jitter', default=.4, type=float)
    parser.add_argument('--aa', default='rand-m7-n4-mstd0.5-inc1', type=str) # Auto Augment policy
    parser.add_argument('--train_interpolation', default='bicubic', type=str)

    # Random erasing params
    parser.add_argument('--reprob', default=.25, type=float)
    parser.add_argument('--remode', default='pixel', type=str)
    parser.add_argument('--recount', default=1, type=int)

    # Dataset params
    parser.add_argument('--modal', default='video', type=str)
    parser.add_argument('--metadata_path', default='', type=str)
    parser.add_argument('--gloss_to_id_path', default='', type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--keypoint_path', default='', type=str)
    parser.add_argument('--split', default=',', type=str)
    parser.add_argument('--n_classes', default=1236, type=int)
    parser.add_argument('--patch_size', default=[4, 4], type=int, nargs='+')
    parser.add_argument('--n_frames', default=400, type=int)
    parser.add_argument('--dataset', default='phoenix-2014', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--log_dir', default='', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str) # 재학습 시 weight 경로
    parser.add_argument('--auto_resume', default=False, action='store_true')
    parser.add_argument('--save_ckpt', default=False, action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--test_best', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--dist_eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--no_amp', default=False, type=bool)

    # DDP params
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--enable_deepspeed', default=True)
    parser.add_argument('--bf16', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # Checkpoint
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', default=0, type=int,
                        help='number of layers for using checkpoint')


    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            
            parser = deepspeed.add_config_arguments(parser) # DDP 환경을 초기화하고 deepspeed parser로 wrapping
            ds_init = deepspeed.initialize
        except:
            print("Please install deepspeed")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def main(args, ds_init):
    torch.cuda.empty_cache()
    # args.rank = int(os.environ["RANK"])
    # args.world_size = int(os.environ['WORLD_SIZE'])
    # args.gpu = int(os.environ['LOCAL_RANK'])
    utils.init_distributed_mode(args)

    # deepspeed_config.json 생성
    if ds_init != None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # 재현성을 위해 각 process마다 다른 seed 설정
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed) # For PyTorch
    np.random.seed(seed) # For Numpy

    # 입력 데이터의 크기에 따라 다양한 알고리즘을 벤치마킹하여 최적의 알고리즘을 선택
    # 입력 데이터의 크기가 고정되어 있을 때 효과적
    cudnn.benchmark = True

    # Train dataset
    train_dataset = build_dataset(modal=args.modal, is_train=True, is_test=False, args=args)

    # 학습 시 validation data를 통한 평가 여부
    if args.disable_eval_during_training:
        val_dataset = None
    else:
        val_dataset = build_dataset(modal=args.modal, is_train=False, is_test=False, args=args)
    test_dataset = build_dataset(modal=args.modal, is_train=False, is_test=True, args=args)

    n_tasks = utils.get_world_size() # 총 gpu(process) 수
    global_rank = utils.get_rank() # process idx
    train_sampler = DistributedSampler(train_dataset, n_tasks, global_rank, True)

    if args.dist_eval: # DDP을 사용해 평가할 경우
        if len(val_dataset) % n_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
        val_sampler = DistributedSampler(val_dataset, n_tasks, global_rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, n_tasks, global_rank, shuffle=False)
    else:
        # 0번 index부터 sampling하여 반환
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    
    # Master process인 경우 log 작성
    if global_rank == 0 and args.log_dir is not None: # log_dir도 존재해야 함
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    
    # 증강 횟수가 1보다 크면 collate_fn으로 multiple_samples_collate 사용
    # video tensor, label, video id 등을 분리
    
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem,
        drop_last=True, collate_fn=collate_fn, persistent_workers=True
    )

    if val_dataset != None:
        val_loader = DataLoader(
            val_dataset, sampler=val_sampler,
            batch_size=int(1 * args.batch_size), num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False, collate_fn=collate_fn, persistent_workers=True
        )
    else:
        val_loader = None

    if test_dataset != None:
        test_loader = DataLoader(
            test_dataset, sampler=test_sampler,
            batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False, collate_fn=collate_fn, persistent_workers=True
        )
    else:
        test_loader = None
    
    model_class = {
        'video': videomamba_base,
        'keypoint': ...
    }.get(args.modal)
    
    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model}")
    
    if args.modal == 'video':
        model = model_class(
            pretrained=args.pretrained, width=args.video_size[1], height=args.video_size[0], n_classes=args.n_classes,
            drop_rate=args.drop, fc_drop_rate=args.fc_drop_rate, drop_path_rate=args.drop_path, 
            kernel_size=args.tubelet_size, patch_size=args.patch_size, n_frames=args.n_frames,
            device=args.device, use_checkpoint=args.use_checkpoint, checkpoint_num=args.checkpoint_num
        )
    elif args.modal == 'text':
        model = model_class(
            tokenizer=tokenizer, pretrained=args.pretrained, max_len=args.max_len, n_classes=args.n_classes,
            drop_rate=args.drop, fc_drop_rate=args.fc_drop_rate, drop_path_rate=args.drop_path,
            device=args.device, use_checkpoint=args.use_checkpoint, checkpoint_num=args.checkpoint_num
        )
    elif args.modal == 'audio':
        model = model_class(
            pretrained=args.pretrained, width=args.audio_size[1], height=args.audio_size[0], n_classes=args.n_classes,
            drop_rate=args.drop, fc_drop_rate=args.fc_drop_rate, drop_path_rate=args.drop_path, 
            device=args.device, use_checkpoint=args.use_checkpoint, checkpoint_num=args.checkpoint_num
        )
        
    model.to(device)

    # ModelEMA(Model Exponential Moving Average)
    # 학습 중에 얻은 모델 parameter를 직접 사용하지 않고, 과거의 parameter도 고려해 가중이동평균을 사용
    # 부드럽게 
    # EMA(t + 1) = EMA(t) * decay + current_value * (1 - decay)
    model_ema = None
    
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
        print(f"Using EMA with decay: {args.model_ema_decay}")
        
    model_without_ddp = model
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(f"Model: {model_without_ddp}")
    print(f"# of params: {n_params}")
    skip_weight_decay_list = model.no_weight_decay()
    print(f"Skip weight decay list: {skip_weight_decay_list}")

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size() # Gradient 업데이트 시 배치 크기
    n_training_steps_per_epoch = len(train_dataset) // total_batch_size # epoch당 step = (traing 데이터 수) / (업데이트 시 배치 크기)
    args.lr = args.lr * total_batch_size * args.n_samples / 256 # 업데이트 시 배치 크기와 데이터 증강을 고려한 lr 조정
    args.min_lr = args.min_lr * total_batch_size * args.n_samples / 256 # 업데이트 시 배치 크기와 데이터 증강을 고려한 lr 조정
    args.warmup_lr = args.warmup_lr * total_batch_size * args.n_samples / 256 # 업데이트 시 배치 크기와 데이터 증강을 고려한 lr 조정
    print(f"LR: {args.lr:.8f}")
    print(f"Batch size: {total_batch_size}")
    print(f"Repeated sample: {args.n_samples}")
    print(f"Update frequent: {args.update_freq}")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of training steps per epoch: {n_training_steps_per_epoch}")

    n_layers = model_without_ddp.get_n_layers() # Mamba block의 수

    # Layer decay
    if args.layer_decay < 1.:
        # Head + 가중치 감쇠가 적용되지 않는 layer 고려
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** i for i in range(n_layers + 1)))
    else:
        assigner = None

    if assigner is not None:
        print(f"Assinger values: {assigner.values}")
    
    amp_autocast = contextlib.nullcontext() # deepspeed에서는 AMP를 자동으로 처리
    loss_scaler = 'none' # loss 계산 시에 overflow 또는 underflow 발생 시 처리

    if args.enable_deepspeed:
        loss_scaler = None # deepspeed는 자동으로 loss scaling을 처리
        # Layer에 따른 parameter 호출
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner != None else None, assigner.get_scale if assigner != None else None
        )
        # initialize
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed
        )

    if args.weight_decay_start is None:
        args.weight_decay_start = args.weight_decay
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    
    # Weight decay scheduler
    wd_schedule_values = utils.wd_scheduler(
        args.weight_decay, args.weight_decay_end, args.start_epoch, args.epochs, args.warmup_epochs, args.weight_decay_start
    ) 
    print(f"Max WD: {max(wd_schedule_values):.6f}, Min WD: {min(wd_schedule_values):.6f}, Warmup WD: {args.weight_decay_start}")

    # lr scheduler
    lr_schedule_scheduler = utils.CosineAnnealingWarmUpRestarts(
        base_lr=args.warmup_lr, max_lr=args.lr, T_0=args.T_0, T_mul=args.T_mul, T_up=args.warmup_epochs,
        n_iters_per_epoch=n_training_steps_per_epoch, gamma=args.lr_gamma
    )
    lr_schedule_values = lr_schedule_scheduler.get_lrs(args.start_epoch - 1, args.epochs)

    print(f"Max lr: {max(lr_schedule_values):.8f}, Min lr: {min(lr_schedule_values):.8f}, Warmup lr: {float(args.warmup_lr)}")

    
    # criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip, cls_cnt_list=cls_cnt_list)
    # weights = torch.tensor([sum(cls_cnt_list) / cnt for cnt in cls_cnt_list]) / torch.tensor(sum(cls_cnt_list))
    pos_weights = (torch.tensor(len(train_dataset), device=args.device) - cls_cnt_list) / cls_cnt_list
    print(f"Pos weight: {pos_weights}")
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # latest 모델 호출
    utils.auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema)

    # for name, val in model.named_parameters():
    #     print(f"{name:<20}{val.shape}")
    # Evaluation
    if args.eval:
        pred_files = os.path.join(args.output_dir, str(global_rank) + '.txt') # 각 process별 예측 결과를 저장할 경로
        # 평가 지표
        test_stats = final_test(
            test_loader, model, device, pred_files, amp_autocast, ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
            n_classes=args.n_classes, cls_cnt_list=cls_cnt_list, pos_weights=pos_weights, args=args
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
        
        exit(0) # Evaluation 종료
    
    # Training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_f1_best = 0.
    
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
            no_amp=args.no_amp, bf16=args.bf16, n_classes=args.n_classes, args=args
        )

        if log_writer is not None:
            log_writer.update(train_avg_loss=train_stats['loss'], head='loss', step=epoch)

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
                args.no_amp, args.bf16, args.n_classes, cls_cnt_list, pos_weights, args
            )
            timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{timestep}] F1-micro of the network on the {len(val_dataset)} val videos: {test_stats['f1_micro'] * 100:.2f}%")
            print(f"[{timestep}] F1-macro of the network on the {len(val_dataset)} val videos: {test_stats['f1_macro'] * 100:.2f}%")
            print(f"[{timestep}] F1-weighted of the network on the {len(val_dataset)} val videos: {test_stats['f1_weight'] * 100:.2f}%")
            print(f"[{timestep}] F1-best of the network on the {len(val_dataset)} val videos: {test_stats['f1_best'] * 100:.2f}%")

            if max_f1_best < test_stats['f1_best']:
                max_f1_best = test_stats['f1_best']

                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_name='best', model_ema=model_ema
                    )
            
            print(f"Max f1-score: {max_f1_best * 100:.2f}%")

            if log_writer is not None:
                log_writer.update(val_loss=test_stats['loss'], head='loss', step=epoch)
                log_writer.update(val_acc=test_stats['acc'], head='perf', step=epoch)
                log_writer.update(val_precision=test_stats['precision'], head='perf', step=epoch)
                log_writer.update(val_recall=test_stats['recall'], head='perf', step=epoch)
                log_writer.update(val_f1_micro=test_stats['f1_micro'], head='perf', step=epoch)
                log_writer.update(val_f1_macro=test_stats['f1_macro'], head='perf', step=epoch)
                log_writer.update(val_f1_weight=test_stats['f1_weight'], head='perf', step=epoch)
                log_writer.update(val_f1_best=test_stats['f1_best'], head='perf', step=epoch)
            
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
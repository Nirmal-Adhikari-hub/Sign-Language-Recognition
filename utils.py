import io
import os
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
import torch.utils
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
from torch import inf
import random

from tensorboardX import SummaryWriter

class SmoothedValue(object):
    """
    값을 추적하고, 그 결과를 시간에 따라 부드럽게 보여줌
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size) # window_size 초과 시 가장 처음 index부터 삭제
        self.total = 0.0 # 모든 값의 합
        self.count = 0 # 값의 개수
        self.fmt = fmt # 출력 형식

    def update(self, value, n=1):
        """새로운 값을 n개 추가하고 count, total 업데이트"""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        
        gpu간의 count와 total 값 동기화
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier() # 모든 process에서 count와 total 값을 tensor 형태로 준비할 때 까지 대기
        dist.all_reduce(t) # 각 process의 tensor을 모두 합해 동일한 값으로 준비
        t = t.tolist() 
        self.count = int(t[0]) # float64 -> int
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    # print
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """scalar 지표 업데이트"""
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """지표의 이름으로 dictionary에 접근할 수 있게 하는 method"""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
                
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        iterator 객체를 순회하면서 데이터를 yield하고, print_freq마다 학습 진행 상황을 출력
        학습 과정에서 성능을 추적하고 기록

        header: 로그 메시지 앞에 덧붙일 문자열
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        # 각 시간을 평균과 최댓값으로 관리
        iter_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})') # 반복에 소요되는 시간(window size = 20)
        data_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})') # 데이터 로딩 시간(window size = 20)
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd' # ex) :5d
        # Log msg format
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]', # 반복 횟수
            'eta: {eta}', # eta: 예상 소요 시간
            '{meters}', # 성능 지표
            'time: {time}', # iter_time
            'data: {data}' # data_time
        ]
        if torch.cuda.is_available():
            log_msg.append('Max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg) 
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            data_time.update(time.time() - end) # 데이터를 가져오는데 걸리는 시간
            yield obj # batch
            iter_time.update(time.time() - end) # 한 번 반복에 걸리는 시간
            if i % print_freq == 0 or i == len(iterable) - 1: # 출력 주기 또는 마지막인 경우
                eta_seconds = iter_time.global_avg * (len(iterable) - i) # (한 번 반복하는데 필요한 평균 시간) x (남은 수)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds))) # 초 단위의 시간 간격으로 변환
                # Log msg 출력
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # 소요 시간 정보 출력
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def log_every_joint(self, video_loader, image_loader, print_freq, header=None, image_num_ratio=1.0):
        # prepare random squeue
        total_len = int(len(video_loader) + len(image_loader) * image_num_ratio)
        random_sequence = np.arange(total_len)
        np.random.shuffle(random_sequence)
        loader_list = [iter(video_loader), iter(image_loader)]
        # prepare print template
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})')
        data_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})')
        space_fmt = ':' + str(len(str(total_len))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for i, random_num in enumerate(random_sequence):
            # randomly selct image or video
            if random_num < len(video_loader):
                loader_idx = 0
                use_image = False
                mark = '<<VIDEO BATCH>>\t'
            else:
                loader_idx = 1
                use_image = True
                mark = '<<IMAGE BATCH>>\t'
            data_time.update(time.time() - end)
            yield (next(loader_list[loader_idx]), use_image)
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_len - 1:
                eta_seconds = iter_time.global_avg * (total_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(mark, log_msg.format(
                        i, total_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(mark, log_msg.format(
                        i, total_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / total_len))

# 기록 및 시각화
class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0 # log 단계
        
    def set_step(self, step=None):
        # step이 전달되면 step을 설정
        if step is not None:
            self.step = step
        # 전달되지 않으면 step에 1을 더함
        else:
            self.step += 1
    
    # writer에 scalar 값들 기록
    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    # 현재까지 기록된 모든 log를 디스크에 저장
    def flush(self):
        self.writer.flush()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO() # 메모리에 임시로 공간을 만듬
    torch.save(checkpoint, mem_file) # checkpoint를 mem_file 변수에 저장
    mem_file.seek(0) # 시작 위치로 이동
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, ckpt_path):
    if is_main_process():
        torch.save(obj, ckpt_path)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


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


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def wd_scheduler(base_value, final_value, start_epoch, epochs, warmup_epochs=0, start_warmup_wd=0.):
    """
    base_value: 최댓값
    final_value: 최종값
    """
    warmup_schedule = np.array([])
    if start_warmup_wd != 0:
        print(f"Set warmup epochs:{warmup_epochs}")
    else:
        pass
    if warmup_epochs > 0:
        # warmup의 처음 value에서 base_value까지를 선형으로 설정
        warmup_schedule = np.linspace(start_warmup_wd, base_value, warmup_epochs)

    # warmup을 제외한 본 학습의 epoch 수
    iters = np.arange(epochs - warmup_epochs)

    # 본 학습의 schedule
    # 진폭: 0.5 * (base_value - final_value), 주기: 2 * pi * n_iters
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]
    )

    # 전체 학습의 schedule
    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs # 유효성 확인
    return np.asarray(schedule[start_epoch:epochs + 1])

class CosineAnnealingWarmUpRestarts():
    def __init__(self,base_lr, max_lr, T_0, T_mul, T_up, gamma, n_iters_per_epoch, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        
        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        self.base_lr = base_lr
        self.n_iter_per_epoch = n_iters_per_epoch
        self.last_epoch = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__()
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lr
        elif self.T_cur < self.T_up:
            return (self.max_lr - self.base_lr)*self.T_cur / self.T_up + self.base_lr 
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mul + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mul == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mul - 1) + 1), self.T_mul))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mul ** n - 1) / (self.T_mul - 1)
                    self.T_i = self.T_0 * self.T_mul ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
    
    def get_lrs(self, last_epoch, epochs):
        scheduler = []
        for _ in range(epochs):
            self.step()
            scheduler.append(self.get_lr())
        
        return scheduler[(last_epoch + 1) * self.n_iter_per_epoch:(epochs + 1) * self.n_iter_per_epoch]

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, model_name=None):
    output_dir = Path(args.output_dir)
    if model_name is None:
        model_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % model_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)

        local_save_dir = output_dir
        tag_name = "checkpoint-%s" % model_name
        model.save_checkpoint(save_dir=local_save_dir, tag=tag_name, client_state=client_state)

# 모델 학습을 재개하거나 평가할 때 저장된 checkpoint를 불러옴
def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)

    # deepspeed을 사용할 것이므로 필요하지 않음
    if loss_scaler is not None:
        # torch.amp
        if args.test_best and args.eval:
            args.resume = os.path.join(output_dir, 'checkpoint-best.pth')
        elif os.path.exists(os.path.join(output_dir, 'checkpoint-latest.pth')):
            args.resume = os.path.join(output_dir, 'checkpoint-latest.pth')
        elif os.path.exists(os.path.join(output_dir, 'checkpoint-best.pth')):
            args.resume = os.path.join(output_dir, 'checkpoint-best.pth')
        elif args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        flag = False
        # evaludation을 하는 것이고, 가장 좋은 결괏값을 뽑는 경우
        if args.test_best and args.eval:
            try:
                load_specific_model(model, model_ema, args, output_dir, model_name='best')
                flag = True
            except Exception:
                print('No best model')
        if not flag:
            try:
                load_specific_model(model, model_ema, args, output_dir, model_name='latest')
                flag = True
            except Exception:
                print('No latest model')
        if not flag:
            try:
                load_specific_model(model, model_ema, args, output_dir, model_name='best')
                flag = True
            except Exception:
                print('No best model')
        if not flag: 
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                load_specific_model(model, model_ema, args, output_dir, model_name=latest_ckpt)
            else:
                print('No other models')

# checkpoint 불러오기
def load_specific_model(model, model_ema, args, output_dir, model_name):
    args.resume = os.path.join(output_dir, f'checkpoint-{model_name}')
    print(f"Auto resume the {model_name} checkpoint")
    # DeepSpeedEngine.load_checkpoint -> load_path(load 경로), client_states(state dictionary)
    _, client_states = model.load_checkpoint(args.output_dir, tag=f'checkpoint-{model_name}') # output_dir에서 checkpoint-{model_name} 가져오기
    args.start_epoch = client_states['epoch'] + 1
    if model_ema is not None:
        if args.model_ema:
            _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


# Deepspeed을 사용하여 분산 학습을 설정하기 위한 구성 파일(config.json) 생성
def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json") # 경로 설정
    with open(args.deepspeed_config, mode="w") as writer:
        # AMP를 사용할 것이기 때문에 여기는 무시
        if args.no_amp:
            ds_config = {
                "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
                "train_micro_batch_size_per_gpu": args.batch_size,
                "steps_per_print": 1000,
                "optimizer": {
                    "type": "AdamW",
                    "adam_w_mode": True,
                    "params": {
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "bias_correction": True,
                        "betas": [
                            0.9,
                            0.999
                        ],
                        "eps": 1e-8
                    }
                }
            }
        else:
            # json 파일 작성
            ds_config = {
                "train_batch_size": args.batch_size * args.update_freq * get_world_size(), # optimizer가 업데이트하는 배치 수
                "train_micro_batch_size_per_gpu": args.batch_size, # optimizer가 업데이트하는 gpu당 배치 수
                "steps_per_print": 1000, # 1000 스텝 마다 결과 출력
                "optimizer": {
                    "type": args.opt,
                    "params": {
                        "lr": args.lr, # lr
                        "weight_decay": args.weight_decay, # 가중치 감쇳값
                        "bias_correction": True, # Adam 계열의 optimizer가 초기 학습 단계에서 편향이 발생할 수 있으므로 보정
                        "betas":args.opt_betas,
                        "eps": args.opt_eps # 발산 방지
                    }
                },
                # 추론 시
                # "fp16": {
                #     "enabled": not args.bf16,
                #     "loss_scale": 0, # over, underflow 발생 시 자동으로 scaling 동적 조정
                #     "initial_scale_power": 16, # 동적으로 조정할 때 scaling의 값을 2의 몇 제곱으로 할 것인지
                #     "loss_scale_window": 500, # scaling 값에 대한 업데이트 주기
                #     "hysteresis": 2, # 2번 이상의 underflow가 있을 때 scaling을 사용
                #     "min_loss_scale": 1 # 최소 scaling 값
                # },
                # bf16 사용
                "bf16": {
                    "enabled": args.bf16
                },
                "zero_optimization": {
                    "stage": 0,
                    # "allgather_partitions": True,
                    # "overlap_comm": True,
                    # "reduce_scatter": True,
                    # "contiguous_gradients": True,
                    # "allgather_bucket_size": 5e8,
                    # "reduce_bucket_size": 5e8,
                    # 'round_robin_gradients': True,
                    # "offload_optimizer": {
                    #     "device": "cpu",
                    #     "pin_memory": True,
                    #     "fast_init": True,
                    # },
                    # "offload_param": {
                    #     "device": "cpu",
                    #     "pin_memory": True
                    # },
                    # "stage3_max_live_parameters" : 6e7,
                    # "stage3_max_reuse_distance" : 6e7,
                    # "stage3_prefetch_bucket_size" : 6e7,
                    # "stage3_param_persistence_threshold" : 6e7,
                    # "sub_group_size": 1e9,
                    # "elastic_checkpoint": True,
                }
            }
        # 들여쓰기를 2칸으로 하여 dictionary를 json으로 바꾸고 작성
        writer.write(json.dumps(ds_config, indent=2))


def collate_fn(batch, gloss_tokenizer):
    """
    Collate function to handle variable-sized frames and gloss_ids.
    Args:
        batch: List of tuples where each tuple contains (frames, gloss_ids, idx).
               frames: Tensor of shape (C, T, H, W)
               gloss_ids: Tensor of shape (N,)
               idx: Index of the sample
    Returns:
        frames: Padded tensor of shape (B, C, T_max, H, W)
        gloss_ids: Padded tensor of shape (B, N_max)
        idx: List of indices
    """
    # Extract components
    frames, origin_ts, gloss_ids, origin_gloss_lens, idx = zip(*batch)
    
    # Determine maximum T (time) and N (gloss_ids length) in the batch
    max_t = max(f.shape[1] for f in frames)  # T dimension of frames
    max_n = max(g.shape[0] for g in gloss_ids)  # Length of gloss_ids

    # Padding frames
    padded_frames = []
    for f in frames:
        c, t, h, w = f.shape
        pad_t = max_t - t
        pad_frame = torch.cat([f, f[:, -1:, :, :].repeat(1, pad_t, 1, 1)], dim=1)  # Repeat last frame
        padded_frames.append(pad_frame)
    padded_frames = torch.stack(padded_frames)  # Shape: (B, C, T_max, H, W)
    # print(f"Padded frames shape: {padded_frames.shape}")
    """
    torch.Size([2, 3, 98, 224, 224])
    """

    # Padding gloss_ids
    padded_gloss_ids = []
    for g in gloss_ids:
        pad_n = max_n - g.shape[0]
        pad_gloss = torch.cat([g, torch.full((pad_n,), gloss_tokenizer.pad_id, dtype=g.dtype)], dim=0)  # Pad with silence_id
        padded_gloss_ids.append(pad_gloss)
    padded_gloss_ids = torch.stack(padded_gloss_ids)  # Shape: (B, N_max)
    # print(f"Padded gloss ids: {padded_gloss_ids}")
    """
    tensor([
        [  6,  16,  60, 417,   5,   1,   1,   1],
        [ 47,   8,  24, 174,  72, 144,  11,  42]]
    )
    """
    # print(f"Origin ts: {origin_ts}")
    """
    tensor([66, 98])
    """
    return padded_frames, torch.tensor(origin_ts), padded_gloss_ids, torch.tensor(origin_gloss_lens), list(idx)

# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import os
import zipfile


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return '.zip@' in img_or_path


class ZipReader(object):
    """A class to read zipped files"""
    zip_bank = dict() # A dictionary to cache opened zip files, with the zip file path as key and `ZipFile` objects as values.

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        """
        Retrieves a `ZipFile` object for the given zip file path. If the zip_file is already cached in the `zip_bank`,
        it returns the cached object. Otherwise, it opens that zip file, caches it, and returns it.
        """
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        """
        Splits a `zip-style path` (a combined zip file path and internal file path  separated by `@`) into
        the zip file path and the relative internal file path.

        Returns:
        tuple: A tuple containing:
            - `zip_path` (str): The zip file path, e.g., "zipfile.zip".
            - `folder_path` (str): The relative path inside the zip file, e.g., "inside/folder/file.txt".
        """
        pos_at = path.index('@')
        assert pos_at != -1, "character '@' is not found from the given path '%s'" % path

        zip_path = path[0: pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        """
        Lists all the folders directly inside a given folder within the zip file (as zip file name structure contains the directory info).
        """
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_folder_name in zfile.namelist():
            file_folder_name = str.strip(file_folder_name, '/')
            if file_folder_name.startswith(folder_path) and \
               len(os.path.splitext(file_folder_name)[-1]) == 0 and \
               file_folder_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_folder_name)
                else:
                    folder_list.append(file_folder_name[len(folder_path)+1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=None):
        """
        List all the files in a given folder within the zip file, optionally foltering by file extensions.
        """
        if extension is None:
            extension = ['.*']
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_folder_name in zfile.namelist():
            file_folder_name = str.strip(file_folder_name, '/')
            if file_folder_name.startswith(folder_path) and \
                    str.lower(os.path.splitext(file_folder_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_folder_name)
                else:
                    file_lists.append(file_folder_name[len(folder_path)+1:])

        return file_lists

    @staticmethod
    def read(path):
        """
        Reads the binary content of the file from within the zip file.
        """
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        # print(f"HEREEEEEEEEEEEEEEEEEEEEEEEE {zip_path, path_img}")
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data
import os
import time
import numpy as np
import pandas as pd
import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from timm.utils import ModelEma
import utils
from functools import partial
from torchmetrics.classification import BinaryAccuracy, MultilabelF1Score, MultilabelPrecision, MultilabelRecall

# def train_class_batch(model, frames, scripts, audios, targets, criterions: dict[str, nn.Module]):
#     comb_outputs, video_outputs, text_outputs, audio_outputs = model(frames, scripts, audios)
#     outputs = {'combined': comb_outputs, 'video': video_outputs, 'text': text_outputs, 'audio': audio_outputs}
    
#     loss_dict = {}
#     for modal, criterion in criterions.items():
#         loss_dict[modal] = criterion(outputs[modal], targets)
#         # if modal == 'combined':
#         #     loss_dict['combined'] = criterion(comb_outputs, targets)
#         # elif modal == 'video':
#         #     loss_dict['video'] = criterion(video_outputs, targets)
#         # elif modal == 'text':
#         #     loss_dict['text'] = criterion(text_outputs, targets)
    
#     return loss_dict, outputs

def train_class_batch(model, data, targets, criterion):
    outputs = model(data)
    loss = criterion(outputs, targets)
    
    return loss, outputs

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    try:
        return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
    except Exception:
        return 0
        
def optimize_f1(preds, labels, n_classes: None, args=None):
    if type(preds) != torch.Tensor:
        preds = torch.tensor(preds)
    if type(labels) != torch.Tensor:
        labels = torch.tensor(labels)
        
    thresholds = torch.arange(0.3, 0.8, 0.05)
    f1_res = []

    for th in thresholds:
        f1 = MultilabelF1Score(num_labels=n_classes, average='micro', threshold=th.item()).to(torch.device(args.device))
        f1_res.append(f1(preds, labels))
    
    best_f1_idx = torch.argmax(torch.tensor(f1_res)).item()

    print(f'Best Threshold: {thresholds[best_f1_idx]}')
    print(f'Best F1-score: {f1_res[best_f1_idx]}')
    
    return f1_res[best_f1_idx].item()
    
def train_one_epoch(
    model: torch.nn.Module, criterion: nn.Module,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
    model_ema: Optional[ModelEma] = None, log_writer=None,
    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
    n_training_steps_per_epoch=None, update_freq=None, no_amp=False, bf16=False, n_classes=None, args=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    acc = BinaryAccuracy().to(device)
    precision = MultilabelPrecision(num_labels=n_classes, average='micro').to(device)
    recall = MultilabelRecall(num_labels=n_classes, average='micro').to(device)
    f1_micro = MultilabelF1Score(num_labels=n_classes, average='micro').to(device)
    f1_macro = MultilabelF1Score(num_labels=n_classes, average='macro').to(device)
    f1_weight = MultilabelF1Score(num_labels=n_classes, average='weighted').to(device)

    if loss_scaler is None:  # For deepspeed
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # Update lr & wd
    if (lr_schedule_values is not None) or (wd_schedule_values is not None):
        for param_group in optimizer.param_groups:
            if lr_schedule_values is not None:
                if "lr_scale" in param_group:
                    param_group['lr'] = lr_schedule_values[epoch] * param_group['lr_scale']
                else:
                    param_group['lr'] = lr_schedule_values[epoch]
            if (wd_schedule_values is not None) and (param_group['weight_decay'] > 0):
                param_group['weight_decay'] = wd_schedule_values[epoch]

    for data_iter_step, (data, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        threshold = data_iter_step / update_freq
        if threshold > n_training_steps_per_epoch:
            continue

        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if loss_scaler is None: # For DeepSpeed
        #     if not no_amp:
        #         frames = frames.bfloat16() if bf16 else frames
        #     loss, outputs = train_class_batch(model, frames, targets, criterion)
        # else:
        #     with amp_autocast:
        #         loss, outputs = train_class_batch(model, frames, targets, criterion)

        data = data.bfloat16() if bf16 else data
        # loss_dict, total_outputs = train_class_batch(model, frames, scripts, audios, targets, criterions)
        loss, outputs = train_class_batch(model, data, targets, criterion)
        
        # loss = args.loss_comb * loss_dict['combined'] + args.loss_video * loss_dict['video'] + args.loss_text *  loss_dict['text'] + args.loss_audio * loss_dict['audio']
        
        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss) # loss을 loss_list에 복사
        loss_list = torch.tensor(loss_list).clone().detach()

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()

        if loss_list_isnan or loss_list_isinf:
            print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # if loss_scaler is None: # For deepspeed
        #     loss /= update_freq
        #     model.backward(loss)
        #     model.step()

        #     if (data_iter_step + 1) % update_freq == 0:
        #         # model.zero_grad()
        #         # Deepspeed will call step() & model.zero_grad() automatic
                        
        #         if model_ema is not None:
        #             model_ema.update(model)
        #     grad_norm = None
        #     loss_scale_value = get_loss_scale_for_deepspeed(model) # optimizer.cur_scale
        # else:
        #     if loss_scaler != 'none':
        #         # this attribute is added by timm on one optimizer (adahessian)
        #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        #         loss /= update_freq
        #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
        #                                 parameters=model.parameters(), create_graph=is_second_order,
        #                                 update_grad=(data_iter_step + 1) % update_freq == 0)
        #         if (data_iter_step + 1) % update_freq == 0:
        #             optimizer.zero_grad()
        #             if model_ema is not None:
        #                 model_ema.update(model)
        #         loss_scale_value = loss_scaler.state_dict()["scale"]
        #     else:
        #         loss /= update_freq
        #         loss.backward()
        #         if (data_iter_step + 1) % update_freq == 0:
        #             if max_norm is not None:
        #                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             if model_ema is not None:
        #                 model_ema.update(model)
        #         loss_scale_value = 0

        loss /= update_freq
        model.backward(loss)
        model.step()

        if (data_iter_step + 1) % update_freq == 0:
            # model.zero_grad()
            # Deepspeed will call step() & model.zero_grad() automatic
                    
            if model_ema is not None:
                model_ema.update(model)

        grad_norm = None
        loss_scale_value = get_loss_scale_for_deepspeed(model) # optimizer.cur_scale

        # if (data_iter_step) % update_freq == 0:
        #     for out, target in zip(torch.sigmoid(outputs).detach().tolist(), targets.detach().tolist()):
        #         out_fmt = " ".join([f"{round(v, 3):<5}" for v in out])
        #         tar_fmt = " ".join([f"{int(v):<5}" for v in target])
        #         res_fmt = " ".join([f"{('O' if round(o) ==  t else 'X'):<5}" for o, t in zip(out, target)])

        #         print("-" * 155)
        #         print(f"Predicted:\t{out_fmt}")
        #         print(f"Ground truth:\t{tar_fmt}")
        #         print(f"Result:\t\t{res_fmt}")
        #         print("-" * 155)

        torch.cuda.synchronize()
    
        # Output: logits -> update method 호출 시 자동으로 sigmoid 적용
        acc.update(outputs, targets)
        precision.update(outputs, targets)
        recall.update(outputs, targets)
        f1_micro.update(outputs, targets)
        f1_macro.update(outputs, targets)
        f1_weight.update(outputs, targets)

        acc_val = acc.compute().item()
        precision_val = precision.compute().item()
        recall_val = recall.compute().item()
        f1_micro_val = f1_micro.compute().item()
        f1_macro_val = f1_macro.compute().item()
        f1_weight_val = f1_weight.compute().item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(acc=acc_val)
        metric_logger.update(precision=precision_val)
        metric_logger.update(recall=recall_val)
        metric_logger.update(f1_micro=f1_micro_val)
        metric_logger.update(f1_macro=f1_macro_val)
        metric_logger.update(f1_weight=f1_weight_val)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"]) # Layer decay 값이 적용된 lr들 중 최솟값과 현재 lr의 곱
            max_lr = max(max_lr, group["lr"]) # Layer decay 값이 적용된 lr들 중 최댓값(1)과 현재 lr의 곱
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(acc=acc_val, head="perf")
            log_writer.update(precision=precision_val, head="perf")
            log_writer.update(recall=recall_val, head="perf")
            log_writer.update(f1_micro=f1_micro_val, head="perf")
            log_writer.update(f1_macro=f1_macro_val, head="perf")
            log_writer.update(f1_weight=f1_weight_val, head="perf")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")


            # step + 1
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    #     metric_logger.update(loss=loss_value)

    #     # Output: logits -> update method 호출 시 자동으로 sigmoid 적용
    #     min_lr = 10.
    #     max_lr = 0.
    #     for group in optimizer.param_groups:
    #         min_lr = min(min_lr, group["lr"]) # Layer decay 값이 적용된 lr들 중 최솟값과 현재 lr의 곱
    #         max_lr = max(max_lr, group["lr"]) # Layer decay 값이 적용된 lr들 중 최댓값(1)과 현재 lr의 곱
    #     metric_logger.update(lr=max_lr)
    #     metric_logger.update(min_lr=min_lr)

    #     weight_decay_value = None
    #     for group in optimizer.param_groups:
    #         if group["weight_decay"] > 0:
    #             weight_decay_value = group["weight_decay"]
    #     metric_logger.update(weight_decay=weight_decay_value)
    #     metric_logger.update(grad_norm=grad_norm)

    #     if log_writer is not None:
    #         log_writer.update(loss=loss_value, head="loss")
    #         log_writer.update(loss_scale=loss_scale_value, head="opt")
    #         log_writer.update(lr=max_lr, head="opt")
    #         log_writer.update(min_lr=min_lr, head="opt")
    #         log_writer.update(weight_decay=weight_decay_value, head="opt")
    #         log_writer.update(grad_norm=grad_norm, head="opt")

    #     for modal, outputs in total_outputs.items():
    #         loss_val = loss_dict[modal]
    #         acc = accs[modal]
    #         precision = precisions[modal]
    #         recall = recalls[modal]
    #         f1_micro = f1_micros[modal]
    #         f1_macro = f1_macros[modal]
    #         f1_weight = f1_weights[modal]

    #         acc.update(outputs, targets)
    #         precision.update(outputs, targets)
    #         recall.update(outputs, targets)
    #         f1_micro.update(outputs, targets)
    #         f1_macro.update(outputs, targets)
    #         f1_weight.update(outputs, targets)

    #         acc_val = acc.compute().item()
    #         precision_val = precision.compute().item()
    #         recall_val = recall.compute().item()
    #         f1_micro_val = f1_micro.compute().item()
    #         f1_macro_val = f1_macro.compute().item()
    #         f1_weight_val = f1_weight.compute().item()

    #         metric_logger.meters[f"{modal}_loss"].update(loss_val)
    #         metric_logger.meters[f"{modal}_acc"].update(acc_val)
    #         metric_logger.meters[f"{modal}_precision"].update(precision_val)
    #         metric_logger.meters[f"{modal}_recall"].update(recall_val)
    #         metric_logger.meters[f"{modal}_f1_micro"].update(f1_micro_val)
    #         metric_logger.meters[f"{modal}_f1_macro"].update(f1_macro_val)
    #         metric_logger.meters[f"{modal}_f1_weight"].update(f1_weight_val)


    #         if log_writer is not None:
    #             log_writer.update(loss=loss_val, head=modal)
    #             log_writer.update(acc=acc_val, head=modal)
    #             log_writer.update(precision=precision_val, head=modal)
    #             log_writer.update(recall=recall_val, head=modal)
    #             log_writer.update(f1_micro=f1_micro_val, head=modal)
    #             log_writer.update(f1_macro=f1_macro_val, head=modal)
    #             log_writer.update(f1_weight=f1_weight_val, head=modal)

    #     if log_writer is not None:
    #         log_writer.set_step()

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



    
@torch.no_grad()
def validation_one_epoch(
    data_loader, model, device, amp_autocast, ds=True, no_amp=False, 
    bf16=False, n_classes=None, cls_cnt_list=None, pos_weights=None, args=None
):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # criterions = {
    #     'combined': nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    #     'video': nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    #     'text': nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    #     'audio': nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # }
    # criterion = AsymmetricLoss(
    #     gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip,
    #     cls_cnt_list=cls_cnt_list, pos_weights=pos_weights, neg_weights=neg_weights
    # )
    # criterion = MultiLabelLoss(pos_weights, neg_weights)
    # criterion = FocalLoss(gamma=args.gamma, cls_cnt_list=cls_cnt_list)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('combined_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('video_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('text_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('audio_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))

    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    # accs = {'combined': BinaryAccuracy().to(device), 'video': BinaryAccuracy().to(device), 'text': BinaryAccuracy().to(device), 'audio': BinaryAccuracy().to(device)}
    # precisions = {'combined': MultilabelPrecision(num_labels=n_classes, average='micro').to(device), 'video': MultilabelPrecision(num_labels=n_classes, average='micro').to(device), 'text': MultilabelPrecision(num_labels=n_classes, average='micro').to(device), 'audio': MultilabelPrecision(num_labels=n_classes, average='micro').to(device)}
    # recalls = {'combined': MultilabelRecall(num_labels=n_classes, average='micro').to(device), 'video': MultilabelRecall(num_labels=n_classes, average='micro').to(device), 'text': MultilabelRecall(num_labels=n_classes, average='micro').to(device), 'audio': MultilabelRecall(num_labels=n_classes, average='micro').to(device)}
    # f1_micros = {'combined': MultilabelF1Score(num_labels=n_classes, average='micro').to(device), 'video': MultilabelF1Score(num_labels=n_classes, average='micro').to(device), 'text': MultilabelF1Score(num_labels=n_classes, average='micro').to(device), 'audio': MultilabelF1Score(num_labels=n_classes, average='micro').to(device)}
    # f1_macros = {'combined': MultilabelF1Score(num_labels=n_classes, average='macro').to(device), 'video': MultilabelF1Score(num_labels=n_classes, average='macro').to(device), 'text': MultilabelF1Score(num_labels=n_classes, average='macro').to(device), 'audio': MultilabelF1Score(num_labels=n_classes, average='macro').to(device)}
    # f1_weights = {'combined': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device), 'video': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device), 'text': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device), 'audio': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device)}
    acc = BinaryAccuracy().to(device)
    precision = MultilabelPrecision(num_labels=n_classes, average='micro').to(device)
    recall = MultilabelRecall(num_labels=n_classes, average='micro').to(device)
    f1_micro = MultilabelF1Score(num_labels=n_classes, average='micro').to(device)
    f1_macro = MultilabelF1Score(num_labels=n_classes, average='macro').to(device)
    f1_weight = MultilabelF1Score(num_labels=n_classes, average='weighted').to(device)

    f1_ths = {}
    interval = torch.arange(.3, .75, .05)
    for th, key in zip(interval, range(len(interval))):
        f1_ths[key] = MultilabelF1Score(num_labels=n_classes, threshold=th.item(), average='micro').to(device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        data = batch[0]
        targets = batch[1]

        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute outputs
        if ds:
            if not no_amp:
                data = data.bfloat16() if bf16 else data
            outputs = model(data)
            # loss_dict = {}

            # for output, (modal, criterion) in zip(outputs, criterions.items()):
            #     loss_dict[modal] = criterion(output, targets)
            loss = criterion(outputs, targets)
        else:
            with amp_autocast:
                outputs = model(data)
                loss = criterion(outputs, targets)
                # loss_dict = {}

                # for output, (modal, criterion) in zip(outputs, criterions.items()):
                #     loss_dict[modal] = criterion(output, targets)

        # batch_size = data.shape[0]
        # # loss_value = args.loss_comb * loss_dict['combined'].item() + args.loss_video * loss_dict['video'].item() + args.loss_text * loss_dict['text'].item() + args.loss_audio * loss_dict['audio'].item()
        # loss_value = loss.item()
        # metric_logger.meters['loss'].update(loss_value, n=batch_size)
        # metric_logger.meters['acc'].update(acc.compute().item(), n=batch_size)
        # metric_logger.meters['precision'].update(precision.compute().item(), n=batch_size)
        # metric_logger.meters['recall'].update(recall.compute().item(), n=batch_size)
        # metric_logger.meters['f1_micro'].update(f1_micro.compute().item(), n=batch_size)
        # metric_logger.meters['f1_macro'].update(f1_macro.compute().item(), n=batch_size)
        # metric_logger.meters['f1_weight'].update(f1_weight.compute().item(), n=batch_size)

        acc.update(outputs, targets)
        precision.update(outputs, targets)
        recall.update(outputs, targets)
        f1_micro.update(outputs, targets)
        f1_macro.update(outputs, targets)
        f1_weight.update(outputs, targets)

        batch_size = data.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.compute().item(), n=batch_size)
        metric_logger.meters['precision'].update(precision.compute().item(), n=batch_size)
        metric_logger.meters['recall'].update(recall.compute().item(), n=batch_size)
        metric_logger.meters['f1_micro'].update(f1_micro.compute().item(), n=batch_size)
        metric_logger.meters['f1_macro'].update(f1_macro.compute().item(), n=batch_size)
        metric_logger.meters['f1_weight'].update(f1_weight.compute().item(), n=batch_size)

        # For the best f1-score
        max_th, f1_best = 0., 0.
        for n, f1_th in f1_ths.items():
            th = round(n * 0.05 + 0.3, 2)
            f1_th.update(outputs, targets)
            metric_logger.meters[f'f1_{th}'].update(f1_th.compute().item(), n=batch_size)
            if f1_best < metric_logger.meters[f'f1_{th}'].global_avg:
                max_th = th
                f1_best = metric_logger.meters[f'f1_{th}'].global_avg
    
    print(f"Best threshold: {max_th}")
    print(f"Best f1-score: {f1_best}")
    metric_logger.meters['f1_best'].update(f1_best)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, amp_autocast, ds=True, no_amp=False, bf16=False, n_classes=None, cls_cnt_list=None, pos_weights=None, args=None):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # criterions = {
    #     'combined': nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    #     'video': nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    #     'text': nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    #     'audio': nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # }
    # criterion = AsymmetricLoss(
    #     gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip,
    #     cls_cnt_list=cls_cnt_list, pos_weights=pos_weights, neg_weights=neg_weights
    # )
    # criterion = MultiLabelLoss(pos_weights, neg_weights)
    # criterion = FocalLoss(gamma=args.gamma, cls_cnt_list=cls_cnt_list)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('combined_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('video_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('text_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('audio_loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))

    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # accs = {'combined': BinaryAccuracy().to(device), 'video': BinaryAccuracy().to(device), 'text': BinaryAccuracy().to(device), 'audio': BinaryAccuracy().to(device)}
    # precisions = {'combined': MultilabelPrecision(num_labels=n_classes, average='micro').to(device), 'video': MultilabelPrecision(num_labels=n_classes, average='micro').to(device), 'text': MultilabelPrecision(num_labels=n_classes, average='micro').to(device), 'audio': MultilabelPrecision(num_labels=n_classes, average='micro').to(device)}
    # recalls = {'combined': MultilabelRecall(num_labels=n_classes, average='micro').to(device), 'video': MultilabelRecall(num_labels=n_classes, average='micro').to(device), 'text': MultilabelRecall(num_labels=n_classes, average='micro').to(device), 'audio': MultilabelRecall(num_labels=n_classes, average='micro').to(device)}
    # f1_micros = {'combined': MultilabelF1Score(num_labels=n_classes, average='micro').to(device), 'video': MultilabelF1Score(num_labels=n_classes, average='micro').to(device), 'text': MultilabelF1Score(num_labels=n_classes, average='micro').to(device), 'audio': MultilabelF1Score(num_labels=n_classes, average='micro').to(device)}
    # f1_macros = {'combined': MultilabelF1Score(num_labels=n_classes, average='macro').to(device), 'video': MultilabelF1Score(num_labels=n_classes, average='macro').to(device), 'text': MultilabelF1Score(num_labels=n_classes, average='macro').to(device), 'audio': MultilabelF1Score(num_labels=n_classes, average='macro').to(device)}
    # f1_weights = {'combined': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device), 'video': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device), 'text': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device), 'audio': MultilabelF1Score(num_labels=n_classes, average='weighted').to(device)}
    acc = BinaryAccuracy().to(device)
    precision = MultilabelPrecision(num_labels=n_classes, average='micro').to(device)
    recall = MultilabelRecall(num_labels=n_classes, average='micro').to(device)
    f1_micro = MultilabelF1Score(num_labels=n_classes, average='micro').to(device)
    f1_macro = MultilabelF1Score(num_labels=n_classes, average='macro').to(device)
    f1_weight = MultilabelF1Score(num_labels=n_classes, average='weighted').to(device)

    f1_ths = {}
    interval = torch.arange(.3, .75, .05)
    for th, key in zip(interval, range(len(interval))):
        f1_ths[key] = MultilabelF1Score(num_labels=n_classes, threshold=th.item(), average='micro').to(device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        data = batch[0]
        targets = batch[1]

        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute outputs
        if ds:
            if not no_amp:
                data = data.bfloat16() if bf16 else data
            outputs = model(data)
            # loss_dict = {}

            # for output, (modal, criterion) in zip(outputs, criterions.items()):
            #     loss_dict[modal] = criterion(output, targets)
            loss = criterion(outputs, targets)
        else:
            with amp_autocast:
                outputs = model(data)
                loss = criterion(outputs, targets)
                # loss_dict = {}

                # for output, (modal, criterion) in zip(outputs, criterions.items()):
                #     loss_dict[modal] = criterion(output, targets)

        batch_size = data.shape[0]
        # loss_value = args.loss_comb * loss_dict['combined'].item() + args.loss_video * loss_dict['video'].item() + args.loss_text * loss_dict['text'].item() + args.loss_audio * loss_dict['audio'].item()
        loss_value = loss.item()
        metric_logger.meters['loss'].update(loss_value, n=batch_size)
        metric_logger.meters['acc'].update(acc.compute().item(), n=batch_size)
        metric_logger.meters['precision'].update(precision.compute().item(), n=batch_size)
        metric_logger.meters['recall'].update(recall.compute().item(), n=batch_size)
        metric_logger.meters['f1_micro'].update(f1_micro.compute().item(), n=batch_size)
        metric_logger.meters['f1_macro'].update(f1_macro.compute().item(), n=batch_size)
        metric_logger.meters['f1_weight'].update(f1_weight.compute().item(), n=batch_size)

        acc.update(outputs, targets)
        precision.update(outputs, targets)
        recall.update(outputs, targets)
        f1_micro.update(outputs, targets)
        f1_macro.update(outputs, targets)
        f1_weight.update(outputs, targets)

        batch_size = data.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.compute().item(), n=batch_size)
        metric_logger.meters['precision'].update(precision.compute().item(), n=batch_size)
        metric_logger.meters['recall'].update(recall.compute().item(), n=batch_size)
        metric_logger.meters['f1_micro'].update(f1_micro.compute().item(), n=batch_size)
        metric_logger.meters['f1_macro'].update(f1_macro.compute().item(), n=batch_size)
        metric_logger.meters['f1_weight'].update(f1_weight.compute().item(), n=batch_size)

        # For the best f1-score
        max_th, f1_best = 0., 0.
        for n, f1_th in f1_ths.items():
            th = round(n * 0.05 + 0.3, 2)
            f1_th.update(outputs, targets)
            metric_logger.meters[f'f1_{th}'].update(f1_th.compute().item(), n=batch_size)
            if f1_best < metric_logger.meters[f'f1_{th}'].global_avg:
                max_th = th
                f1_best = metric_logger.meters[f'f1_{th}'].global_avg
    
    print(f"Best threshold: {max_th}")
    print(f"Best f1-score: {f1_best}")
    metric_logger.meters['f1_best'].update(f1_best)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

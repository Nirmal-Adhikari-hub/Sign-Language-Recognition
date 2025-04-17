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
from itertools import chain
from torchmetrics.text import WordErrorRate
from datasets.phoenix_cleanup import clean_phoenix_2014

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

def train_class_batch(model, data, inp_lens, targets, tgt_lens, criterion):
    # print("[DEBUG]: Before forward pass:")
    # print(torch.cuda.memory_summary())
    outputs = model(data, inp_lens)
    # CTC Loss 계산 직전 디버깅 코드
    # print("outputs['gloss_logits'] shape:", outputs['gloss_logits'].shape) 
    # print("outputs['gloss_probabilities_log'] shape:", outputs['gloss_probabilities_log'].shape) 
    # print("outputs['valid_len_out'] shape:", outputs['valid_len_out']) 
    # print("target_lengths:", tgt_lens)
    # print("targets sample:", targets)

    loss = criterion(
        log_probs=outputs['gloss_probabilities_log'].permute(1, 0, 2), targets=targets,
        input_lengths=outputs['valid_len_out'], target_lengths=tgt_lens
    )
    # print("[DEBUG]: Before after loss calculations:")
    # print(torch.cuda.memory_summary())
    
    return loss, outputs

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    try:
        return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
    except Exception:
        return 0
        
def train_one_epoch(
    model: torch.nn.Module, criterion: nn.Module,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
    model_ema: Optional[ModelEma] = None, log_writer=None,
    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
    n_training_steps_per_epoch=None, update_freq=None, no_amp=False, bf16=False, gloss_tokenizer=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    wer = WordErrorRate(sync_on_compute=False)

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

    
    for data_iter_step, (data, input_lengths, targets, target_lengths, name) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        threshold = data_iter_step / update_freq
        if threshold > n_training_steps_per_epoch:
            continue

        data = data.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        # import os
        # import matplotlib.pyplot as plt

        # def save_batch_frames(data, batch_idx, name, output_dir="visualizations"):
        #     """
        #     한 배치의 모든 프레임을 저장하는 함수.
        #     :param data: Tensor (B, C, T, H, W)
        #     :param batch_idx: 현재 배치 번호
        #     :param output_dir: 저장 디렉토리
        #     """
        #     os.makedirs(output_dir, exist_ok=True)
        #     batch_size, _, num_frames, _, _ = data.shape

        #     for i in range(batch_size):  # 배치 내 모든 샘플
        #         if i == 0:
        #             print(name)
        #         sample_dir = os.path.join(output_dir, f"batch_{batch_idx}_sample_{name[i]}")
        #         os.makedirs(sample_dir, exist_ok=True)
                
        #         for t in range(num_frames):  # 샘플 내 모든 프레임
        #             frame = data[i, :, t, :, :].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
                    
        #             # 저장 경로 설정
        #             save_path = os.path.join(sample_dir, f"frame_{t}.png")
                    
        #             # 이미지 저장
        #             plt.imshow(frame)
        #             plt.axis('off')
        #             plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        #             plt.close()
                    
        #         print(f"Saved: {sample_dir}")


        # if data_iter_step == 0:  # 10번마다 한번씩 시각화
        #     save_batch_frames(data, data_iter_step, name)


        # 첫 번째 배치만 확인하므로 종료
        # print(f"Data: {data}")
        # print(f"Input lengths: {input_lengths}")
        # print(f"Targets: {targets}")
        # print(f"Target lengths: {target_lengths}")
        # if loss_scaler is None: # For DeepSpeed
        #     if not no_amp:
        #         frames = frames.bfloat16() if bf16 else frames
        #     loss, outputs = train_class_batch(model, frames, targets, criterion)
        # else:
        #     with amp_autocast:
        #         loss, outputs = train_class_batch(model, frames, targets, criterion)

        data = data.bfloat16() if bf16 else data
        # loss_dict, total_outputs = train_class_batch(model, frames, scripts, audios, targets, criterions)

        # print(f"=============================Model type: {type(model)}, data: {data.shape}, data_iter_Step: {data_iter_step}, name: {name}")  # Should be DeepSpeedEngine

        loss, outputs = train_class_batch(model, data, input_lengths, targets, target_lengths, criterion)

        # print(f"Outputs shape: {outputs.shape}")
        # print(f"Input lengths: {input_lengths}")
        decoded_gloss_ids = model.video_backbone.decode(
            gloss_logits=outputs['gloss_logits'], beam_size=1, input_lengths=outputs['valid_len_out']
        )

        # print(f"===============================Decoded gloss ids: {decoded_gloss_ids}")
        # print(f"gloss_logits grad_fn: {outputs['gloss_logits'].grad_fn}")
        # print(f"gloss_probabilities_log grad_fn: {outputs['gloss_probabilities_log'].grad_fn}")
        # print(f"===============================Incorrect Decoded gloss ids: {incorrect_decoded_gloss_ids}")

        decoded_glosses = [
            clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(decoded_gloss_id)))
            for decoded_gloss_id in decoded_gloss_ids
        ]
        target_glosses = [
            clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(target[:tgt_len].tolist())))
            for target, tgt_len in zip(targets, target_lengths)
        ]

        # loss = args.loss_comb * loss_dict['combined'] + args.loss_video * loss_dict['video'] + args.loss_text *  loss_dict['text'] + args.loss_audio * loss_dict['audio']
        
        loss_value = loss.item()

        # print(f"===================================Loss, loss.requires_grad: {loss_value}")

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss) # Copy loss to loss_list
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

        # ====================================START========================================
        # if hasattr(model, "parameters") and hasattr(model, "config"):
        #     print(f"✅ Deepspeed is active! Model config: {model.config}")
        # else:
        #     print(f"❌ Deepspeed is NOT active!")
        # print(f"+++++++++++ BEFORE LOSS BACKWARD +++++++++++")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"✅ BEFORE BACKWARD: Gradient exists for {name}")

        # Check if gradients exist
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         print(f"🚨 WARNING: {name} is FROZEN and will NOT update!")
        #     if param.grad is None:
        #         print(f"🚨 WARNING: {name} STILL has NO gradient after loss.backward()!")
        #     else:
        #         print(f"✅ Gradient computed for {name}")

        # --- DEBUG: Check the loss and output gradient connection ---
        # print("DEBUG: Loss grad_fn:", loss.grad_fn)

        # If the model output dictionary contains 'gloss_logits', register a hook to print its gradient norm.
        # if 'gloss_logits' in outputs:
        #     outputs['gloss_logits'].register_hook(lambda grad: print("DEBUG: gloss_logits grad norm:", grad.norm()))

        # --- End of debug code ---
        # ====================================END========================================

        loss /= update_freq
        # if hasattr(model, "backward"):
        #     print("✅ DeepSpeed `model.backward()` exists, calling it...")
        #     model.backward(loss)
        #     # model.backward(loss)
        # else:
        #     print("❌ DeepSpeed `model.backward()` does NOT exist, calling `loss.backward()` directly...")
        # loss.backward(retain_graph=True)

        model.backward(loss)
        # =========================================STATR===================================

        # --- DeepSpeed Underlying Model Gradients Diagnostics ---
        # try:
        #     print("DEBUG: Checking underlying model parameter gradients via model.module.named_parameters()")
        #     for name, param in model.module.named_parameters():
        #         if param.requires_grad:
        #             grad_norm = param.grad.norm() if param.grad is not None else None
        #             print(f"  {name}: grad norm = {grad_norm}")
        # except Exception as e:
        #     print("DEBUG: Could not retrieve gradients from model.module:", e)
        # --- End Diagnostics ---
        # --- End DeepSpeed Gradient Diagnostics ---

        # print(f"_________________________________________Loss shape: {loss} | Loss requires_grad: {loss.requires_grad}")
        # print(f"Loss value: {loss.item()}, requires_grad: {loss.requires_grad}")
        # print(f"Loss dtype: {loss.dtype}, Loss shape: {loss.shape}")
        # print(f"Loss computation graph: {loss.grad_fn}")  # This should NOT be None


        # for param_group in optimizer.param_groups:
        #     print(f"Before Step - LR: {param_group['lr']} | Weight Decay: {param_group['weight_decay']}")
        # ===========================================ENMD=================================

        model.step()

        #=====================================STAERT=======================================
        # print(f"+++++++++++ AFTER LOSS BACKWARD FROM OPTIMIZER +++++++++++")
        # for group in optimizer.param_groups:
        #     for param in group['params']:
        #         if param.grad is None:
        #             param_name = [name for name, p in model.named_parameters() if p is param]
        #             print(f"🚨 Optimizer Warning: Parameter {param_name} has NO gradient!")
        #         else:
        #             param_name = [name for name, p in model.named_parameters() if p is param]
        #             print(f"✅ Optimizer has gradient for parameter {param_name}")

            # print(f"After Step - LR: {param_group['lr']} | Weight Decay: {param_group['weight_decay']}")


        # Check if gradients exist
        # print(f"+++++++++++ AFTER LOSS BACKWARD +++++++++++")
        # for name, param in model.named_parameters():
        #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        #     if param.grad is None:
        #         print(f"🚨 WARNING: {name} STILL has NO gradient after loss.backward()! {param.shape}")
        #     else:
        #         print(f"✅ Gradient computed for {name}")
        # =======================================END=====================================

        if (data_iter_step + 1) % update_freq == 0:
            # model.zero_grad()
            # Deepspeed will call step() & model.zero_grad() automatic
                    
            if model_ema is not None:
                model_ema.update(model)

        grad_norm = None
        loss_scale_value = get_loss_scale_for_deepspeed(model) # optimizer.cur_scale

        """print sample"""
        if (data_iter_step) % update_freq == 0:
            print("-" * 100)
            pred = decoded_glosses[0]
            gt = target_glosses[0]
            print(f"Predicted:\t{pred}")
            print(f"Ground truth:\t{gt}")
            print("-" * 100)
        torch.cuda.synchronize()
    
        # Output: logits -> update method 호출 시 자동으로 sigmoid 적용
        wer.update(decoded_glosses, target_glosses)

        wer_val = wer.compute().item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(wer=wer_val)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"]) # The product of the minimum value among the LRs with the layer decay value applied and the current LR
            max_lr = max(max_lr, group["lr"]) # The product of the maximum value (1) among the LRs to which the layer decay value is applied and the current LR
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
            log_writer.update(wer=wer_val, head="perf")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")


            # step + 1
            log_writer.set_step()
        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(
    data_loader, model, device, amp_autocast, ds=True, no_amp=False, bf16=False, gloss_tokenizer=None,
):
    criterion = nn.CTCLoss(gloss_tokenizer.silence_id, zero_infinity=True)
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
    wer = WordErrorRate(sync_on_compute=False)

    for batch in metric_logger.log_every(data_loader, 10, header):
        data, input_lengths, targets, target_lengths, _ = batch

        data = data.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        # compute outputs
        if ds:
            if not no_amp:
                data = data.bfloat16() if bf16 else data
            outputs = model(data, input_lengths)
            # loss_dict = {}

            # for output, (modal, criterion) in zip(outputs, criterions.items()):
            #     loss_dict[modal] = criterion(output, targets)
            loss = criterion(
                log_probs=outputs['gloss_probabilities_log'].permute(1, 0, 2), targets=targets,
                input_lengths=outputs['valid_len_out'], target_lengths=target_lengths
            )
        else:
            with amp_autocast:
                outputs = model(data, input_lengths)
                loss = criterion(
                    log_probs=outputs['gloss_probabilities_log'].permute(1, 0, 2), targets=targets,
                    input_lengths=outputs['valid_len_out'], target_lengths=target_lengths
                )
                # loss_dict = {}

                # for output, (modal, criterion) in zip(outputs, criterions.items()):
                #     loss_dict[modal] = criterion(output, targets)
        decoded_gloss_ids = model.video_backbone.decode(
            gloss_logits=outputs['gloss_logits'], beam_size=5, input_lengths=outputs['valid_len_out']
        )
        decoded_glosses = [
            clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(decoded_gloss_id)))
            for decoded_gloss_id in decoded_gloss_ids
        ]

        target_glosses = [
            clean_phoenix_2014(" ".join(gloss_tokenizer.convert_ids_to_tokens(target[:tgt_len].tolist())))
            for target, tgt_len in zip(targets, target_lengths)
        ]

        wer.update(decoded_glosses, target_glosses)

        batch_size = data.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['wer'].update(wer.compute().item(), n=batch_size)

        """print sample"""
        print("-" * 40)
        pred = decoded_glosses[0]
        gt = target_glosses[0]
        print(f"Predicted:\t{pred}")
        print(f"Ground truth:\t{gt}")
        print("-" * 40)
        
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
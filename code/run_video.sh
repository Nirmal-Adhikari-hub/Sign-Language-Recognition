# !/bin/bash
export OMP_NUM_THREADS=1

N_GPUS=2
MASTER_PORT=$((12000 + $RANDOM % 20000))

task=s2g
modal=video
dataset=phoenix-2014
batch_size=1
epochs=50
warmup_epochs=5
update_freq=16
lr=1e-3
min_lr=1e-6
warmup_lr=1e-6
T_0=50
T_mul=1
lr_gamma=0.5
num_frames=256
classes=1235

torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" run.py \
        --task "$task" \
        --modal "$modal" \
        --batch_size "$batch_size" \
        --epochs "$epochs" \
        --update_freq "$update_freq" \
        --save_ckpt_freq 100 \
        --pretrained \
        --patch_size 16 16 \
        --in_chans 3 \
        --embed_dim 384 \
        --depth 12 \
        --drop_rate 0.1 \
        --drop_path_rate 0.1 \
        --head_drop_rate 0.1 \
        --model_ema \
        --opt "adamw" \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.001 \
        --lr "$lr" \
        --min_lr "$min_lr" \
        --warmup_lr "$warmup_lr" \
        --warmup_epochs "$warmup_epochs" \
        --T_0 "$T_0" \
        --T_mul "$T_mul" \
        --lr_gamma "$lr_gamma" \
        --aug_size 210 210 \
        --t_min 0.5 \
        --t_max 1.5 \
        --video_size 224 224 \
        --metadata_path "/nas/Dataset/Phoenix/" \
        --gloss_to_id_path "/nas/Dataset/Phoenix/gloss2ids.pkl" \
        --video_path "/nas/Dataset/Phoenix/phoenix-2014-videos.zip" \
        --split "," \
        --n_classes $classes \
        --num_frames "$num_frames" \
        --output_dir "/nas/Nirmal/workspace/slr_results/videomae" \
        --log_dir "/nas/Nirmal/workspace/slr_results/videomae" \
        --resume "/nas/Nirmal/workspace/slr_results/videomae" \
        --dataset "$dataset" \
        --auto_resume \
        --save_ckpt \
        --num_workers 12 \
        --distributed \
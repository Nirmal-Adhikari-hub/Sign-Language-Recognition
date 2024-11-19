# !/bin/bash
export OMP_NUM_THREADS=1

N_GPUS=2
MASTER_PORT=$((12000 + $RANDOM % 20000))

modal=video
batch_size=4
epochs=50
warmup_epochs=5
update_freq=64
patch_size="16 16"
lr=4e-4
min_lr=1e-6
warmup_lr=1e-6
T_0=50
T_mul=1
lr_gamma=0.5
n_frames=16
classes=22

torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" run.py \
        --modal "$modal" \
        --batch_size "$batch_size" \
        --epochs "$epochs" \
        --update_freq "$update_freq" \
        --video_size 224 224 \
        --aug_size 256 400 \
        --save_ckpt_freq 100 \
        --pretrained \
        --tubelet_size 1 \
        --drop 0.1 \
        --fc_drop_rate 0.3 \
        --drop_path 0.1 \
        --model_ema \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.001 \
        --lr "$lr" \
        --layer_decay 0.9 \
        --min_lr "$min_lr" \
        --warmup_lr "$warmup_lr" \
        --warmup_epochs "$warmup_epochs" \
        --T_0 "$T_0" \
        --T_mul "$T_mul" \
        --lr_gamma "$lr_gamma" \
        --aa "rand-m9-mstd0.5" \
        --metadata_path "/home/ubuntu/workspace/mc/data/metadata" \
        --video_path "/home/ubuntu/workspace/mc/data/videos" \
        --split "," \
        --n_classes $classes \
        --patch_size $patch_size \
        --n_frames "$n_frames" \
        --trimmed 180 \
        --time_stride 60 \
        --output_dir "/home/ubuntu/workspace/mc/data/exp/output_A100/$modal" \
        --log_dir "/home/ubuntu/workspace/mc/data/exp/output_A100/$modal" \
        --resume "/home/ubuntu/workspace/mc/data/exp/output_A100/$modal" \
        --data_set "Condensed_movies" \
        --auto_resume \
        --save_ckpt \
        --num_workers 12 \
        --distributed \
        --dist_eval \

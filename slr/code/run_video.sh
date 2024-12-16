# !/bin/bash
export OMP_NUM_THREADS=1

N_GPUS=2
MASTER_PORT=$((12000 + $RANDOM % 20000))

modal=video
batch_size=2
epochs=50
warmup_epochs=5
update_freq=1
patch_size="4 4" 
lr=1e-3
min_lr=1e-6
warmup_lr=1e-6
T_0=50
T_mul=1
lr_gamma=0.5
n_frames=400
classes=1236

torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" run.py \
        --modal "$modal" \
        --batch_size "$batch_size" \
        --epochs "$epochs" \
        --update_freq "$update_freq" \
        --video_size 224 224 \
        --aug_size 210 210 \
        --save_ckpt_freq 100 \
        --pretrained \
        --tubelet_size 2 \
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
        --t_min 0.5 \
        --t_max 1.5 \
        --aa "rand-m7-mstd0.25" \
        --metadata_path "/nas/Dataset/Phoenix" \
        --gloss_to_id_path "/nas/Dataset/Phoenix/gloss2ids.pkl" \
        --video_path "/nas/Dataset/Phoenix/phoenix-2014-videos.zip" \
        --split "," \
        --n_classes $classes \
        --patch_size $patch_size \
        --n_frames "$n_frames" \
        --output_dir "/home/kks/workspace/slr/code/results" \
        --log_dir "/home/kks/workspace/slr/code/results" \
        --resume "/home/kks/workspace/slr/code/results" \
        --dataset "phoenix-2014" \
        --auto_resume \
        --save_ckpt \
        --num_workers 12 \
        --distributed \

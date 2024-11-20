# !/bin/bash
export OMP_NUM_THREADS=1

N_GPUS=2
MASTER_PORT=$((12000 + $RANDOM % 20000))

modal=audio
batch_size=8
epochs=50
warmup_epochs=5
update_freq=128
patch_size="16 16"
lr=1e-4
min_lr=1e-6
warmup_lr=1e-6
T_0=50
T_mul=1
lr_gamma=0.5
classes=22

torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" run.py \
        --modal "$modal" \
        --batch_size "$batch_size" \
        --epochs "$epochs" \
        --update_freq "$update_freq" \
        --audio_size 32 18000 \
        --save_ckpt_freq 100 \
        --pretrained \
        --drop 0.2 \
        --fc_drop_rate 0.2 \
        --drop_path 0.2 \
        --head_drop 0.2 \
        --model_ema \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --weight_decay_start 0.05 \
        --weight_decay_end 0.05 \
        --lr "$lr" \
        --layer_decay 0.75\
        --min_lr "$min_lr" \
        --warmup_lr "$warmup_lr" \
        --warmup_epochs "$warmup_epochs" \
        --T_0 "$T_0" \
        --T_mul "$T_mul" \
        --lr_gamma "$lr_gamma" \
        --metadata_path "/home/ubuntu/workspace/data/metadata" \
        --video_path "/home/ubuntu/workspace/data/videos" \
        --audio_path "/home/ubuntu/workspace/data/audios_32" \
        --split "," \
        --n_classes $classes \
        --trimmed 180 \
        --time_stride 60 \
        --output_dir "/home/ubuntu/workspace/data/exp/output_A100/$modal-test" \
        --log_dir "/home/ubuntu/workspace/data/exp/output_A100/$modal-test" \
        --resume "/home/ubuntu/workspace/data/exp/output_A100/$modal-test" \
        --data_set "Condensed_movies" \
        --auto_resume \
        --save_ckpt \
        --num_workers 12 \
        --distributed \
        --dist_eval \
        # --eval \
        # --test_best \
        # --use_checkpoint \
        # --checkpoint_num 24 \
        # --output_dir "/nas/kks/data/condensed_movies/exp/output_A100/$model-bs_$batch_size-epochs_$epochs-w_epochs_$warmup_epochs-freq_$update_freq-lr_$lr-minlr_$min_lr-wlr_$warmup_lr-patch_size$patch_size-nframes_$n_frames-nSamples_$n_samples-gammaPos_$gamma_pos-gammaNeg_$gamma_neg-clip_$clip-classes_$classes-$precision-pretrained_$pretrained" \
        # --log_dir "/nas/kks/data/condensed_movies/exp/output_A100/$model-bs_$batch_size-epochs_$epochs-w_epochs_$warmup_epochs-freq_$update_freq-lr_$lr-minlr_$min_lr-wlr_$warmup_lr-patch_size$patch_size-nframes_$n_frames-nSamples_$n_samples-gammaPos_$gamma_pos-gammaNeg_$gamma_neg-clip_$clip-classes_$classes-$precision-pretrained_$pretrained" \
        # --resume "/nas/kks/data/condensed_movies/exp/output_A100/$model-bs_$batch_size-epochs_$epochs-w_epochs_$warmup_epochs-freq_$update_freq-lr_$lr-minlr_$min_lr-wlr_$warmup_lr-patch_size$patch_size-nframes_$n_frames-nSamples_$n_samples-gammaPos_$gamma_pos-gammaNeg_$gamma_neg-clip_$clip-classes_$classes-$precision-pretrained_$pretrained" \
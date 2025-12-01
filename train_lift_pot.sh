#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/act-ora-split-${task_name}/${task_config}-${expert_data_num} \
    --policy_class ACTOracleSplit \
    --kl_weight 1 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 12 \
    --dim_feedforward 4096 \
    --num_epochs 10000 \
    --lr 5e-5 \
    --seed ${seed} \
    --wandb \
    --wandb_project robotwin-act-${task_name} \
    --wandb_run_name ora-split_${task_name}_pc \
    --enable_wm_augmentation \
    --wm_augmentation_prob 0.25 \
    --wm_config_dir /home/joe/womap/configs \
    --wm_config_name train_robotwin_lift_pot \
    --wm_ckpt_path /home/joe/RoboTwin/coopwm_ckpt/lift_pot/lift_pot_coruscant_2025_11_19_08_58_40-ep80.pth.tar \
    --wm_ckpt_dir /home/joe/RoboTwin/coopwm_ckpt/lift_pot \
    --wm_split_ratio 0.4

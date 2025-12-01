#!/bin/bash

# Example:
#   bash policy/ACTTwin/eval.sh handover_block demo_clean /home/joe/RoboTwin/policy/ACTTwin/checkpoints/act_twin_handover_block 50 0 0

policy_name=ACTOracleSplit

task_name=${1}
task_config=${2}
CKPT_DIR=${3}        
expert_data_num=${4}
seed=${5}
gpu_id=${6}
eval_config=${7}

DEBUG=False

echo -e "\033[33mGPU id (to use): ${gpu_id}\033[0m"

cd ../..

echo "🚀 Starting evaluation with reduced verbosity..."
echo "Task: $task_name | Config: $task_config | Eval Type: $eval_config"
echo "Checkpoint: $CKPT_DIR"
echo "=" * 60

# Set environment variables to reduce verbosity
export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH=/home/joe/womap/src:$PYTHONPATH
export HYDRA_FULL_ERROR=0
export LOG_LEVEL=WARNING

# Set logging levels
export TF_CPP_MIN_LOG_LEVEL=2  # TensorFlow logging
export OMP_NUM_THREADS=1

# Override checkpoints to run 
# Note: Even if these checkpoints don't exist the run will continue with a random policy 
LEFT_CKPT=${LEFT_CKPT:-left_policy_best.ckpt}
RIGHT_CKPT=${RIGHT_CKPT:-right_policy_best.ckpt} 


# Set default split ratio if not provided
WM_SPLIT_RATIO=${WM_SPLIT_RATIO:-0.4}

# Run evaluation with filtered output
python script/eval_policy.py \
  --config policy/${policy_name}/deploy_policy.yml \
  --overrides \
  --eval_type ${eval_config} \
  --task_name ${task_name} \
  --task_config ${task_config} \
  --ckpt_dir ${CKPT_DIR} \
  --seed ${seed} \
  --temporal_agg false \
  --left_ckpt_name ${LEFT_CKPT} \
  --right_ckpt_name ${RIGHT_CKPT} \
  --wm_split_ratio ${WM_SPLIT_RATIO} \
  # 2>&1 | grep -v -E "(INFO:|DEBUG:|WARNING:|FutureWarning:|UserWarning:)"


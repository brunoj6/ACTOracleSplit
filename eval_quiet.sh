#!/bin/bash

# Alternative eval script with better logging control
# Usage: bash eval_quiet.sh grab_roller demo_clean /path/to/checkpoints 50 0 0 coopwm

policy_name=ACTOracleSplit

task_name=${1}
task_config=${2}
CKPT_DIR=${3}        
expert_data_num=${4}
seed=${5}
gpu_id=${6}
eval_config=${7}

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

echo -e "\033[33mGPU id (to use): ${gpu_id}\033[0m"

cd ../..

# (optional) allow overriding which ckpts to load
LEFT_CKPT=${LEFT_CKPT:-left_policy_last_4000.ckpt}
RIGHT_CKPT=${RIGHT_CKPT:-right_policy_last_4000.ckpt}

echo "Loading checkpoints: $LEFT_CKPT, $RIGHT_CKPT"
echo "Starting evaluation..."
echo ""

# Run evaluation with minimal output
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
  2>&1 | grep -v -E "(INFO:|DEBUG:|WARNING:|FutureWarning:|UserWarning:|Loading world model weights|Successfully loaded)" | \
  grep -E "(Success|Fail|episode|Success rate|Loading|✅|❌|⚠️|🎬|Saved.*video)"

echo ""
echo "🎉 Evaluation completed!"

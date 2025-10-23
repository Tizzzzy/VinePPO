#!/bin/bash

# --- 1. Set Configuration Variables ---
echo "Setting up configuration..."

export MASTER_PORT=31337

# Set the base config for Rho PPO on GSM8K + single GPU config
export CONFIGSTR="configs/polIter_rho1bSft2_ppo_GSM8K.jsonnet,\
configs/trainers/devBz16.jsonnet"

# Set your desired output directory
export APP_DIRECTORY="experiments/rho_ppo_gsm8k_single_gpu"

# Set a seed (optional, from README)
export APP_SEED="2746318213"

# Optional: Set this if you use Weights & Biases.
# If not, you can delete this line.
export WANDB_RUN_ID="1b4a3ed40eeaae71d5f0fbbd8a220395978e7520"


# --- 2. Run the Experiment ---

# Get the number of GPUs (this should be 1 for your setup)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPU(s)."

# 1. Run the training
echo "--- Starting Training ---"
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop

# 2. Run the evaluation
echo "--- Starting Evaluation ---"
deepspeed --no_local_rank --num_gpus=$NUM_GPUS   \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_evaluation

echo "--- Experiment Complete ---"

# chmod +x run_single_gpu.sh
# ./run_single_gpu.sh
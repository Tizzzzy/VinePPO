echo "Setting up configuration..."

export MASTER_PORT=31337
export CONFIGSTR="configs/polIter_rho1bSft2_ppo_GSM8K.jsonnet,\
configs/trainers/devBz16.jsonnet"
export APP_DIRECTORY="experiments/rho_ppo_gsm8k_single_gpu"
export APP_SEED="2746318213"
export WANDB_RUN_ID="1b4a3ed40eeaae71d5f0fbbd8a220395978e7520"

# --- 2. Run the Experiment ---

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPU(s)."

# 1. Run evaluation on the RAW SFT model (BEFORE training)
echo "--- Starting Evaluation on RAW Model ---"
deepspeed --no_local_rank --num_gpus=$NUM_GPUS   \
         src/treetune/main.py --configs "$CONFIGSTR" \
           run_evaluation

echo "--- Experiment Complete ---"
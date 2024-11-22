#!/bin/bash
#SBATCH --job-name=train_ae        # Job name
#SBATCH --nodes=1                  # Number of nodes

# Load modules (adjust based on your system)
# module load python/3.8             # Example Python module
# module load cuda/11.8              # Example CUDA module

# Activate your environment
source cmdenv/bin/activate

export CACHE_DIRECTORY=.cache
export WANDB_API_KEY=1acbb825d8f7573fd546423424fd19abc40dccdb

# Run
python train_motion_diffusion.py \ 
  --nnodes=[NUM_NODES] \ 
  --nproc_per_node=[NUM_GPUS] 
    --dataset-name UCF101 \ 
    --data-path /data/UCF-101 \
    --global-batch-size [BATCH_SIZE] \   
    --results-dir [LOG_DIRECTORY]
    --mode pixel \
    --ckpt-every 20000
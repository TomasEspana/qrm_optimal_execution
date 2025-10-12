#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/relative_heatmap/test.txt

module purge
module load anaconda3/2025.6
module load cudatoolkit/12.8

conda activate /home/te6653/.conda/envs/rl

echo "START $(date) on $HOSTNAME"; echo "SLURM_JOBID=$SLURM_JOB_ID NODES=$SLURM_JOB_NUM_NODES TASKS=$SLURM_NTASKS"
export WANDB_MODE=offline
export WANDB_DIR=/scratch/network/te6653/qrm_optimal_execution/wandb_runs/$SLURM_JOB_ID
export WANDB_CACHE_DIR=/scratch/network/te6653/qrm_optimal_execution/wandb_cache
srun python /home/te6653/research/qrm_optimal_execution/scripts/run_test_heatmap.py
echo "END $(date)"

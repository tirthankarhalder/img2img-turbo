#!/bin/bash
#SBATCH --job-name=diagnostic
#SBATCH --output=logs/diag-%j.out
#SBATCH --error=logs/diag-%j.err
#SBATCH --partition=gpupart_l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

echo "=== Diagnostic Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""

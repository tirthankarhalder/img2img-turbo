#!/bin/bash
#SBATCH --job-name=minimal
#SBATCH --output=logs/minimal-%j.out
#SBATCH --error=logs/minimal-%j.err
#SBATCH --partition=gpu_l40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

set -x  # Debug mode - print every command

echo "START: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

cd /home/du2/22CS30064/img2img-turbo
echo "Current directory: $(pwd)"

ls -lh test_pix2pix_turbo.py || echo "test_pix2pix_turbo.py NOT FOUND"
ls -lh output/pix2pix_turbo_l40/checkpoints/model_28501.pkl || echo "MODEL NOT FOUND"

source /home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh
echo "Conda loaded"

conda activate img2img-turbo
echo "Environment activated"

python --version
echo "Python works"

python -c "import torch; print(torch.__version__)"
echo "PyTorch works"

echo "END: $(date)"

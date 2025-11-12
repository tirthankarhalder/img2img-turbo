#!/bin/bash
#SBATCH -J pix2pix-test
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err
#SBATCH -p gpu_l40
#SBATCH --exclude=gnode5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00

cd /home/du2/22CS30064/img2img-turbo
source /home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh
conda activate img2img-turbo

echo "Job: $SLURM_JOB_ID on $(hostname) at $(date)"

python -u test_pix2pix_turbo.py \
    --model_path output/pix2pix_turbo_l40/checkpoints/model_28501.pkl \
    --input_dir data/dataset_mmWave/test_A \
    --output_dir test_results_28501 \
    --save_comparison \
    --device cuda

echo "Completed at $(date)"

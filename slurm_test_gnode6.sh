#!/bin/bash
#SBATCH --job-name=test-gnode6
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err
#SBATCH --partition=gpu_l40
#SBATCH --nodelist=gnode6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd /home/du2/22CS30064/img2img-turbo
source /home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh
conda activate img2img-turbo

echo "Job started on $(hostname) at $(date)"

python -u test_pix2pix_turbo.py \
    --model_path output/pix2pix_turbo_l40/checkpoints/model_28501.pkl \
    --input_dir data/dataset_mmWave/test_A \
    --output_dir test_results_28501 \
    --save_comparison \
    --device cuda

echo "Job finished at $(date)"

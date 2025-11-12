#!/bin/bash
#SBATCH --job-name=test-now
#SBATCH --output=logs/img-%j.out
#SBATCH --error=logs/img-%j.err
#SBATCH --partition=gpupart_l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd /home/du2/22CS30064/img2img-turbo
source /home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh
conda activate img2img-turbo

echo "Job started at $(date)"
echo "Running on: $SLURM_NODELIST"

python src/inference_paired.py --model_path "output/pix2pix_turbo_l40/checkpoints/model_28501.pkl" \
    --input_image "data/dataset_mmWave/test_A/000000.png" \
    --prompt "polishing shifted by 340 degrees at Frame no. 62" \
    --output_dir "outputs"

echo "Job completed at $(date)"

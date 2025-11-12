#!/bin/bash
#SBATCH -J direct-test
#SBATCH -o /home/du2/22CS30064/img2img-turbo/direct_out_%j.txt
#SBATCH -e /home/du2/22CS30064/img2img-turbo/direct_err_%j.txt
#SBATCH -p gpu_l40
#SBATCH --gres=gpu:1
#SBATCH -t 00:02:00

echo "Test from $(hostname)"
date

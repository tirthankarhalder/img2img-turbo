#!/bin/bash
#SBATCH -J default-test
#SBATCH -p gpu_l40
#SBATCH --gres=gpu:1
#SBATCH -t 00:02:00

echo "Test from $(hostname)"
date
pwd
ls -lh

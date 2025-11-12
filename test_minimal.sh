#!/bin/bash
#SBATCH -J test
#SBATCH -o test_%j.out
#SBATCH -e test_%j.err
#SBATCH -p gpu_l40
#SBATCH --gres=gpu:1
#SBATCH -t 00:02:00

echo "Hello"

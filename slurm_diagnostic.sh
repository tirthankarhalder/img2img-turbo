#!/bin/bash
#SBATCH --job-name=diagnostic
#SBATCH --output=logs/diag-%j.out
#SBATCH --error=logs/diag-%j.err
#SBATCH --partition=gpu_l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

echo "=== Diagnostic Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""

cd /home/du2/22CS30064/img2img-turbo
source /home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh
conda activate img2img-turbo

echo "=== GPU Information ==="
nvidia-smi
echo ""

echo "=== Python & PyTorch ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "=== GPU Compatibility Check ==="
python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0)}'); print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')"
echo ""

echo "=== Testing Model Load ==="
python << 'PYEOF'
import sys
sys.path.append('src')
try:
    from pix2pix_turbo import Pix2Pix_Turbo
    print("✓ Import successful")
    
    print("Loading model...")
    model = Pix2Pix_Turbo(pretrained_path="output/pix2pix_turbo_l40/checkpoints/model_28501.pkl")
    print("✓ Model loaded")
    
    print("Moving to GPU...")
    model = model.cuda()
    print("✓ Model on GPU")
    
    print("SUCCESS: Model is ready!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
PYEOF

echo ""
echo "=== Diagnostic Complete ==="

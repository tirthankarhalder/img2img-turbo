#!/bin/bash

# Complete setup script for pix2pix-turbo testing
# Run this once to set up everything

set -e

REPO_PATH="/home/du2/22CS30064/img2img-turbo"

echo "============================================================"
echo "Pix2Pix-Turbo Testing Environment Setup"
echo "============================================================"
echo ""

# Navigate to repository
if [ ! -d "$REPO_PATH" ]; then
    echo "ERROR: Repository not found at $REPO_PATH"
    exit 1
fi

cd "$REPO_PATH"

# Verify repository structure
echo "[1/6] Verifying repository structure..."
if [ ! -d "src" ]; then
    echo "ERROR: src/ directory not found"
    exit 1
fi

if [ ! -f "src/pix2pix_turbo.py" ]; then
    echo "ERROR: src/pix2pix_turbo.py not found"
    exit 1
fi
echo "✓ Repository structure verified"
echo ""

# Create necessary directories
echo "[2/6] Creating directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p evaluation_archive
echo "✓ Created: logs/, outputs/, evaluation_archive/"
echo ""

# Scan for available datasets and checkpoints
echo "[3/6] Scanning for trained models..."
echo ""

if [ -d "output/pix2pix_turbo" ]; then
    FOUND_MODELS=0
    
    for dataset_dir in output/pix2pix_turbo/*/; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            
            if [ -d "${dataset_dir}checkpoints" ]; then
                checkpoint_count=$(ls "${dataset_dir}checkpoints"/*.pkl 2>/dev/null | wc -l)
                
                if [ $checkpoint_count -gt 0 ]; then
                    echo "Dataset: $dataset_name"
                    echo "  Location: $dataset_dir"
                    echo "  Checkpoints: $checkpoint_count"
                    
                    # Show first and last checkpoint
                    first_ckpt=$(ls "${dataset_dir}checkpoints"/*.pkl 2>/dev/null | head -1)
                    last_ckpt=$(ls "${dataset_dir}checkpoints"/*.pkl 2>/dev/null | tail -1)
                    
                    echo "  First: $(basename $first_ckpt)"
                    echo "  Last: $(basename $last_ckpt)"
                    echo ""
                    
                    FOUND_MODELS=1
                fi
            fi
        fi
    done
    
    if [ $FOUND_MODELS -eq 0 ]; then
        echo "⚠ No trained models found"
        echo "  Train a model first before testing"
    fi
else
    echo "⚠ No output/pix2pix_turbo directory found"
    echo "  Train a model first before testing"
fi
echo ""

# Scan for test data
echo "[4/6] Scanning for test data..."
echo ""

if [ -d "data" ]; then
    FOUND_DATA=0
    
    for test_dir in data/*/test_A; do
        if [ -d "$test_dir" ]; then
            dataset_name=$(basename $(dirname "$test_dir"))
            image_count=$(ls "$test_dir"/*.png 2>/dev/null | wc -l)
            
            if [ $image_count -gt 0 ]; then
                echo "Dataset: $dataset_name"
                echo "  Test images: $image_count"
                echo "  Location: $test_dir"
                echo ""
                
                FOUND_DATA=1
            fi
        fi
    done
    
    if [ $FOUND_DATA -eq 0 ]; then
        echo "⚠ No test data found"
    fi
else
    echo "⚠ No data/ directory found"
fi
echo ""

# Check conda environment
echo "[5/6] Checking environment..."

if [ -f "/home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh
    
    if conda env list | grep -q "img2img-turbo"; then
        echo "✓ Conda environment 'img2img-turbo' found"
        
        # Activate and check packages
        conda activate img2img-turbo
        
        if python -c "import torch; import PIL" 2>/dev/null; then
            echo "✓ Required packages installed"
            python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
        else
            echo "⚠ Some required packages may be missing"
        fi
    else
        echo "⚠ Conda environment 'img2img-turbo' not found"
        echo "  Create environment: conda env create -f environment.yml"
    fi
else
    echo "⚠ Conda not found at expected location"
fi
echo ""

# Make scripts executable
echo "[6/6] Making scripts executable..."

if [ -f "slurm_test_pix2pix.sh" ]; then
    chmod +x slurm_test_pix2pix.sh
    echo "✓ slurm_test_pix2pix.sh"
fi

if [ -f "slurm_batch_test.sh" ]; then
    chmod +x slurm_batch_test.sh
    echo "✓ slurm_batch_test.sh"
fi

if [ -f "run_tests_example.sh" ]; then
    chmod +x run_tests_example.sh
    echo "✓ run_tests_example.sh"
fi

if [ -f "batch_evaluate.sh" ]; then
    chmod +x batch_evaluate.sh
    echo "✓ batch_evaluate.sh"
fi

echo ""

# Generate quick start guide
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""

cat > QUICKSTART.txt << 'EOF'
Pix2Pix-Turbo Testing Quick Start
==================================

INTERACTIVE TESTING (on login node)
------------------------------------
1. Quick test:
   python simple_test.py \
       output/pix2pix_turbo_l40/checkpoints/model_28501.pkl \
       data/dataset_mmWave/test_A/image.png \
       "prompt"

2. Full test:
   python test_pix2pix_turbo.py \
       --model_path output/pix2pix_turbo_l40/checkpoints/model_28501.pkl \
       --input_dir data/dataset_mmWave/test_A \
       --output_dir outputs/test_results \
       --save_comparison

SLURM TESTING (recommended for large tests)
--------------------------------------------
1. Edit configuration:
   nano slurm_test_pix2pix.sh
   
   Change:
   - export CHECKPOINT_NAME="model_28501.pkl"
   - export TEST_A_DIR="data/dataset_mmWave/test_A"

2. Submit job:
   sbatch slurm_test_pix2pix.sh

3. Monitor:
   squeue -u $USER
   tail -f logs/test-JOBID.out

4. View results:
   cat test_results_*/test_report.txt

BATCH TESTING (test multiple checkpoints)
------------------------------------------
1. Edit configuration:
   nano slurm_batch_test.sh
   
   Change:
   - CHECKPOINTS="model_10000.pkl model_20000.pkl model_28501.pkl"

2. Submit:
   sbatch slurm_batch_test.sh

3. Results in:
   batch_test_results_l40/

USEFUL COMMANDS
---------------
# List checkpoints
ls -lh output/pix2pix_turbo_l40/checkpoints/

# List test images
ls -lh data/dataset_mmWave/test_A/

# Check job status
squeue -u $USER

# Cancel job
scancel JOBID

# View job details
seff JOBID

# Transfer results to local
scp -r username@cluster:/home/du2/22CS30064/img2img-turbo/test_results_* ./

TROUBLESHOOTING
---------------
# CUDA OOM: Use CPU
Add to script: --device cpu

# Module not found: Check path
cd /home/du2/22CS30064/img2img-turbo
pwd

# Environment issues: Reactivate
conda activate img2img-turbo

For detailed help, see:
- SLURM_TESTING_GUIDE.md
- YOUR_SYSTEM_GUIDE.md
- TESTING_README.md
EOF

echo "Quick start guide created: QUICKSTART.txt"
echo ""

# Show what to do next
echo "Next Steps:"
echo "==========="
echo ""
echo "1. For interactive testing (small tests):"
echo "   python simple_test.py MODEL_PATH INPUT_IMAGE PROMPT"
echo ""
echo "2. For SLURM testing (recommended):"
echo "   a. Edit: nano slurm_test_pix2pix.sh"
echo "   b. Submit: sbatch slurm_test_pix2pix.sh"
echo "   c. Monitor: tail -f logs/test-JOBID.out"
echo ""
echo "3. For batch testing multiple checkpoints:"
echo "   a. Edit: nano slurm_batch_test.sh"
echo "   b. Submit: sbatch slurm_batch_test.sh"
echo ""
echo "4. Quick reference:"
echo "   cat QUICKSTART.txt"
echo ""
echo "5. Detailed guides:"
echo "   - SLURM_TESTING_GUIDE.md (SLURM commands)"
echo "   - YOUR_SYSTEM_GUIDE.md (your specific setup)"
echo "   - TESTING_README.md (comprehensive testing)"
echo ""

# Create a convenient alias suggestion
cat > setup_aliases.sh << 'EOF'
#!/bin/bash
# Convenient aliases for testing
# Add these to your ~/.bashrc

alias cdimg='cd /home/du2/22CS30064/img2img-turbo'
alias test-quick='cd /home/du2/22CS30064/img2img-turbo && python simple_test.py'
alias test-full='cd /home/du2/22CS30064/img2img-turbo && python test_pix2pix_turbo.py'
alias check-jobs='squeue -u $USER'
alias watch-jobs='watch -n 5 squeue -u $USER'

echo "Aliases loaded! Use:"
echo "  cdimg       - Go to img2img-turbo directory"
echo "  test-quick  - Quick test command"
echo "  test-full   - Full test command"
echo "  check-jobs  - Check your SLURM jobs"
echo "  watch-jobs  - Watch jobs with auto-refresh"
EOF

chmod +x setup_aliases.sh

echo "Optional: Add convenient aliases"
echo "   source setup_aliases.sh"
echo "   Or add to ~/.bashrc: cat setup_aliases.sh >> ~/.bashrc"
echo ""

echo "============================================================"
echo "Ready to test!"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=test-pix2pix     # Job name
#SBATCH --output=logs/test-%j.out   # Standard output log
#SBATCH --error=logs/test-%j.err    # Error log
#SBATCH --partition=gpupart_l40        # Partition/queue name
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --mem=32G                   # Memory per node
#SBATCH --time=20:00:00             # Max runtime (2 hours)

# ============================================================================
# CONFIGURATION - Edit these variables for your test
# ============================================================================

# Checkpoint to test
export CHECKPOINT_NAME="model_28501.pkl"

# Dataset name and paths
export DATASET_NAME="dataset_mmWave"
export TEST_A_DIR="/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave/test_A"
export TEST_B_DIR="/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave/test_B"
export PROMPTS_JSON="/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave/test_prompts.json"

# Model path (modify if different from training output)
export MODEL_PATH="/home/du2/22CS30064/img2img-turbo/output/pix2pix_turbo_l40/checkpoints/${CHECKPOINT_NAME}"

# Output directory for test results
export OUTPUT_DIR="test_results_l40_${CHECKPOINT_NAME%.pkl}"

# Default prompt (if prompts.json not available)
export DEFAULT_PROMPT=""

# ============================================================================
# Setup
# ============================================================================

echo "============================================================"
echo "Pix2Pix-Turbo Testing Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# Navigate to project directory
cd /home/du2/22CS30064/img2img-turbo || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda
source /home/du2/22CS30064/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate img2img-turbo

# Verify CUDA is available
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Display configuration
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_NAME"
echo "  Model path: $MODEL_PATH"
echo "  Test A dir: $TEST_A_DIR"
echo "  Test B dir: $TEST_B_DIR"
echo "  Prompts: $PROMPTS_JSON"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Verify files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_PATH"
    echo "Available checkpoints:"
    ls -lh output/pix2pix_turbo_l40/checkpoints/*.pkl 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

if [ ! -d "$TEST_A_DIR" ]; then
    echo "ERROR: Test A directory not found: $TEST_A_DIR"
    exit 1
fi

echo "✓ All required files verified"
echo ""

# ============================================================================
# Run Testing
# ============================================================================

echo "============================================================"
echo "Running Comprehensive Test"
echo "============================================================"
echo ""

# Run the test script
python -u src/test_pix2pix_turbo.py \
    --model_path "$MODEL_PATH" \
    --input_dir "$TEST_A_DIR" \
    --gt_dir "$TEST_B_DIR" \
    --prompts_json "$PROMPTS_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --save_comparison \
    --device cuda


TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Testing failed with exit code $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi

echo ""
echo "✓ Testing completed successfully"
echo ""

# ============================================================================
# Create Visualization Grids
# ============================================================================

echo "============================================================"
echo "Creating Visualization Grids"
echo "============================================================"
echo ""

python -u create_grid.py \
    --mode comparison \
    --input_dir "$TEST_A_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output "${OUTPUT_DIR}/comparison_grid.png" \
    --grid_size 4 4 \
    --image_size 256

GRID_EXIT_CODE=$?

if [ $GRID_EXIT_CODE -eq 0 ]; then
    echo "✓ Visualization grid created"
else
    echo "WARNING: Grid creation failed, but continuing..."
fi

echo ""

# ============================================================================
# Generate Report
# ============================================================================

echo "============================================================"
echo "Generating Test Report"
echo "============================================================"
echo ""

REPORT_FILE="${OUTPUT_DIR}/test_report.txt"

{
    echo "Pix2Pix-Turbo Test Report"
    echo "=========================="
    echo ""
    echo "Job Information:"
    echo "  Job ID: $SLURM_JOB_ID"
    echo "  Node: $SLURM_NODELIST"
    echo "  Partition: gpu_h100"
    echo ""
    echo "Test Configuration:"
    echo "  Checkpoint: $CHECKPOINT_NAME"
    echo "  Model path: $MODEL_PATH"
    echo "  Dataset: $DATASET_NAME"
    echo "  Test directory: $TEST_A_DIR"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""
    echo "Test Execution:"
    echo "  Started: $(date)"
    
    # Count outputs
    NUM_OUTPUTS=$(ls "${OUTPUT_DIR}"/*_output.png 2>/dev/null | wc -l)
    NUM_COMPARISONS=$(ls "${OUTPUT_DIR}"/*_comparison.png 2>/dev/null | wc -l)
    
    echo "  Images generated: $NUM_OUTPUTS"
    echo "  Comparisons created: $NUM_COMPARISONS"
    echo ""
    echo "Output Files:"
    echo "  Individual outputs: ${OUTPUT_DIR}/*_output.png"
    echo "  Comparisons: ${OUTPUT_DIR}/*_comparison.png"
    echo "  Visualization grid: ${OUTPUT_DIR}/comparison_grid.png"
    echo ""
    echo "Next Steps:"
    echo "  1. Review outputs in: $OUTPUT_DIR"
    echo "  2. Check comparison_grid.png for overview"
    echo "  3. Compare with ground truth in: $TEST_B_DIR"
    
} > "$REPORT_FILE"

cat "$REPORT_FILE"
echo ""
echo "✓ Report saved to: $REPORT_FILE"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================"
echo "Job Complete!"
echo "============================================================"
echo ""
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - Output images: ${OUTPUT_DIR}/*_output.png"
echo "  - Comparisons: ${OUTPUT_DIR}/*_comparison.png"
echo "  - Grid: ${OUTPUT_DIR}/comparison_grid.png"
echo "  - Report: ${OUTPUT_DIR}/test_report.txt"
echo ""
echo "Finished: $(date)"
echo ""
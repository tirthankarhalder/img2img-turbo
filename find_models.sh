#!/bin/bash

# Script to find all your trained models and test data
# Run this to see what's available for testing

cd /home/du2/22CS30064/img2img-turbo

echo "============================================================"
echo "Finding Trained Models and Test Data"
echo "============================================================"
echo ""

# Find all .pkl checkpoint files
echo "Searching for model checkpoints (.pkl files)..."
echo ""

PKL_FILES=$(find . -name "*.pkl" -type f 2>/dev/null | grep -v ".ipynb_checkpoints")

if [ -z "$PKL_FILES" ]; then
    echo "❌ No .pkl checkpoint files found!"
    echo ""
    echo "Have you trained a model yet?"
    echo "Training typically saves checkpoints in:"
    echo "  - output/pix2pix_turbo/*/checkpoints/"
    echo "  - output/pix2pix_turbo_l40/checkpoints/"
    exit 1
fi

echo "✓ Found checkpoint files:"
echo ""

# Group by directory
LAST_DIR=""
COUNT=0

while IFS= read -r file; do
    DIR=$(dirname "$file")
    
    if [ "$DIR" != "$LAST_DIR" ]; then
        if [ $COUNT -gt 0 ]; then
            echo ""
        fi
        echo "Directory: $DIR"
        LAST_DIR="$DIR"
        COUNT=0
    fi
    
    BASENAME=$(basename "$file")
    SIZE=$(du -h "$file" | cut -f1)
    echo "  - $BASENAME ($SIZE)"
    COUNT=$((COUNT + 1))
done <<< "$PKL_FILES"

echo ""
echo "============================================================"
echo "Total checkpoints found: $(echo "$PKL_FILES" | wc -l)"
echo "============================================================"
echo ""

# Find test data
echo "Searching for test data directories..."
echo ""

TEST_DIRS=$(find data -type d -name "test_A" 2>/dev/null)

if [ -z "$TEST_DIRS" ]; then
    echo "❌ No test_A directories found in data/"
    echo ""
    echo "Expected structure:"
    echo "  data/YOUR_DATASET/test_A/"
else
    echo "✓ Found test directories:"
    echo ""
    
    while IFS= read -r dir; do
        DATASET=$(basename $(dirname "$dir"))
        IMG_COUNT=$(ls "$dir"/*.png 2>/dev/null | wc -l)
        
        if [ $IMG_COUNT -gt 0 ]; then
            echo "Dataset: $DATASET"
            echo "  Location: $dir"
            echo "  Images: $IMG_COUNT .png files"
            
            # Show first few filenames
            echo "  Sample files:"
            ls "$dir"/*.png 2>/dev/null | head -3 | while read -r img; do
                echo "    - $(basename $img)"
            done
            echo ""
        fi
    done <<< "$TEST_DIRS"
fi

echo "============================================================"
echo "Quick Test Commands"
echo "============================================================"
echo ""

# Generate example commands based on what was found
FIRST_MODEL=$(echo "$PKL_FILES" | head -1)
FIRST_TEST_DIR=$(echo "$TEST_DIRS" | head -1)

if [ -n "$FIRST_MODEL" ] && [ -n "$FIRST_TEST_DIR" ]; then
    FIRST_TEST_IMG=$(ls "$FIRST_TEST_DIR"/*.png 2>/dev/null | head -1)
    
    if [ -n "$FIRST_TEST_IMG" ]; then
        echo "Example 1: Quick test with your first checkpoint and image"
        echo ""
        echo "python simple_test.py \\"
        echo "    $FIRST_MODEL \\"
        echo "    $FIRST_TEST_IMG \\"
        echo "    \"\" \\"
        echo "    outputs/quick_test.png"
        echo ""
        echo "---"
        echo ""
        echo "Example 2: Full test on directory"
        echo ""
        echo "python test_pix2pix_turbo.py \\"
        echo "    --model_path $FIRST_MODEL \\"
        echo "    --input_dir $FIRST_TEST_DIR \\"
        echo "    --output_dir outputs/test_results \\"
        echo "    --save_comparison"
        echo ""
        echo "---"
        echo ""
        echo "Example 3: SLURM job"
        echo ""
        echo "Edit slurm_test_pix2pix.sh with:"
        echo "  export CHECKPOINT_NAME=\"$(basename $FIRST_MODEL)\""
        echo "  export MODEL_PATH=\"$FIRST_MODEL\""
        echo "  export TEST_A_DIR=\"$FIRST_TEST_DIR\""
        echo ""
        echo "Then run: sbatch slurm_test_pix2pix.sh"
    fi
fi

echo ""
echo "============================================================"
echo "Next Steps"
echo "============================================================"
echo ""
echo "1. Run setup: ./complete_setup.sh"
echo "2. Try a quick test with the example command above"
echo "3. For SLURM testing, edit and submit: sbatch slurm_test_pix2pix.sh"
echo ""

"""
Simple test script for pix2pix-turbo
Matches img2img-turbo repository structure exactly
"""

import torch
from PIL import Image
import sys
import os

# Add src to path - matching repo structure
sys.path.append('src')
from pix2pix_turbo import Pix2Pix_Turbo


def test_model(model_path, input_image_path, prompt='', output_path='output.png'):
    """
    Quick test function for pix2pix-turbo model
    
    Args:
        model_path: Path to trained checkpoint (.pkl file)
        input_image_path: Path to input image
        prompt: Text prompt (can be empty string)
        output_path: Where to save output
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model - use pretrained_path parameter as in repo
    print(f"Loading model from {model_path}...")
    model = Pix2Pix_Turbo(pretrained_path=model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load and preprocess input image
    print(f"Loading input image from {input_image_path}...")
    input_image = Image.open(input_image_path).convert('RGB')
    input_image = input_image.resize((512, 512), Image.LANCZOS)
    
    # Run inference - model handles PIL images directly
    print(f"Running inference with prompt: '{prompt}'")
    with torch.no_grad():
        output_image = model(input_image, prompt)
    
    # Save output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    output_image.save(output_path)
    print(f"Output saved to {output_path}")
    
    # Create side-by-side comparison
    comparison = Image.new('RGB', (1024, 512))
    comparison.paste(input_image, (0, 0))
    comparison.paste(output_image, (512, 0))
    comparison_path = output_path.replace('.png', '_comparison.png')
    comparison.save(comparison_path)
    print(f"Comparison saved to {comparison_path}")
    
    return output_image


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python simple_test.py MODEL_PATH INPUT_IMAGE [PROMPT] [OUTPUT_PATH]")
        print("\nExamples:")
        print("  # Fill50k dataset")
        print("  python simple_test.py \\")
        print("      output/pix2pix_turbo/fill50k/checkpoints/model_6001.pkl \\")
        print("      data/my_fill50k/test_A/40000.png \\")
        print("      'violet circle with orange background' \\")
        print("      outputs/test_output.png")
        print()
        print("  # Edges2handbag dataset")
        print("  python simple_test.py \\")
        print("      output/pix2pix_turbo/edges2handbag/checkpoints/model_best.pkl \\")
        print("      data/edges2handbag/test_A/image_001.png \\")
        print("      '' \\")
        print("      outputs/handbag_output.png")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_image_path = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else ''
    output_path = sys.argv[4] if len(sys.argv) > 4 else 'output.png'
    
    test_model(model_path, input_image_path, prompt, output_path)

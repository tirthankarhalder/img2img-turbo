#!/usr/bin/env python3
"""
test_pix2pix_turbo.py
Batch evaluator for Pix2Pix-Turbo.
Reuses `run_inference_single` from src/inference_paired.py for all test images.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import torch
from torchvision import transforms

from inference_paired import run_inference_single

# Optional LPIPS metric
try:
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    lpips_fn = None

def compute_metrics(pred, gt, device='cpu'):
    pred_np = np.array(pred).astype(np.float32) / 255.0
    gt_np = np.array(gt).astype(np.float32) / 255.0
    psnr_val = psnr(pred_np, gt_np, data_range=1.0)
    ssim_val = ssim(pred_np, gt_np, channel_axis=-1, data_range=1.0)
    lpips_val = None
    if lpips_fn:
        t = transforms.ToTensor()
        img1_t = t(pred).unsqueeze(0).to(device)
        img2_t = t(gt).unsqueeze(0).to(device)
        lpips_val = lpips_fn(img1_t, img2_t).item()
    return psnr_val, ssim_val, lpips_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint (.pkl)')
    parser.add_argument('--input_dir', required=True, help='Directory with input (test_A) images')
    parser.add_argument('--output_dir', required=True, help='Directory to save outputs')
    parser.add_argument('--gt_dir', default=None, help='Directory with ground truth (test_B) images')
    parser.add_argument('--prompts_json', default=None, help='Path to JSON file mapping image -> prompt')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--save_comparison', action='store_true', help='Save side-by-side comparisons')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts (if any)
    if args.prompts_json and os.path.exists(args.prompts_json):
        with open(args.prompts_json, 'r') as f:
            prompts = json.load(f)
    else:
        print("⚠️ No prompt file provided — using empty prompts.")
        prompts = {}

    # Loop through test images
    records = []
    input_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in tqdm(input_files, desc="Running inference"):
        input_path = os.path.join(args.input_dir, fname)
        prompt = prompts.get(fname, "")

        # Run inference
        output_img = run_inference_single(
            input_image_path=input_path,
            prompt=prompt,
            model_path=args.model_path,
            model_name='',
            device=args.device,
            use_fp16=args.use_fp16
        )

        # Save generated image
        output_path = os.path.join(args.output_dir, fname)
        output_img.save(output_path)

        # Compute metrics if GT available
        psnr_val = ssim_val = lpips_val = None
        if args.gt_dir:
            gt_path = os.path.join(args.gt_dir, fname)
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('RGB')
                psnr_val, ssim_val, lpips_val = compute_metrics(output_img, gt_img, args.device)

                # Optional side-by-side comparison
                if args.save_comparison:
                    comp = Image.new('RGB', (output_img.width * 3, output_img.height))
                    comp.paste(Image.open(input_path).convert('RGB'), (0, 0))
                    comp.paste(output_img, (output_img.width, 0))
                    comp.paste(gt_img, (2 * output_img.width, 0))
                    comp.save(os.path.join(args.output_dir, fname.replace('.png', '_comparison.png')))

        records.append({
            "image": fname,
            "prompt": prompt,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "output_path": output_path
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "metrics_report.csv")
    df.to_csv(csv_path, index=False)

    # HTML report
    html_path = os.path.join(args.output_dir, "report.html")
    with open(html_path, 'w') as f:
        f.write("<h1>Pix2Pix-Turbo Test Report</h1>")
        f.write(f"<p>Model: {args.model_path}</p>")
        f.write(f"<p>Total Samples: {len(df)}</p>")
        if 'psnr' in df.columns:
            f.write(f"<p>Average PSNR: {df['psnr'].mean():.3f}</p>")
            f.write(f"<p>Average SSIM: {df['ssim'].mean():.3f}</p>")
        f.write(df.head(50).to_html(escape=False))

    print(f"✅ Metrics saved to {csv_path}")
    print(f"✅ HTML report saved to {html_path}")


if __name__ == "__main__":
    main()

import os
import json
from tqdm import tqdm
from PIL import Image
import torch

# Import the inference logic directly
from src.inference_paired import Pix2Pix_Turbo, canny_from_pil
import torchvision.transforms.functional as F
from torchvision import transforms


def run_inference(model, input_image_path, prompt, output_dir, use_fp16=False):
    """Wrapper around src/inference_paired.py logic for a single image."""
    os.makedirs(output_dir, exist_ok=True)

    input_image = Image.open(input_image_path).convert('RGB')
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    bname = os.path.basename(input_image_path)

    with torch.no_grad():
        c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
        if use_fp16:
            c_t = c_t.half()
        output_image = model(c_t, prompt)
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    out_path = os.path.join(output_dir, bname)
    output_pil.save(out_path)
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompts_json", type=str, default="")
    parser.add_argument("--default_prompt", type=str, default="")
    parser.add_argument("--use_fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = Pix2Pix_Turbo(pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    # Load prompts if available
    prompts = {}
    if args.prompts_json and os.path.exists(args.prompts_json):
        with open(args.prompts_json, "r") as f:
            prompts = json.load(f)

    # Collect input images
    image_paths = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Found {len(image_paths)} images in {args.input_dir}")
    results = []

    for img_path in tqdm(image_paths, desc="Generating outputs"):
        fname = os.path.basename(img_path)
        prompt = prompts.get(fname, args.default_prompt)
        out_path = run_inference(model, img_path, prompt, args.output_dir, args.use_fp16)
        results.append({"input": img_path, "output": out_path, "prompt": prompt})

    # Save results summary
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Inference complete. Results saved in {args.output_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

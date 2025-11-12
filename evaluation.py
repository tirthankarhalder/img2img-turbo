import argparse
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# --- We must import these for the pipeline class ---
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from src.unet_2d_condition import UNet2DConditionModel  # We still need this from src
from src.utils import load_lora_weights_orig           # And this

# --- NEW SOLUTION: Add the project directory to Python's path ---
# This is still needed to find the 'src' folder for the two imports above
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# -----------------------------------------------------------------
# --- PIPELINE CLASS COPIED FROM 'src/inference_paired.py' ---
# -----------------------------------------------------------------
class PairedControlNetPipeline(torch.nn.Module):
    def __init__(self, unet, vae, tokenizer, text_encoder, controlnet, scheduler, safety_checker, feature_extractor):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.controlnet = controlnet
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, Image.Image] = None,
        control_image: Union[torch.FloatTensor, Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
    ):
        device = self.unet.device
        do_classifier_free_guidance = False  # Hard-coded to False as per sd-turbo

        if isinstance(self.controlnet, MultiControlNetModel):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # 1. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self.text_encoder(
            self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        )[0]

        # 2. Prepare control image
        if isinstance(control_image, Image.Image):
            control_image = np.array(control_image.convert("RGB"))
            control_image = torch.from_numpy(control_image).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0

        control_image = control_image.to(device=device, dtype=prompt_embeds.dtype)
        
        # 3. Prepare image latents
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
            image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        
        image = image.to(device=device, dtype=prompt_embeds.dtype)
        
        image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
        image_latents = image_latents * self.vae.config.scaling_factor

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        shape = image_latents.shape
        latents = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)

        # 6. Denoising loop
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    image_latents, t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    latents, t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample

                latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

        # 8. Post-processing
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        if output_type == "pil":
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            image = [Image.fromarray(im) for im in image]
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    def to(self, device):
        self.unet.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        self.controlnet.to(device)
        return self
# -----------------------------------------------------------------
# --- END OF PIPELINE CLASS ---
# -----------------------------------------------------------------

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    # ... (Rest of get_args is UNCHANGED) ...
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to your trained model checkpoint file (e.g., model_6001.pkl)",
    )
    parser.add_argument(
        "--test_A_dir",
        type=str,
        required=True,
        help="Directory containing the input test images (e.g., data/my_dataset/test_A)",
    )
    parser.add_argument(
        "--test_B_dir",
        type=str,
        required=True,
        help="Directory containing the ground truth images (e.g., data/my_dataset/test_B)",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=None,
        help="Path to the test_prompts.json file (e.g., data/dataset_mmWave/test_prompts.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_outputs",
        help="Directory to save the generated images (optional)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a high-quality, detailed, realistic photo",
        help="The DEFAULT prompt to use if a file is not in the prompts_json",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="The resolution to use for processing images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for reproducible results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu)",
    )
    return parser.parse_args()


def load_image(image_path, size=512):
    """Loads and transforms an image to a PIL.Image object."""
    try:
        image = Image.open(image_path).convert("RGB").resize((size, size))
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def pil_to_tensor(img, device):
    """Converts a PIL Image to a [0, 1] tensor on the correct device."""
    img_np = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    tensor = tensor / 255.0
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    
    print(f"--- Initializing Model on {device} ---")

    # --- 1. Load the Base Model Components ---
    base_model = "stabilityai/sd-turbo"
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=torch.float16).to(device)
    controlnet = ControlNetModel.from_unet(unet, torch_dtype=torch.float16).to(device)
    
    # --- 2. Load Your Trained Weights ---
    print(f"Loading trained weights from: {args.model_path}")
    # This load_lora_weights_orig function is imported from src/utils.py
    # The 'sys.path' fix at the top makes this import work
    unet, vae = load_lora_weights_orig(unet, vae, args.model_path, "lora", 0.8)

    # --- 3. Create the Pipeline ---
    # We now use the class we defined inside this file
    pipeline = PairedControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler.from_pretrained(base_model, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- 4. Prepare Directories, Metrics, and Prompts ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    ssim_scores, psnr_scores = [], []

    # Load prompts
    prompts = {}
    default_prompt = args.prompt
    if args.prompts_json and os.path.exists(args.prompts_json):
        print(f"Loading prompts from {args.prompts_json}")
        try:
            with open(args.prompts_json, 'r') as f:
                prompts = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load prompts_json: {e}. Using default prompt.")
    else:
        print(f"Warning: Prompts file not found at {args.prompts_json}. Using default prompt for all images.")

    # --- 5. Find all test images ---
    input_paths = sorted(glob(os.path.join(args.test_A_dir, "*.png")) + 
                         glob(os.path.join(args.test_A_dir, "*.jpg")))
    
    if not input_paths:
        print(f"\nError: No images found in {args.test_A_dir}")
        return
        
    print(f"Found {len(input_paths)} images in {args.test_A_dir}")
    print("--- Starting Evaluation ---")

    # --- 6. Loop, Generate, and Evaluate ---
    for input_path in tqdm(input_paths, desc="Evaluating Model"):
        filename = os.path.basename(input_path)
        gt_path = os.path.join(args.test_B_dir, filename)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Skipping {filename}. No matching ground truth file.")
            continue
            
        # Load images
        input_image = load_image(input_path, size=args.image_size)
        gt_image = load_image(gt_path, size=args.image_size) # Corrected a typo here
        
        if input_image is None or gt_image is None:
            continue

        # Get the specific prompt for this file
        current_prompt = prompts.get(filename, default_prompt)
        
        # --- 7. Generate the output image ---
        generated_image = pipeline(
            prompt=current_prompt,
            image=input_image,
            control_image=input_image,
            num_inference_steps=1,
            guidance_scale=0.0,
            controlnet_conditioning_scale=0.9,
            generator=generator
        ).images[0]
        
        output_path = os.path.join(args.output_dir, filename)
        generated_image.save(output_path)

        # --- 8. Calculate Metrics ---
        gt_tensor = pil_to_tensor(gt_image, device)
        gen_tensor = pil_to_tensor(generated_image, device)
        ssim_scores.append(ssim(gen_tensor, gt_tensor).item())
        psnr_scores.append(psnr(gen_tensor, gt_tensor).item())

    # --- 9. Report Final Averages ---
    if not ssim_scores:
        print("\nError: No images were successfully processed.")
        return

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    
    print("\n--- ✅ Evaluation Complete ---")
    print(f"Total Images Processed: {len(ssim_scores)}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"\nGenerated images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

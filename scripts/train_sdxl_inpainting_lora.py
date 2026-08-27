import os
import argparse
import math
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw

from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, DDPMScheduler
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SDXLInpaintingLoRATrainer")

class InpaintingTextToImageDataset(Dataset):
    """
    Dataset loader for Stage 2: Garment Regional Inpainting & Application Pairs.
    """
    def __init__(self, hf_repo_or_local_path: str, split: str = "train", resolution: int = 768):
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

        logger.info(f"Loading Inpainting dataset split '{split}' from Hugging Face Hub: {hf_repo_or_local_path}")
        self.dataset = load_dataset(hf_repo_or_local_path, name="garment_application", split=split)

    def generate_synthetic_region_mask(self, target_region: str, width: int = 768, height: int = 768) -> Image.Image:
        """
        Generates binary segmentation mask corresponding to targeted garment regions (collar, sleeves, cuffs, chest, hemline).
        """
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        r = target_region.lower()
        if "collar" in r or "neckline" in r:
            draw.polygon([(width*0.3, height*0.1), (width*0.7, height*0.1), (width*0.6, height*0.35), (width*0.4, height*0.35)], fill=255)
        elif "sleeve" in r or "cuff" in r:
            draw.rectangle([0, int(height*0.2), int(width*0.25), int(height*0.8)], fill=255)
            draw.rectangle([int(width*0.75), int(height*0.2), width, int(height*0.8)], fill=255)
        elif "chest" in r:
            draw.rectangle([int(width*0.25), int(height*0.2), int(width*0.75), int(height*0.55)], fill=255)
        elif "hemline" in r:
            draw.rectangle([int(width*0.15), int(height*0.75), int(width*0.85), height], fill=255)
        else: # full garment default mask
            draw.rectangle([int(width*0.2), int(height*0.15), int(width*0.8), int(height*0.9)], fill=255)
        return mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        target_region = item.get("target_region", "collar")
        instruction = item.get("instruction", item.get("text", "Apply craft"))

        mask_img = self.generate_synthetic_region_mask(target_region, image.width, image.height)

        tensor_image = self.transform(image)
        tensor_mask = self.mask_transform(mask_img)

        # Masked image representation
        tensor_masked_image = tensor_image * (1 - tensor_mask)

        return {
            "pixel_values": tensor_image,
            "mask_values": tensor_mask,
            "masked_pixel_values": tensor_masked_image,
            "caption": instruction
        }

def encode_prompts(tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, prompts, device):
    with torch.no_grad():
        t1 = tokenizer_1(prompts, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
        p1 = text_encoder_1(t1, output_hidden_states=True).hidden_states[-2]

        t2 = tokenizer_2(prompts, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
        e2 = text_encoder_2(t2, output_hidden_states=True)
        p2 = e2.hidden_states[-2]
        pooled = e2.text_embeds

        prompt_embeds = torch.cat([p1, p2], dim=-1)
    return prompt_embeds, pooled

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 SDXL Inpainting LoRA for Regional Garment Application")
    parser.add_argument("--pretrained-model", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--dataset-name", type=str, required=True, help="Hugging Face Dataset repo ID")
    parser.add_argument("--output-dir", type=str, default="./outputs/induscraft_inpainting_lora")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)

    tokenizer_1 = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer_2", use_fast=False)
    text_encoder_1 = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder", torch_dtype=dtype).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    text_encoder_1.eval()
    text_encoder_2.eval()

    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.to(device)
    unet.print_trainable_parameters()

    train_dataset = InpaintingTextToImageDataset(args.dataset_name, split="train", resolution=args.resolution)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    global_step = 0
    logger.info(f"Starting Stage 2 Inpainting LoRA training for {args.num_train_epochs} epochs...")

    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        for batch in progress_bar:
            images = batch["pixel_values"].to(device, dtype=dtype)
            masks = batch["mask_values"].to(device, dtype=dtype)
            masked_images = batch["masked_pixel_values"].to(device, dtype=dtype)
            prompts = batch["caption"]

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(masked_images).latent_dist.sample() * vae.config.scaling_factor
                resized_masks = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            latent_model_input = torch.cat([noisy_latents, resized_masks, masked_latents], dim=1)

            prompt_embeds, pooled_prompt_embeds = encode_prompts(
                tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, prompts, device
            )

            add_time_ids = torch.tensor(
                [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
                dtype=dtype, device=device
            ).repeat(bsz, 1)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds.to(dtype), "time_ids": add_time_ids}

            model_pred = unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds.to(dtype),
                added_cond_kwargs=added_cond_kwargs
            ).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})

    final_path = Path(args.output_dir) / "induscraft_inpainting_lora_final"
    unet.save_pretrained(final_path)
    logger.info(f"Stage 2 Inpainting LoRA Training complete! Model saved to {final_path}")

if __name__ == "__main__":
    main()

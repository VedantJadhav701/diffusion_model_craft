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
from PIL import Image

from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionXLPipeline
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SDXLLoRATrainer")

class HFOrLocalTextToImageDataset(Dataset):
    """
    Dataset loader supporting either a local folder with metadata or Hugging Face dataset.
    """
    def __init__(self, hf_repo_or_local_path: str, split: str = "train", resolution: int = 768):
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        if os.path.exists(hf_repo_or_local_path):
            # Local directory with jsonl metadata
            import json
            metadata_file = Path(hf_repo_or_local_path) / f"{split}.jsonl"
            self.items = []
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        img_path = Path(hf_repo_or_local_path).parent / "images" / data["image"]
                        if img_path.exists():
                            self.items.append({"image": str(img_path), "text": data.get("text", "")})
            self.use_hf = False
        else:
            # Hugging Face Dataset repo
            logger.info(f"Loading dataset split '{split}' from Hugging Face Hub: {hf_repo_or_local_path}")
            self.dataset = load_dataset(hf_repo_or_local_path, split=split)
            self.use_hf = True

    def __len__(self):
        return len(self.dataset) if self.use_hf else len(self.items)

    def __getitem__(self, idx):
        if self.use_hf:
            item = self.dataset[idx]
            image = item["image"]
            text = item.get("text", "")
        else:
            item = self.items[idx]
            image = Image.open(item["image"])
            text = item["text"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor_image = self.transform(image)
        return {"pixel_values": tensor_image, "caption": text}

def encode_prompts(tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, prompts, device):
    """
    Encodes text prompts for SDXL dual text encoders.
    Returns prompt_embeds and pooled_prompt_embeds.
    """
    with torch.no_grad():
        # Text Encoder 1 (CLIP ViT-L/14)
        tokens_1 = tokenizer_1(prompts, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
        encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
        prompt_embeds_1 = encoder_output_1.hidden_states[-2]

        # Text Encoder 2 (OpenCLIP ViT-bigG/14)
        tokens_2 = tokenizer_2(prompts, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
        encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
        prompt_embeds_2 = encoder_output_2.hidden_states[-2]
        pooled_prompt_embeds = encoder_output_2.text_embeds

        # Concatenate prompt embeddings along hidden dimension (768 + 1280 = 2048)
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds

def main():
    parser = argparse.ArgumentParser(description="Train SDXL UNet LoRA for IndusCraft")
    parser.add_argument("--pretrained-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dataset-name", type=str, required=True, help="HF dataset repo or local metadata folder path")
    parser.add_argument("--output-dir", type=str, default="./outputs/induscraft_lora")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16

    # 1. Load Tokenizers & Text Encoders
    logger.info("Loading tokenizers and text encoders...")
    tokenizer_1 = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer_2", use_fast=False)
    text_encoder_1 = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder", torch_dtype=dtype).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    text_encoder_1.eval()
    text_encoder_2.eval()

    # 2. Load VAE & UNet
    logger.info("Loading VAE and UNet...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # 3. Configure LoRA on UNet
    logger.info(f"Injecting LoRA adapters into UNet (rank={args.lora_rank}, alpha={args.lora_alpha})...")
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

    # 4. Prepare Dataset & DataLoader
    train_dataset = HFOrLocalTextToImageDataset(args.dataset_name, split="train", resolution=args.resolution)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # 5. Training Loop
    global_step = 0
    total_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    logger.info(f"Starting training for {args.num_train_epochs} epochs (~{total_steps} optimizer steps)...")

    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for batch in progress_bar:
            images = batch["pixel_values"].to(device, dtype=dtype)
            prompts = batch["caption"]

            # Encode images to VAE latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

            # Sample noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            prompt_embeds, pooled_prompt_embeds = encode_prompts(
                tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, prompts, device
            )

            # Additional SDXL Micro-conditioning time IDs (original size, crop coords, target size)
            add_time_ids = torch.tensor(
                [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
                dtype=dtype, device=device
            ).repeat(bsz, 1)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds.to(dtype),
                "time_ids": add_time_ids
            }

            # UNet forward pass
            model_pred = unet(
                noisy_latents,
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

            if global_step % args.save_steps == 0:
                save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                unet.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    # Final Save
    final_path = Path(args.output_dir) / "induscraft_sdxl_lora_final"
    unet.save_pretrained(final_path)
    logger.info(f"Training complete! Model saved to {final_path}")

if __name__ == "__main__":
    main()

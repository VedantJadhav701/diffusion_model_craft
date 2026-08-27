import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

# Disable unstable Xet LFS backend to enforce reliable standard LFS uploading
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi, login
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value

from data_pipeline.config import (
    IMAGES_DIR, LAYER1_TRAIN_METADATA_PATH, LAYER1_VAL_METADATA_PATH,
    LAYER2_TRAIN_METADATA_PATH, LAYER2_VAL_METADATA_PATH,
    LAYER3_TRAIN_METADATA_PATH, LAYER3_VAL_METADATA_PATH, CRAFT_METADATA
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HFUploader")

def generate_multi_layer_dataset_card(repo_id: str, summary_counts: dict) -> str:
    """Generates Markdown Dataset Card for Hugging Face Hub."""
    craft_list_str = "\n".join([f"- **{info['full_name']}** ({key}): {info['region']}" for key, info in CRAFT_METADATA.items()])
    
    card_content = f"""---
license: cc-by-4.0
task_categories:
- text-to-image
- image-to-image
- inpainting
tags:
- indian-art
- fashion-design
- traditional-crafts
- textile
- sdxl-lora
- regional-inpainting
pretty_name: IndusCraft - Multi-Layer Indian Traditional Crafts & Fashion Design Dataset
size_categories:
- 10K<n<100K
---

# 🎨 IndusCraft: Multi-Layer Indian Traditional Crafts & Fashion Design Dataset

IndusCraft is a multi-layer multimodal dataset designed for fashion and textile education (e.g. Pearl Academy). It supports two primary AI workflows:
1. **Workflow A — Create New Design**: Generates multi-view fashion designs (Front view, Detail macro view, Seamless Pattern view, Technical Flat view).
2. **Workflow B — Apply Craft to Existing Garment**: Regional inpainting and editing that applies traditional Indian craft onto specific garment regions (Collar, Sleeves, Cuffs, Hemline, Chest) while preserving the student's original garment.

## 📊 Dataset Layers & Splitting Summary

- **Layer 1: Craft Reference Knowledge**: {summary_counts.get('l1_train', 0)} Train / {summary_counts.get('l1_val', 0)} Val
- **Layer 2: Garment Application Pairs & Instructions**: {summary_counts.get('l2_train', 0)} Train / {summary_counts.get('l2_val', 0)} Val
- **Layer 3: Design Details & Macro Texture Views**: {summary_counts.get('l3_train', 0)} Train / {summary_counts.get('l3_val', 0)} Val

## 🧵 Included Indian Craft Categories

{craft_list_str}

## 🛠️ Created by
Prepared with IndusCraft Data Pipeline for Pearl Academy & Fashion Education AI Training.
"""
    return card_content

def push_with_retry(dataset_dict: DatasetDict, repo_id: str, config_name: str, token: Optional[str] = None, private: bool = False, max_retries: int = 3):
    """Pushes dataset to hub with retry logic for network resilience."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Pushing '{config_name}' layer to HF Hub (Attempt {attempt+1}/{max_retries})...")
            dataset_dict.push_to_hub(
                repo_id,
                config_name=config_name,
                token=token,
                private=private,
                max_shard_size="100MB"
            )
            logger.info(f"Successfully pushed '{config_name}' layer!")
            return
        except Exception as e:
            logger.warning(f"Push for '{config_name}' failed on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(3.0 * (attempt + 1))
            else:
                raise e

def prepare_and_push_to_hf(
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False
):
    """
    Pushes all 3 multi-layer dataset configurations to Hugging Face Hub.
    """
    if token:
        login(token=token)

    api = HfApi()
    logger.info(f"Ensuring Hugging Face dataset repository exists: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    except Exception as e:
        logger.warning(f"Could not automatically create HF repo '{repo_id}': {e}. Proceeding with push_to_hub...")

    def load_records(jsonl_path: Path):
        records = []
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        image_full_path = IMAGES_DIR / item["image"]
                        if image_full_path.exists():
                            records.append({
                                "image": str(image_full_path),
                                "text": item.get("text", item.get("initial_caption", "")),
                                "craft": item.get("craft", "unknown"),
                                "view_type": item.get("view_type", "front_view"),
                                "target_region": item.get("target_region", "full_garment"),
                                "instruction": item.get("instruction", item.get("text", "")),
                            })
        return records

    l1_train = load_records(LAYER1_TRAIN_METADATA_PATH)
    l1_val = load_records(LAYER1_VAL_METADATA_PATH)
    l2_train = load_records(LAYER2_TRAIN_METADATA_PATH)
    l2_val = load_records(LAYER2_VAL_METADATA_PATH)
    l3_train = load_records(LAYER3_TRAIN_METADATA_PATH)
    l3_val = load_records(LAYER3_VAL_METADATA_PATH)

    features = Features({
        "image": HFImage(),
        "text": Value("string"),
        "craft": Value("string"),
        "view_type": Value("string"),
        "target_region": Value("string"),
        "instruction": Value("string")
    })

    def create_split_dict(train_recs, val_recs):
        train_ds = Dataset.from_dict({
            "image": [r["image"] for r in train_recs],
            "text": [r["text"] for r in train_recs],
            "craft": [r["craft"] for r in train_recs],
            "view_type": [r["view_type"] for r in train_recs],
            "target_region": [r["target_region"] for r in train_recs],
            "instruction": [r["instruction"] for r in train_recs],
        }, features=features)

        val_ds = Dataset.from_dict({
            "image": [r["image"] for r in val_recs],
            "text": [r["text"] for r in val_recs],
            "craft": [r["craft"] for r in val_recs],
            "view_type": [r["view_type"] for r in val_recs],
            "target_region": [r["target_region"] for r in val_recs],
            "instruction": [r["instruction"] for r in val_recs],
        }, features=features)

        return DatasetDict({"train": train_ds, "validation": val_ds})

    # Push all 3 dataset layers with robust retries
    push_with_retry(create_split_dict(l1_train, l1_val), repo_id, "craft_reference", token=token, private=private)
    push_with_retry(create_split_dict(l2_train, l2_val), repo_id, "garment_application", token=token, private=private)
    push_with_retry(create_split_dict(l3_train, l3_val), repo_id, "design_details", token=token, private=private)

    # Upload Dataset Card
    counts = {
        "l1_train": len(l1_train), "l1_val": len(l1_val),
        "l2_train": len(l2_train), "l2_val": len(l2_val),
        "l3_train": len(l3_train), "l3_val": len(l3_val)
    }
    card_text = generate_multi_layer_dataset_card(repo_id, counts)
    try:
        api.upload_file(
            path_or_fileobj=card_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
    except Exception as e:
        logger.warning(f"Dataset card upload failed: {e}")

    logger.info(f"Successfully uploaded 3 dataset layers to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload 3-Layer IndusCraft dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo name e.g. username/induscraft-dataset")
    parser.add_argument("--token", default=None, help="Hugging Face API token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    prepare_and_push_to_hf(args.repo_id, args.token, args.private)

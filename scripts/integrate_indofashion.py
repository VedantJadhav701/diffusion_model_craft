import os
import json
import re
import random
import shutil
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IndoFashionIntegrator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDO_DIR = PROJECT_ROOT / "indofashion_dataset"
RAW_BASE_DIR = PROJECT_ROOT / "data" / "raw" / "web" / "base_garment"
RAW_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "raw.jsonl"

TARGET_CLASSES = {
    "saree": "saree",
    "lehenga": "lehenga",
    "women_kurta": "women's kurta",
    "duppatta": "duppatta",
    "sherwanis": "sherwani",
    "kurta_men": "men's kurta"
}

def clean_product_title(title: str, class_label: str) -> str:
    # Lowercase
    t = title.lower()
    
    # Remove text in parentheses/brackets
    t = re.sub(r'\(.*?\)', '', t)
    t = re.sub(r'\[.*?\]', '', t)
    
    # Common noise terms
    noise = [
        "with blouse piece", "with blouse", "unstitched blouse piece", "unstiched blouse piece",
        "without blouse piece", "free size", "onesize", "one size", "below 500 rupees",
        "under 300", "latest design", "new collection", "party wear", "casual", "beautiful",
        "for women", "women's", "men's", "women", "men", "designer", "sadi offer", "saree for", "sari for",
        "kurta for", "lehenga for", "sherwani for"
    ]
    for np in noise:
        t = t.replace(np, "")
        
    # Replace dividers/punctuation with space
    t = re.sub(r'[.,;:\-_&|\\/]', ' ', t)
    t = " ".join(t.split())
    
    # Fallback if title is empty
    if not t.strip():
        t = f"traditional Indian ethnic wear {TARGET_CLASSES[class_label]}"
        
    return t.strip()

def main():
    if not INDO_DIR.exists():
        logger.error(f"IndoFashion directory not found at {INDO_DIR}")
        return

    train_json = INDO_DIR / "train_data.json"
    if not train_json.exists():
        logger.error(f"train_data.json not found at {train_json}")
        return

    RAW_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Read train_data.json
    logger.info("Reading IndoFashion train_data.json metadata...")
    class_groups = {c: [] for c in TARGET_CLASSES}
    
    with open(train_json, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    cls = item.get("class_label")
                    if cls in TARGET_CLASSES:
                        class_groups[cls].append(item)
                except Exception as e:
                    pass

    # Sample and process
    max_per_class = 500
    random.seed(42)
    
    logger.info(f"Target classes found: { {k: len(v) for k, v in class_groups.items()} }")
    logger.info(f"Sampling up to {max_per_class} images per category...")

    new_records = []
    
    for cls, items in class_groups.items():
        if not items:
            continue
        
        # Shuffle to get a random subset
        random.shuffle(items)
        sampled = items[:max_per_class]
        
        success_count = 0
        for idx, item in enumerate(sampled):
            rel_path = item.get("image_path")
            src_file = INDO_DIR / rel_path
            
            if not src_file.exists():
                continue
                
            # Create a clean file name and path
            filename = f"indofashion_{cls}_{idx:06d}.jpg"
            dest_file = RAW_BASE_DIR / filename
            
            try:
                # Validate image load & resolution
                with Image.open(src_file) as img:
                    width, height = img.size
                    img_format = img.format
                
                if width < 300 or height < 300:
                    continue
                    
                # Copy to data/raw/web/base_garment
                shutil.copy2(src_file, dest_file)
                
                # Create raw metadata record
                clean_title = clean_product_title(item.get("product_title", ""), cls)
                caption = f"[front view] of plain Indian ethnic wear {TARGET_CLASSES[cls]}: {clean_title}"
                
                record = {
                    "id": f"indofashion_{cls}_{idx:06d}",
                    "filename": filename,
                    "raw_path": str(dest_file),
                    "craft": "base_garment",
                    "source": "indofashion_dataset",
                    "source_url": item.get("image_url", ""),
                    "license": "research_only",
                    "width": width,
                    "height": height,
                    "format": img_format if img_format else "JPEG",
                    "initial_caption": caption
                }
                new_records.append(record)
                success_count += 1
            except Exception as e:
                logger.warning(f"Error processing {src_file}: {e}")
                
        logger.info(f"Successfully integrated {success_count} images for '{cls}'")

    # Append to raw.jsonl
    if new_records:
        RAW_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(RAW_METADATA_PATH, "a", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
        logger.info(f"=== INTEGRATION COMPLETE ===")
        logger.info(f"Added {len(new_records)} new base garment records to {RAW_METADATA_PATH}")
        logger.info("Please run: python scripts/run_data_pipeline.py to clean, caption, and rebuild split datasets.")

if __name__ == "__main__":
    main()

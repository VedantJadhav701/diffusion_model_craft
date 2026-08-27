import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import cv2
    import numpy as np
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False

from data_pipeline.config import (
    RAW_DIR, IMAGES_DIR, REJECTED_DIR, CLEANED_METADATA_PATH,
    MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, MAX_ASPECT_RATIO, DUPLICATE_PHASH_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Cleaning")

def detect_human_face(img: Image.Image) -> bool:
    """
    Detects if an image contains human face portraits.
    Filters out human face photos to keep only garments, fabrics, designs, and crafts.
    """
    if not HAS_OPENCV:
        return False
    try:
        np_img = np.array(img.convert("L"))
        faces = face_cascade.detectMultiScale(np_img, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        return len(faces) > 0
    except Exception:
        return False

def compute_simple_hash(img: Image.Image) -> str:
    """Fallback average hash if imagehash library is not installed."""
    img_resized = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(img_resized.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join(["1" if p > avg else "0" for p in pixels])
    return hex(int(bits, 2))[2:].zfill(16)

def compute_perceptual_hash(img: Image.Image):
    """Computes pHash using imagehash or fallback."""
    if HAS_IMAGEHASH:
        return imagehash.phash(img)
    else:
        return compute_simple_hash(img)

def verify_and_clean_image(raw_path: Path) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Checks for file corruption, resolution constraints, aspect ratio, and human face portraits.
    """
    if not raw_path.exists():
        return False, "file_not_found", None

    try:
        with Image.open(raw_path) as img:
            img.verify()
        
        img = Image.open(raw_path)
        img.load()
        
        width, height = img.size
        
        # Resolution filter
        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            return False, f"low_resolution_{width}x{height}", None

        # Aspect ratio filter
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > MAX_ASPECT_RATIO:
            return False, f"extreme_aspect_ratio_{aspect_ratio:.2f}", None

        # Filter out human face portraits to keep pure garments & craft designs
        if detect_human_face(img):
            return False, "contains_human_face", None

        # RGB Conversion check
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        return True, "valid", img

    except Exception as e:
        return False, f"corrupt_or_unreadable_{str(e)}", None

def run_cleaning_pipeline(raw_records: List[Dict]) -> List[Dict]:
    """
    Full cleaning, verification, face filtering, deduplication, and standardization pipeline.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_records = []
    rejected_records = []
    seen_hashes = []

    logger.info(f"Starting cleaning & face-filtering pipeline for {len(raw_records)} images...")

    for record in tqdm(raw_records, desc="Cleaning & Filtering"):
        raw_path = Path(record.get("raw_path", ""))
        is_valid, status_msg, img = verify_and_clean_image(raw_path)

        if not is_valid or img is None:
            record["rejection_reason"] = status_msg
            rejected_records.append(record)
            if raw_path.exists():
                dest_rejected = REJECTED_DIR / raw_path.name
                shutil.copy2(raw_path, dest_rejected)
            continue

        # Compute hash for duplicate detection
        img_hash = compute_perceptual_hash(img)
        is_duplicate = False

        if HAS_IMAGEHASH:
            for prev_hash in seen_hashes:
                if (img_hash - prev_hash) <= DUPLICATE_PHASH_THRESHOLD:
                    is_duplicate = True
                    break
        else:
            if img_hash in seen_hashes:
                is_duplicate = True

        if is_duplicate:
            record["rejection_reason"] = "duplicate_or_near_duplicate"
            rejected_records.append(record)
            dest_rejected = REJECTED_DIR / raw_path.name
            shutil.copy2(raw_path, dest_rejected)
            continue

        seen_hashes.append(img_hash)

        clean_filename = f"{record['id']}.jpg"
        clean_path = IMAGES_DIR / clean_filename

        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(clean_path, format="JPEG", quality=95)

        clean_record = record.copy()
        clean_record["image"] = clean_filename
        clean_record["clean_path"] = str(clean_path)
        clean_record["width"] = img.width
        clean_record["height"] = img.height
        clean_record["phash"] = str(img_hash)
        clean_record["status"] = "cleaned"

        cleaned_records.append(clean_record)

    logger.info(f"Cleaning complete: {len(cleaned_records)} passed (garments/crafts only), {len(rejected_records)} rejected.")

    with open(CLEANED_METADATA_PATH, "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return cleaned_records

if __name__ == "__main__":
    from data_pipeline.acquisition import RAW_METADATA_PATH
    if RAW_METADATA_PATH.exists():
        records = []
        with open(RAW_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        run_cleaning_pipeline(records)

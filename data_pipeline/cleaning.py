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

DOCUMENT_NOISE_KEYWORDS = [
    "pdf", "page1", "page2", "page3", "catalog", "table_and_index", "decimal_classification",
    "exhibition", "drawing_print", "state_magazine", "railway_station", "museum_building",
    "stamp", "map_", "_map.", "document", "newspaper", "book_cover", "article", "index_for_arranging"
]

def is_document_or_noisy_scene(caption: str, filename: str) -> bool:
    text = (caption + " " + filename).lower()
    for kw in DOCUMENT_NOISE_KEYWORDS:
        if kw in text:
            return True
    return False

def is_scanned_document_page(img: Image.Image) -> bool:
    if not HAS_OPENCV:
        return False
    try:
        np_img = np.array(img.convert("L"))
        white_pixel_ratio = float(np.mean(np_img > 242))
        return white_pixel_ratio > 0.75
    except Exception:
        return False

def compute_image_sharpness(img: Image.Image) -> float:
    """Computes Laplacian variance to measure image sharpness and reject blurry photos."""
    if not HAS_OPENCV:
        return 999.0
    try:
        np_img = np.array(img.convert("L"))
        return float(cv2.Laplacian(np_img, cv2.CV_64F).var())
    except Exception:
        return 999.0

def verify_and_clean_image(raw_path: Path, caption: str = "", strict_1024: bool = False) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Checks for file corruption, resolution constraints, aspect ratio, document scans, human face portraits, and optional strict 1024px sharpness.
    """
    if not raw_path.exists():
        return False, "file_not_found", None

    # Filter out obvious PDF scans and document metadata names
    if is_document_or_noisy_scene(caption, raw_path.name):
        return False, "document_or_noisy_scene", None

    try:
        with Image.open(raw_path) as img:
            img.verify()
        
        img = Image.open(raw_path)
        img.load()
        
        width, height = img.size
        
        min_w = 1024 if strict_1024 else MIN_IMAGE_WIDTH
        min_h = 1024 if strict_1024 else MIN_IMAGE_HEIGHT
        max_ar = 2.0 if strict_1024 else MAX_ASPECT_RATIO

        # Resolution filter
        if width < min_w or height < min_h:
            return False, f"resolution_below_{min_w}x{min_h}_{width}x{height}", None

        # Aspect ratio filter
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_ar:
            return False, f"extreme_aspect_ratio_{aspect_ratio:.2f}", None

        # Document scan filter (high white background text scans)
        if is_scanned_document_page(img):
            return False, "scanned_document_page", None

        # Filter out human face portraits to keep pure garments & craft designs
        if detect_human_face(img):
            return False, "contains_human_face", None

        # Strict 1024px Blur & Sharpness Filter
        if strict_1024:
            sharpness = compute_image_sharpness(img)
            if sharpness < 80.0:
                return False, f"blurry_image_sharpness_{sharpness:.1f}", None

        # RGB Conversion check
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        return True, "valid", img

    except Exception as e:
        return False, f"corrupt_or_unreadable_{str(e)}", None

def run_cleaning_pipeline(raw_records: List[Dict], strict_1024: bool = False) -> List[Dict]:
    """
    Full cleaning, verification, face filtering, deduplication, and standardization pipeline.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_records = []
    rejected_records = []
    seen_hashes = []

    mode_str = " (Strict 1024px+ Pristine Mode)" if strict_1024 else ""
    logger.info(f"Starting cleaning & face-filtering pipeline for {len(raw_records)} images{mode_str}...")

    for record in tqdm(raw_records, desc="Cleaning & Filtering"):
        raw_path = Path(record.get("raw_path", ""))
        caption = record.get("initial_caption", "")
        is_valid, status_msg, img = verify_and_clean_image(raw_path, caption, strict_1024=strict_1024)

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

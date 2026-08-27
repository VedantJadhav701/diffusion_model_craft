import os
import json
import shutil
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm

from data_pipeline.config import (
    RAW_DIR, RAW_HF_DIR, RAW_WEB_DIR, IMAGES_DIR, METADATA_DIR, REJECTED_DIR,
    MODELS_DIR, BASE_MODEL_DIR, LORA_MODEL_DIR, OUTPUTS_DIR, SAMPLES_DIR,
    CHECKPOINTS_DIR, LOGS_DIR, RAW_METADATA_PATH, CRAFT_METADATA
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Acquisition")

def ensure_directories():
    """Ensure all required workspace directories exist."""
    directories = [
        RAW_DIR, RAW_HF_DIR, RAW_WEB_DIR, IMAGES_DIR, METADATA_DIR, REJECTED_DIR,
        MODELS_DIR, BASE_MODEL_DIR, LORA_MODEL_DIR, OUTPUTS_DIR, SAMPLES_DIR,
        CHECKPOINTS_DIR, LOGS_DIR
    ]
    for craft in CRAFT_METADATA.keys():
        directories.append(RAW_HF_DIR / craft)
        directories.append(RAW_WEB_DIR / craft)

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("All project directories verified/created successfully.")

def ingest_local_directory(source_dir: str, craft_name: str, license_info: str = "custom_local") -> List[Dict]:
    """
    Ingests images from a local directory into raw data storage and builds metadata records.
    """
    ensure_directories()
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return []

    craft_name_clean = craft_name.lower().strip()
    target_raw_dir = RAW_WEB_DIR / craft_name_clean
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    records = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in source_path.rglob("*") if f.suffix.lower() in valid_extensions]
    logger.info(f"Found {len(image_files)} images in {source_dir} for craft '{craft_name_clean}'")

    for idx, img_file in enumerate(tqdm(image_files, desc=f"Ingesting {craft_name_clean}")):
        filename = f"{craft_name_clean}_raw_{idx:06d}{img_file.suffix.lower()}"
        dest_file = target_raw_dir / filename
        
        try:
            shutil.copy2(img_file, dest_file)
            with Image.open(dest_file) as img:
                width, height = img.size
                format_name = img.format

            record = {
                "id": f"{craft_name_clean}_{idx:06d}",
                "filename": filename,
                "raw_path": str(dest_file),
                "craft": craft_name_clean,
                "source": "local_directory",
                "source_path": str(img_file),
                "license": license_info,
                "width": width,
                "height": height,
                "format": format_name,
                "initial_caption": CRAFT_METADATA.get(craft_name_clean, {}).get("default_trigger", f"{craft_name_clean} traditional art")
            }
            records.append(record)
        except Exception as e:
            logger.error(f"Failed to ingest image {img_file}: {e}")

    if records:
        with open(RAW_METADATA_PATH, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(records)} raw metadata records to {RAW_METADATA_PATH}")

    return records

def fetch_wikimedia_craft_images(craft_name: str, max_images: int = 50) -> List[Dict]:
    """
    Fetches open-license (CC-BY / Public Domain) high-res imagery for a craft from Wikimedia Commons API.
    """
    ensure_directories()
    craft_name_clean = craft_name.lower().strip()
    target_raw_dir = RAW_WEB_DIR / craft_name_clean
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    search_query = CRAFT_METADATA.get(craft_name_clean, {}).get("full_name", f"{craft_name_clean} traditional art")
    search_keywords = CRAFT_METADATA.get(craft_name_clean, {}).get("keywords", [craft_name_clean])
    query_str = f"{search_query} OR {' OR '.join(search_keywords[:3])}"

    url = (
        "https://commons.wikimedia.org/w/api.php?"
        "action=query&generator=search&"
        f"gsrsearch={urllib.parse.quote(query_str)}&"
        f"gsrlimit={max_images}&gsrnamespace=6&"
        "prop=imageinfo&iiprop=url|size|mime|extmetadata&format=json"
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "IndusCraftBot/1.0 (https://github.com/induscraft; research@induscraft.org)"}
    )

    logger.info(f"Searching Wikimedia Commons for '{craft_name_clean}' images...")
    records = []
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})

        image_urls = []
        for page_id, page_info in pages.items():
            imageinfo = page_info.get("imageinfo", [])
            if imageinfo:
                info = imageinfo[0]
                img_url = info.get("url")
                mime = info.get("mime", "")
                width = info.get("width", 0)
                height = info.get("height", 0)

                if img_url and "image" in mime and width >= 400 and height >= 400:
                    image_urls.append((img_url, width, height, info.get("extmetadata", {})))

        logger.info(f"Found {len(image_urls)} eligible web images for '{craft_name_clean}'. Downloading...")

        for idx, (img_url, width, height, metadata) in enumerate(tqdm(image_urls, desc=f"Downloading {craft_name_clean}")):
            ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1]
            if not ext or ext.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"

            filename = f"{craft_name_clean}_web_{idx:06d}{ext.lower()}"
            dest_file = target_raw_dir / filename

            try:
                img_req = urllib.request.Request(img_url, headers={"User-Agent": "IndusCraftBot/1.0"})
                with urllib.request.urlopen(img_req, timeout=15) as img_resp, open(dest_file, "wb") as f_out:
                    f_out.write(img_resp.read())

                # Verify downloaded file
                with Image.open(dest_file) as img:
                    real_width, real_height = img.size
                    img_format = img.format

                license_name = metadata.get("LicenseShortName", {}).get("value", "CC-BY/Public-Domain")
                title = metadata.get("ObjectName", {}).get("value", craft_name_clean)

                record = {
                    "id": f"{craft_name_clean}_web_{idx:06d}",
                    "filename": filename,
                    "raw_path": str(dest_file),
                    "craft": craft_name_clean,
                    "source": "wikimedia_commons",
                    "source_url": img_url,
                    "license": license_name,
                    "width": real_width,
                    "height": real_height,
                    "format": img_format,
                    "initial_caption": f"{craft_name_clean} traditional art: {title}"
                }
                records.append(record)
            except Exception as e:
                logger.warning(f"Could not download {img_url}: {e}")

    except Exception as e:
        logger.error(f"Wikimedia API search failed for '{craft_name_clean}': {e}")

    if records:
        with open(RAW_METADATA_PATH, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(records)} web metadata records for '{craft_name_clean}'")

    return records

def fetch_all_crafts_online(max_per_craft: int = 30) -> Dict[str, int]:
    """
    Automatically downloads craft design and wearable images across all 10 target Indian crafts.
    """
    ensure_directories()
    results = {}
    for craft_key in CRAFT_METADATA.keys():
        logger.info(f"=== Fetching online images for craft: '{craft_key}' ===")
        recs = fetch_wikimedia_craft_images(craft_key, max_images=max_per_craft)
        results[craft_key] = len(recs)
    return results

if __name__ == "__main__":
    ensure_directories()
    fetch_all_crafts_online(max_per_craft=10)

import os
import json
import time
import socket
import shutil
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

from data_pipeline.config import (
    RAW_DIR, RAW_HF_DIR, RAW_WEB_DIR, IMAGES_DIR, METADATA_DIR, REJECTED_DIR,
    MODELS_DIR, BASE_MODEL_DIR, LORA_MODEL_DIR, OUTPUTS_DIR, SAMPLES_DIR,
    CHECKPOINTS_DIR, LOGS_DIR, RAW_METADATA_PATH, CRAFT_METADATA
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Acquisition")

socket.setdefaulttimeout(10.0)

WIKIMEDIA_USER_AGENT = "IndusCraftBot/1.0 (https://github.com/VedantJadhav701/diffusion_model_craft; research@induscraft.org) Python-urllib/3.10"

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

def download_single_image_worker(task: tuple) -> Optional[Dict]:
    """
    Worker function for downloading 1024px scaled JPEG thumbnails.
    """
    idx, img_url, width, height, metadata, craft_name_clean, target_raw_dir = task

    filename = f"{craft_name_clean}_web_{idx:06d}.jpg"
    dest_file = target_raw_dir / filename

    try:
        req = urllib.request.Request(
            img_url,
            headers={
                "User-Agent": WIKIMEDIA_USER_AGENT,
                "Accept": "image/jpeg,image/webp,image/*;q=0.8"
            }
        )
        with urllib.request.urlopen(req, timeout=10.0) as response, open(dest_file, "wb") as f_out:
            f_out.write(response.read())

        with Image.open(dest_file) as img:
            real_width, real_height = img.size
            img_format = img.format

        if real_width < 300 or real_height < 300:
            if dest_file.exists():
                dest_file.unlink()
            return None

        license_name = metadata.get("LicenseShortName", {}).get("value", "CC-BY/Public-Domain")
        title = metadata.get("ObjectName", {}).get("value", craft_name_clean)

        return {
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
    except Exception:
        if dest_file.exists():
            try:
                dest_file.unlink()
            except Exception:
                pass
        return None

def fetch_wikimedia_craft_images(craft_name: str, max_images: int = 500) -> List[Dict]:
    """
    Fetches open-license 1024px scaled image previews from Wikimedia Commons API.
    """
    ensure_directories()
    craft_name_clean = craft_name.lower().strip()
    target_raw_dir = RAW_WEB_DIR / craft_name_clean
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    craft_search_map = {
        "chikankari": ["chikankari", "lucknow embroidery", "chikan work", "indian embroidery", "chikankari kurta", "white embroidery india"],
        "phulkari": ["phulkari", "punjabi phulkari", "phulkari dupatta", "bagh embroidery", "khaddar embroidery", "phulkari suit"],
        "kalamkari": ["kalamkari", "kalamkari saree", "srikalahasti kalamkari", "indian block print", "machilipatnam kalamkari", "kalamkari fabric"],
        "ajrakh": ["ajrakh", "ajrak", "kutch block print", "woodblock printing india", "ajrakh print", "indigo madder print"],
        "bandhani": ["bandhani", "bandhej", "tie dye saree india", "leheriya", "bandhani dupatta", "gujarat tie dye"],
        "kantha": ["kantha", "kantha stitch", "bengal embroidery", "kantha quilt", "kantha saree", "running stitch craft"],
        "paithani": ["paithani", "paithani saree", "peacock zari", "maharashtra silk", "paithani border", "yeola paithani"],
        "ikat": ["ikat", "pochampally", "patola", "double ikat", "ikat saree", "pasapalli"],
        "madhubani": ["madhubani", "mithila painting", "bihar art", "indian folk painting", "madhubani art", "mithila wall art"],
        "warli": ["warli", "warli painting", "warli art", "tribal painting india", "warli folk art", "maharashtra tribal art"]
    }

    search_terms = craft_search_map.get(craft_name_clean, [craft_name_clean, f"{craft_name_clean} art"])

    seen_urls = set()
    image_candidates = []

    for query in search_terms:
        if len(image_candidates) >= max_images:
            break

        url = (
            "https://commons.wikimedia.org/w/api.php?"
            "action=query&generator=search&"
            f"gsrsearch={urllib.parse.quote(query)}&"
            f"gsrlimit=100&gsrnamespace=6&"
            "prop=imageinfo&iiprop=url|size|mime|extmetadata&iiurlwidth=1024&format=json"
        )

        req = urllib.request.Request(url, headers={"User-Agent": WIKIMEDIA_USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=10.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})

            for page_id, page_info in pages.items():
                imageinfo = page_info.get("imageinfo", [])
                if imageinfo:
                    info = imageinfo[0]
                    img_url = info.get("thumburl", info.get("url"))
                    mime = info.get("mime", "")
                    width = info.get("thumbwidth", info.get("width", 0))
                    height = info.get("thumbheight", info.get("height", 0))

                    if img_url and img_url not in seen_urls:
                        seen_urls.add(img_url)
                        image_candidates.append((img_url, width, height, info.get("extmetadata", {})))
        except Exception as e:
            logger.warning(f"Query '{query}' failed: {e}")

    logger.info(f"Found {len(image_candidates)} candidates for '{craft_name_clean}'. Fast downloading up to {max_images}...")

    tasks = [
        (idx, item[0], item[1], item[2], item[3], craft_name_clean, target_raw_dir)
        for idx, item in enumerate(image_candidates[:max_images])
    ]

    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single_image_worker, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {craft_name_clean}"):
            res = future.result()
            if res:
                records.append(res)

    if records:
        with open(RAW_METADATA_PATH, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(records)} metadata records for '{craft_name_clean}'")

    return records

def fetch_hf_datasets_craft_images(craft_name: str, max_images: int = 500) -> List[Dict]:
    """
    Fetches craft images from public Hugging Face datasets (e.g. DiffusionDB / Indian Art).
    """
    ensure_directories()
    craft_name_clean = craft_name.lower().strip()
    target_raw_dir = RAW_HF_DIR / craft_name_clean
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    records = []
    try:
        from datasets import load_dataset
        logger.info(f"Searching Hugging Face Hub datasets for '{craft_name_clean}'...")
        # Querying public diffusiondb 2M dataset for matching prompts
        dataset = load_dataset("poloclub/diffusiondb", "2m_first_10k", split="train")
        
        matching_items = []
        keywords = CRAFT_METADATA.get(craft_name_clean, {}).get("keywords", [craft_name_clean])
        
        for item in dataset:
            prompt = item.get("prompt", "").lower()
            if any(kw in prompt for kw in keywords) or craft_name_clean in prompt:
                matching_items.append(item)
                if len(matching_items) >= max_images:
                    break

        logger.info(f"Found {len(matching_items)} HF dataset matches for '{craft_name_clean}'. Saving locally...")
        for idx, item in enumerate(matching_items):
            filename = f"{craft_name_clean}_hf_{idx:06d}.jpg"
            dest_file = target_raw_dir / filename
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dest_file, "JPEG", quality=95)

            record = {
                "id": f"{craft_name_clean}_hf_{idx:06d}",
                "filename": filename,
                "raw_path": str(dest_file),
                "craft": craft_name_clean,
                "source": "huggingface_diffusiondb",
                "license": "open_access",
                "width": img.width,
                "height": img.height,
                "format": "JPEG",
                "initial_caption": item.get("prompt", f"{craft_name_clean} art")
            }
            records.append(record)

        if records:
            with open(RAW_METADATA_PATH, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(records)} HF dataset records for '{craft_name_clean}'")

    except Exception as e:
        logger.warning(f"HF Dataset fetch failed for '{craft_name_clean}': {e}")

    return records

def fetch_all_crafts_online(max_per_craft: int = 500) -> Dict[str, int]:
    """
    Automatically downloads craft design and wearable preview images across all 10 target Indian crafts.
    """
    ensure_directories()
    results = {}
    for craft_key in CRAFT_METADATA.keys():
        logger.info(f"=== Fetching online images for craft: '{craft_key}' ===")
        recs_web = fetch_wikimedia_craft_images(craft_key, max_images=max_per_craft)
        recs_hf = fetch_hf_datasets_craft_images(craft_key, max_images=max_per_craft)
        results[craft_key] = len(recs_web) + len(recs_hf)
    return results

if __name__ == "__main__":
    ensure_directories()
    fetch_all_crafts_online(max_per_craft=500)

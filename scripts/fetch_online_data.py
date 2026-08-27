import argparse
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_pipeline.acquisition import ensure_directories, fetch_wikimedia_craft_images, fetch_all_crafts_online, CRAFT_METADATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FetchOnlineData")

def main():
    parser = argparse.ArgumentParser(description="Fetch Indian Traditional Craft & Wearable images from open web/HF sources")
    parser.add_argument("--craft", type=str, default="all", help="Specific craft key (or 'all' for all 10 crafts)")
    parser.add_argument("--max-per-craft", type=int, default=30, help="Maximum images to fetch per craft category")
    args = parser.parse_args()

    ensure_directories()

    if args.craft.lower() == "all":
        logger.info(f"Downloading up to {args.max_per_craft} open-access images across all 10 Indian craft categories...")
        summary = fetch_all_crafts_online(max_per_craft=args.max_per_craft)
        total = sum(summary.values())
        logger.info(f"=== DOWNLOAD COMPLETE ===")
        for craft, count in summary.items():
            logger.info(f"  - {craft}: {count} images")
        logger.info(f"Total downloaded images: {total}")
    else:
        craft_key = args.craft.lower().strip()
        if craft_key not in CRAFT_METADATA:
            logger.warning(f"Craft '{craft_key}' not recognized. Available: {list(CRAFT_METADATA.keys())}")
        recs = fetch_wikimedia_craft_images(craft_key, max_images=args.max_per_craft)
        logger.info(f"Downloaded {len(recs)} images for '{craft_key}'.")

if __name__ == "__main__":
    main()

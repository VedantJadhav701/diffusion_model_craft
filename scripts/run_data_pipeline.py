import argparse
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_pipeline.acquisition import ensure_directories, ingest_local_directory, RAW_METADATA_PATH
from data_pipeline.cleaning import run_cleaning_pipeline, CLEANED_METADATA_PATH
from data_pipeline.captioning import process_and_caption_dataset
from data_pipeline.hf_uploader import prepare_and_push_to_hf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PipelineRunner")

def main():
    parser = argparse.ArgumentParser(description="IndusCraft End-to-End Data Pipeline")
    parser.add_argument("--source-dir", type=str, default=None, help="Directory containing raw images to ingest")
    parser.add_argument("--craft", type=str, default="chikankari", help="Craft category label for source images")
    parser.add_argument("--vlm-model", type=str, default=None, help="HF model name for VLM captioning (e.g. Salesforce/blip2-opt-2.7b or microsoft/Florence-2-base)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--push-hf", type=str, default=None, help="Hugging Face repo ID to upload dataset (e.g. username/induscraft-dataset)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--private", action="store_true", help="Set uploaded HF dataset as private")
    
    args = parser.parse_args()

    ensure_directories()

    # Step 1: Ingestion
    if args.source_dir:
        logger.info(f"=== STEP 1: Ingesting raw images from {args.source_dir} ===")
        ingest_local_directory(args.source_dir, args.craft)
    else:
        logger.info("=== STEP 1: Ingestion skipped (no --source-dir provided). Using existing raw metadata if present. ===")

    if not RAW_METADATA_PATH.exists():
        logger.error("No raw metadata found. Please specify --source-dir to ingest raw images first.")
        return

    # Read raw records
    raw_records = []
    with open(RAW_METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    # Step 2: Cleaning & Deduplication
    logger.info("=== STEP 2: Cleaning & Deduplicating Images ===")
    cleaned_records = run_cleaning_pipeline(raw_records)

    if not cleaned_records:
        logger.error("No valid clean images remaining after cleaning phase.")
        return

    # Step 3: VLM Captioning & Normalization & Train/Val Split
    logger.info("=== STEP 3: VLM Captioning & Normalization ===")
    train_recs, val_recs = process_and_caption_dataset(
        cleaned_records,
        vlm_model_name=args.vlm_model,
        val_split_ratio=args.val_split
    )

    logger.info(f"=== PIPELINE COMPLETE ===")
    logger.info(f"Train samples: {len(train_recs)} | Validation samples: {len(val_recs)}")

    # Step 4: Hugging Face Upload (Optional)
    if args.push_hf:
        logger.info(f"=== STEP 4: Uploading Dataset to Hugging Face ({args.push_hf}) ===")
        prepare_and_push_to_hf(args.push_hf, token=args.hf_token, private=args.private)

if __name__ == "__main__":
    main()

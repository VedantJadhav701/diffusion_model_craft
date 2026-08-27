import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from tqdm import tqdm

from data_pipeline.config import (
    CLEANED_METADATA_PATH, LAYER1_TRAIN_METADATA_PATH, LAYER1_VAL_METADATA_PATH,
    LAYER2_TRAIN_METADATA_PATH, LAYER2_VAL_METADATA_PATH, LAYER3_TRAIN_METADATA_PATH,
    LAYER3_VAL_METADATA_PATH, CRAFT_METADATA, GARMENT_REGIONS, DESIGN_VIEWS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Captioning")

class VLMCaptioner:
    """
    Interface for local VLM models (e.g. Florence-2, Qwen2-VL, BLIP-2) or rule-based craft captioning.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.pipeline = None
        if model_name:
            try:
                from transformers import pipeline
                logger.info(f"Loading VLM captioner pipeline: {model_name}...")
                self.pipeline = pipeline("image-to-text", model=model_name)
            except Exception as e:
                logger.warning(f"Could not load VLM model '{model_name}': {e}. Falling back to craft metadata builder.")

    def generate_raw_description(self, image_path: str) -> str:
        if self.pipeline:
            try:
                result = self.pipeline(image_path)
                if result and isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
            except Exception as e:
                logger.error(f"VLM inference failed for {image_path}: {e}")
        return ""

def build_layer1_craft_caption(craft_key: str, view_type: str = "front_view", vlm_description: str = "") -> str:
    """
    Generates standardized caption for Layer 1 (Craft Knowledge) with Multi-view tags.
    """
    craft_info = CRAFT_METADATA.get(craft_key.lower(), {})
    trigger = craft_info.get("default_trigger", f"{craft_key} style art")
    region = craft_info.get("region", "India")
    description = craft_info.get("description", "traditional intricate handcrafted textile art")

    view_tag = f"[{view_type.replace('_', ' ')}]"
    prefix = f"A high resolution detailed fashion design photograph, {view_tag}, of authentic {trigger}"
    region_clause = f"originating from {region}" if region else ""

    if vlm_description:
        vlm_clean = vlm_description.lower().strip()
        for noise in ["a picture of", "an image of", "a photo of", "there is"]:
            if vlm_clean.startswith(noise):
                vlm_clean = vlm_clean[len(noise):].strip()
        caption = f"{prefix}, {region_clause}. Depicting {vlm_clean}. Characterized by {description}."
    else:
        caption = f"{prefix}, {region_clause}, featuring {description}."

    return " ".join(caption.split()).replace(" ,", ",").replace(" .", ".")

def build_layer2_application_instruction(craft_key: str, region: str = "collar", garment: str = "kurta") -> str:
    """
    Generates targeted regional editing instruction for Layer 2 (Design Application).
    Example: 'Apply subtle Lucknowi chikankari embroidery on collar and sleeve cuffs of the white kurta, keeping the rest of the garment plain.'
    """
    craft_info = CRAFT_METADATA.get(craft_key.lower(), {})
    full_name = craft_info.get("full_name", f"{craft_key} craft")
    return f"Apply authentic {full_name} onto the {region} of the {garment}, preserving the original garment structure and background."

def process_multi_layer_dataset(
    cleaned_records: List[Dict],
    vlm_model_name: Optional[str] = None,
    val_split_ratio: float = 0.1,
    seed: int = 42
):
    """
    Processes cleaned image records into 3 Multi-Layer dataset metadata files.
    """
    vlm = VLMCaptioner(vlm_model_name)

    layer1_records = []
    layer2_records = []
    layer3_records = []

    logger.info(f"Categorizing and captioning {len(cleaned_records)} records into 3 Dataset Layers...")

    for idx, rec in enumerate(tqdm(cleaned_records, desc="Multi-Layer Processing")):
        craft_key = rec.get("craft", "chikankari")
        image_path = rec.get("clean_path", "")

        raw_vlm = ""
        if image_path and os.path.exists(image_path):
            raw_vlm = vlm.generate_raw_description(image_path)

        # Distribute into Layer 1 (Reference), Layer 2 (Application), and Layer 3 (Details)
        rec["vlm_raw_caption"] = raw_vlm

        # Assign view type cyclically or based on features
        view_type = DESIGN_VIEWS[idx % len(DESIGN_VIEWS)]
        target_region = GARMENT_REGIONS[idx % len(GARMENT_REGIONS)]

        # Layer 1 Record (Craft Reference)
        rec_l1 = rec.copy()
        rec_l1["view_type"] = view_type
        rec_l1["text"] = build_layer1_craft_caption(craft_key, view_type, raw_vlm)
        rec_l1["prompt"] = rec_l1["text"]
        layer1_records.append(rec_l1)

        # Layer 2 Record (Design Application Pair Instruction)
        rec_l2 = rec.copy()
        rec_l2["target_region"] = target_region
        rec_l2["instruction"] = build_layer2_application_instruction(craft_key, target_region, "garment")
        rec_l2["text"] = rec_l2["instruction"]
        rec_l2["prompt"] = rec_l2["instruction"]
        layer2_records.append(rec_l2)

        # Layer 3 Record (Macro Design Details)
        if view_type in ["detail_view", "pattern_view"] or idx % 3 == 0:
            rec_l3 = rec.copy()
            rec_l3["text"] = f"A macro detailed close-up shot of {craft_key} {view_type.replace('_', ' ')} texture, displaying intricate motif patterns and textile craftsmanship."
            rec_l3["prompt"] = rec_l3["text"]
            layer3_records.append(rec_l3)

    random.seed(seed)
    for records_list, train_path, val_path, layer_name in [
        (layer1_records, LAYER1_TRAIN_METADATA_PATH, LAYER1_VAL_METADATA_PATH, "Layer 1 Craft Reference"),
        (layer2_records, LAYER2_TRAIN_METADATA_PATH, LAYER2_VAL_METADATA_PATH, "Layer 2 Application Pairs"),
        (layer3_records, LAYER3_TRAIN_METADATA_PATH, LAYER3_VAL_METADATA_PATH, "Layer 3 Design Details"),
    ]:
        random.shuffle(records_list)
        num_val = int(len(records_list) * val_split_ratio)
        val_recs = records_list[:num_val]
        train_recs = records_list[num_val:]

        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for r in val_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"{layer_name}: {len(train_recs)} train, {len(val_recs)} val records written.")

if __name__ == "__main__":
    if CLEANED_METADATA_PATH.exists():
        recs = []
        with open(CLEANED_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        process_multi_layer_dataset(recs)

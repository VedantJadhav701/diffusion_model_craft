import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_HF_DIR = RAW_DIR / "huggingface"
RAW_WEB_DIR = RAW_DIR / "web"
IMAGES_DIR = DATA_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"
REJECTED_DIR = DATA_DIR / "rejected"

# Sub-directories for 3 Dataset Layers
LAYER1_CRAFT_REF_DIR = IMAGES_DIR / "layer1_craft_ref"
LAYER2_APP_PAIRS_DIR = IMAGES_DIR / "layer2_app_pairs"
LAYER3_DESIGN_DETAILS_DIR = IMAGES_DIR / "layer3_design_details"

MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "base"
LORA_MODEL_DIR = MODELS_DIR / "lora"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Metadata JSONL Paths for the 3 Dataset Layers
RAW_METADATA_PATH = METADATA_DIR / "raw.jsonl"
CLEANED_METADATA_PATH = METADATA_DIR / "cleaned.jsonl"
LAYER1_TRAIN_METADATA_PATH = METADATA_DIR / "layer1_craft_ref_train.jsonl"
LAYER1_VAL_METADATA_PATH = METADATA_DIR / "layer1_craft_ref_val.jsonl"
LAYER2_TRAIN_METADATA_PATH = METADATA_DIR / "layer2_app_pairs_train.jsonl"
LAYER2_VAL_METADATA_PATH = METADATA_DIR / "layer2_app_pairs_val.jsonl"
LAYER3_TRAIN_METADATA_PATH = METADATA_DIR / "layer3_design_details_train.jsonl"
LAYER3_VAL_METADATA_PATH = METADATA_DIR / "layer3_design_details_val.jsonl"

CRAFT_METADATA = {
    "base_garment": {
        "full_name": "Indian Ethnic Wear",
        "region": "India",
        "type": "Garment Structure",
        "keywords": ["ethnic wear", "traditional garment", "plain outfit", "unembellished clothing"],
        "default_trigger": "plain Indian ethnic clothing",
        "description": "traditional Indian garment silhouette including kurtas, sarees, lehengas, and sherwanis without heavy pattern embellishment"
    },
    "chikankari": {
        "full_name": "Chikankari Embroidery",
        "region": "Lucknow, Uttar Pradesh, India",
        "type": "Embroidery",
        "keywords": ["chikankari", "lucknow embroidery", "white floral embroidery", "chikan work", "jaali motif"],
        "default_trigger": "chikankari embroidery style",
        "description": "delicate and artful hand embroidery traditionally done on fine muslin, cotton, or silk using white thread with shadow work, floral motifs, and intricate jaali (lattice) patterns"
    },
    "phulkari": {
        "full_name": "Phulkari Embroidery",
        "region": "Punjab, India",
        "type": "Embroidery",
        "keywords": ["phulkari", "punjabi phulkari", "flower craft embroidery", "bagh dupatta"],
        "default_trigger": "phulkari embroidery style",
        "description": "vibrant floral geometric embroidery originating from Punjab, using bright untwisted silk floss (pat) threads on coarse cotton cloth (khaddar)"
    },
    "kalamkari": {
        "full_name": "Kalamkari Painting & Textile",
        "region": "Andhra Pradesh & Telangana, India",
        "type": "Hand-painted & Block Printed Textile",
        "keywords": ["kalamkari", "srikalahasti kalamkari", "machilipatnam kalamkari", "pen painted textile"],
        "default_trigger": "kalamkari art style",
        "description": "traditional hand-painted or block-printed cotton textile art depicting mythological scenes, tree of life, peacocks, and intricate floral vines using natural organic dyes"
    },
    "ajrakh": {
        "full_name": "Ajrakh Block Print",
        "region": "Kutch, Gujarat & Barmer, Rajasthan, India",
        "type": "Block Print",
        "keywords": ["ajrakh", "ajrak block print", "kutch woodblock print", "indigo madder print"],
        "default_trigger": "ajrakh block print style",
        "description": "traditional double-sided woodblock printed textile featuring complex geometric and floral tiled patterns dyed with deep indigo, crimson madder, and resist-dyed earth colors"
    },
    "bandhani": {
        "full_name": "Bandhani Tie-Dye",
        "region": "Gujarat & Rajasthan, India",
        "type": "Tie-Dye",
        "keywords": ["bandhani", "bandhej", "tie dye india", "leheriya", "dot print textile"],
        "default_trigger": "bandhani tie-dye style",
        "description": "ancient tie-dye textile technique forming precise intricate clusters of small dot motifs, square grids, and vibrant contrasting color fields"
    },
    "kantha": {
        "full_name": "Kantha Stitch Craft",
        "region": "West Bengal & Bangladesh",
        "type": "Embroidery & Quilt",
        "keywords": ["kantha", "kantha stitch", "bengal kantha", "running stitch embroidery"],
        "default_trigger": "kantha embroidery style",
        "description": "expressive traditional running-stitch embroidery technique creating textured folk motifs, floral mandalas, animals, and story scenes on layered textiles"
    },
    "paithani": {
        "full_name": "Paithani Weave",
        "region": "Paithan, Maharashtra, India",
        "type": "Silk Weaving",
        "keywords": ["paithani", "paithani saree", "peacock zari border", "tapestry silk weave"],
        "default_trigger": "paithani silk weave style",
        "description": "luxurious hand-woven silk textile characterized by rich oblique square borders and pallu featuring gold zari thread work with peacock (mor), lotus, and parrot motifs"
    },
    "ikat": {
        "full_name": "Ikat Textile Weave",
        "region": "Odisha, Telangana (Pochampally), & Gujarat (Patan), India",
        "type": "Resist-Dyed Weave",
        "keywords": ["ikat", "double ikat", "pochampally ikat", "patan patola", "pasapalli weave"],
        "default_trigger": "ikat woven pattern style",
        "description": "masterful resist-dyed yarn woven textile displaying signature feathered geometric edges, diamond grids, and rhythmic traditional motifs"
    },
    "madhubani": {
        "full_name": "Madhubani / Mithila Painting",
        "region": "Mithila region, Bihar, India",
        "type": "Folk Painting",
        "keywords": ["madhubani painting", "mithila art", "bihar folk art", "double line outline painting"],
        "default_trigger": "madhubani painting style",
        "description": "vibrant folk art style featuring line drawings filled with vivid natural pigments, double line outlines, geometric patterns, peacocks, fish, and mythical figures"
    },
    "warli": {
        "full_name": "Warli Tribal Art",
        "region": "Maharashtra, India",
        "type": "Tribal Painting",
        "keywords": ["warli painting", "warli art", "maharashtra tribal art", "geometric folk painting"],
        "default_trigger": "warli tribal art style",
        "description": "minimalist ancient tribal art composed of basic geometric shapes (circles, triangles, lines) painted in white pigment on red ochre or earth backgrounds depicting village life and ritual dances"
    },
    "zardozi": {
        "full_name": "Zardozi & Gota Patti Embroidery",
        "region": "Rajasthan & Uttar Pradesh, India",
        "type": "Heavy Metallic Embroidery",
        "keywords": ["zardozi", "gota patti", "gold thread embroidery", "zari embroidery", "royal embroidery"],
        "default_trigger": "zardozi metallic embroidery style",
        "description": "regal heavy embroidery using gold and silver metallic threads, beads, sequin work, and metallic ribbon applique on velvet, silk, or satin fabrics"
    },
    "kashida": {
        "full_name": "Kashida & Aari Embroidery",
        "region": "Kashmir, India",
        "type": "Needlework Embroidery",
        "keywords": ["kashida", "kashmiri embroidery", "aari work", "kashmir shawl embroidery", "chinar motif"],
        "default_trigger": "kashmiri kashida embroidery style",
        "description": "exquisite Kashmiri chain-stitch and needlework embroidery featuring chinar leaves, paisley motifs, floral vines, and rich jewel-tone wool/silk threads"
    },
    "pattachitra": {
        "full_name": "Pattachitra Scroll Painting",
        "region": "Odisha & West Bengal, India",
        "type": "Traditional Cloth Scroll Painting",
        "keywords": ["pattachitra", "patachitra", "odisha painting", "raghurajpur art", "scroll painting india"],
        "default_trigger": "pattachitra scroll painting style",
        "description": "intricate traditional cloth scroll painting depicting mythological narratives, crisp dark outlines, decorative borders, and vibrant natural mineral colors"
    },
    "bagh": {
        "full_name": "Bagh Block Print",
        "region": "Bagh, Dhar, Madhya Pradesh, India",
        "type": "Vegetable Dye Block Print",
        "keywords": ["bagh print", "bagh block print", "madhya pradesh block print", "natural dye block print"],
        "default_trigger": "bagh block print style",
        "description": "traditional hand-block printed cotton textile using carved wooden blocks with natural black and red vegetable dyes forming bold geometric and floral grids"
    },
    "banarasi": {
        "full_name": "Banarasi Silk Brocade Weave",
        "region": "Varanasi, Uttar Pradesh, India",
        "type": "Silk & Zari Brocade Weaving",
        "keywords": ["banarasi saree", "banarasi silk", "zari brocade", "varanasi weave", "kinkhab"],
        "default_trigger": "banarasi silk brocade weave style",
        "description": "opulent hand-woven silk brocade textile featuring intricate gold and silver zari metallic thread work, floral jaal weaves, and heavy ornamental borders"
    },
    "chanderi": {
        "full_name": "Chanderi & Maheshwari Weave",
        "region": "Madhya Pradesh, India",
        "type": "Sheer Silk-Cotton Weaving",
        "keywords": ["chanderi", "chanderi saree", "maheshwari weave", "sheer silk cotton", "zari border saree"],
        "default_trigger": "chanderi sheer weave style",
        "description": "lightweight, glossy, sheer silk-cotton woven fabric characterized by delicate zari borders, small coin (ashrafi) motifs, and fine translucent texture"
    },
    "pithora": {
        "full_name": "Pithora Tribal Art",
        "region": "Gujarat & Madhya Pradesh, India",
        "type": "Tribal Folk Painting",
        "keywords": ["pithora painting", "rathwa art", "tribal wall painting", "pithora horse art"],
        "default_trigger": "pithora tribal painting style",
        "description": "sacred tribal wall painting featuring stylized figures of horses, suns, sacred trees, and village deity rituals rendered in bright primary pigments"
    },
    "toda": {
        "full_name": "Toda Embroidery (Pukhoor)",
        "region": "Nilgiris, Tamil Nadu, India",
        "type": "Reversible Darning Stitch Embroidery",
        "keywords": ["toda embroidery", "pukhoor", "nilgiri craft", "toda shawl", "black red embroidery"],
        "default_trigger": "toda geometric embroidery style",
        "description": "distinctive reversible darning-stitch embroidery done on unbleached white cotton using bold red and black woolen threads in precise geometric stripes"
    }
}

# Fashion Regions & Design View Categories
GARMENT_REGIONS = ["collar", "sleeves", "cuffs", "chest", "hemline", "body", "neckline", "full_garment"]
DESIGN_VIEWS = ["front_view", "detail_view", "pattern_view", "flat_garment"]

# Image Cleaning & Quality Filters
MIN_IMAGE_WIDTH = 512
MIN_IMAGE_HEIGHT = 512
MAX_ASPECT_RATIO = 2.5
DUPLICATE_PHASH_THRESHOLD = 6

# Default Diffusion Model Settings
DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_INPAINTING_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_TRAIN_RESOLUTION = 768

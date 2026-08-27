# 🎨 IndusCraft: Indian Traditional Craft & Textile AI Design Engine for Fashion Education

**IndusCraft** is an AI-powered fashion and textile design platform designed for fashion education (e.g. Pearl Academy). It enables students and designers to generate authentic Indian traditional craft design concepts and apply crafts directly to existing garments.

---

## 💡 Dual Core AI Workflows

```
                                INDUS CRAFT
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
  WORKFLOW A: CREATE NEW DESIGN                           WORKFLOW B: APPLY CRAFT
 (Text / Image -> Multi-View Generation)               (Image + Mask -> Regional Editing)
         │                                                       │
 ┌───────┼───────────────┬────────────────┐              ┌───────┴───────┐
 ▼       ▼               ▼                ▼              ▼               ▼
Front  Detail         Pattern           Flat          Upload Garment   Select Region & Craft
View   View           View              View          (Preserved)     (Collar, Sleeves, Cuffs)
```

### 1. Workflow A — Create New Design (Multi-View Design Exploration)
- **Input**: Text prompt (e.g., *"Create a white kurta with intricate Lucknowi chikankari embroidery, floral jaali pattern, contemporary silhouette"*).
- **Multi-View Outputs**:
  - `[front view]`: Full garment presentation on model/mannequin.
  - `[detail view]`: Macro close-up shot of stitch work, embroidery, or weave texture.
  - `[pattern view]`: Seamless, repeating textile print or motif layout.
  - `[flat garment]`: Technical flat layout for garment pattern making.

### 2. Workflow B — Apply Craft to Existing Garment (Regional Inpainting & Garment Preservation)
- **Input**: Student's uploaded blank garment image + targeted region selection (`Collar`, `Sleeves`, `Cuffs`, `Chest`, `Hemline`) + Craft instruction.
- **Output**: The student's **original garment is strictly preserved** in shape, fit, background, and unselected areas, while authentic craft embroidery/print is seamlessly inpainted into the chosen region.

---

## 📊 3-Layer Dataset Architecture

| Layer | Dataset Name | Objective | Contents |
|---|---|---|---|
| **Layer 1** | **Craft Reference Knowledge** | Teaches authentic craft aesthetics | 50K-100K images + VLM craft captions (`chikankari`, `phulkari`, `kalamkari`, `ajrakh`, `bandhani`, `kantha`, `paithani`, `ikat`, `madhubani`, `warli`). |
| **Layer 2** | **Garment Application Pairs** | Teaches regional inpainting & garment preservation | 20K-50K pairs: Plain garment image + region mask + editing instruction + designed garment. |
| **Layer 3** | **Design Details & Macro Views** | Teaches macro texture & motif repeatability | 10K-30K macro close-up shots of collars, cuffs, sleeves, necklines, hemlines, stitch work, and repeat prints. |

---

## 📁 Repository Structure

```
diffusion_model/
├── context.md               # Vision document & architectural blueprint
├── requirements.txt         # Dependencies (diffusers, transformers, peft, datasets, imagehash)
├── README.md                # Usage guide & instructions
│
├── data_pipeline/           # Multi-Layer Data Processing Package
│   ├── config.py            # Craft metadata, 3 dataset layer paths, region tags, view types
│   ├── acquisition.py       # Automated web/HF craft image fetcher & local folder ingester
│   ├── cleaning.py          # Integrity check, min resolution (512x512), pHash deduplication
│   ├── captioning.py         # 3-Layer prompt normalizer (Multi-view tags & regional instructions)
│   └── hf_uploader.py       # Multi-config Hugging Face Hub dataset uploader
│
├── scripts/                 # Executable Python Scripts
│   ├── fetch_online_data.py # Auto-download craft images for all 10 crafts
│   ├── run_data_pipeline.py # End-to-end data pipeline runner
│   ├── push_to_hf.py        # Upload prepared 3-layer dataset to HF Hub
│   ├── train_sdxl_lora.py   # Stage 1: Craft Knowledge & Multi-View UNet LoRA Training
│   └── train_sdxl_inpainting_lora.py # Stage 2: Regional Garment Inpainting LoRA Training
│
└── notebooks/               # Interactive Jupyter Notebooks
    ├── 01_dataset_preparation_and_hf_push.ipynb  # Step 1: Prep 3-layer dataset & upload to HF
    └── 02_sdxl_lora_training.ipynb               # Step 2: Cloud GPU Multi-Stage LoRA Training
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
conda activate thermo_agent
pip install -r requirements.txt
```

### 2. Fetch Data, Prepare 3-Layer Dataset & Upload to HF Hub
Open [`notebooks/01_dataset_preparation_and_hf_push.ipynb`](file:///c:/Users/HP/projects/diffusion_model/notebooks/01_dataset_preparation_and_hf_push.ipynb) or run from CLI:
```bash
# Auto-download open-access images across all 10 Indian crafts
python scripts/fetch_online_data.py --craft all --max-per-craft 30

# Process 3-layer dataset and upload to Hugging Face Hub
python scripts/run_data_pipeline.py --push-hf "your-hf-username/induscraft-dataset"
```

### 3. Train Models on Cloud GPU (DGX / RunPod / Colab Pro / Kaggle)

#### Stage 1 — Train Craft Knowledge & Multi-View LoRA (Workflow A)
```bash
python scripts/train_sdxl_lora.py \
    --pretrained-model "stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset-name "your-hf-username/induscraft-dataset" \
    --output-dir "./outputs/induscraft_stage1_craft_lora" \
    --resolution 768 \
    --train-batch-size 2 \
    --num-train-epochs 10 \
    --lora-rank 32
```

#### Stage 2 — Train Regional Garment Inpainting LoRA (Workflow B)
```bash
python scripts/train_sdxl_inpainting_lora.py \
    --pretrained-model "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
    --dataset-name "your-hf-username/induscraft-dataset" \
    --output-dir "./outputs/induscraft_stage2_inpainting_lora" \
    --resolution 768 \
    --train-batch-size 2 \
    --num-train-epochs 10 \
    --lora-rank 32
```

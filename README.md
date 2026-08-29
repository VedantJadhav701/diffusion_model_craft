# 🎨 IndusCraft: Indian Traditional Craft & Textile AI Design Engine for Fashion Education

**IndusCraft** is an AI-powered fashion and textile design platform designed for fashion education (e.g. Pearl Academy). It enables students and designers to generate authentic Indian traditional craft design concepts and apply crafts directly to existing garments.

---

## 🚀 Official Published Models & Adapters on Hugging Face Hub

- 📊 **Multimodal Dataset**: [vedantjadhav701/induscraft-dataset](https://huggingface.co/datasets/vedantjadhav701/induscraft-dataset)
- ⚡ **Stage 1 Craft Knowledge LoRA Adapter**: [vedantjadhav701/induscraft-stage1-craft-lora](https://huggingface.co/vedantjadhav701/induscraft-stage1-craft-lora)
- ⚡ **Stage 2 Inpainting LoRA Adapter**: [vedantjadhav701/induscraft-stage2-inpainting-lora](https://huggingface.co/vedantjadhav701/induscraft-stage2-inpainting-lora)
- 🎨 **Stage 1 Merged SDXL Engine**: [vedantjadhav701/induscraft-sdxl-merged](https://huggingface.co/vedantjadhav701/induscraft-sdxl-merged)
- 🧥 **Stage 2 Merged SDXL Inpaint Engine**: [vedantjadhav701/induscraft-sdxl-inpaint-merged](https://huggingface.co/vedantjadhav701/induscraft-sdxl-inpaint-merged)

---

## 📈 Evaluation & Training Results

### 1. Training Convergence Metrics
Both Stage 1 and Stage 2 SDXL LoRAs were trained for 15 Epochs using PyTorch + Diffusers with automatic mixed precision (`fp16`) and gradient checkpointing:

| Stage | Training Target | Epochs | Rank / Alpha | Final Loss | Status |
|---|---|---|---|---|---|
| **Stage 1** | Craft Knowledge & Multi-View Generation | 15 | 64 / 64 | **0.0463** | ✅ Complete |
| **Stage 2** | Regional Garment Inpainting | 15 | 64 / 64 | **0.0177** | ✅ Complete |

### 2. Quantitative Evaluation (CLIP Score Alignment)
Evaluated using OpenAI ViT-B-32-quickgelu text-image alignment metric across sample multi-view generations:

| Sample ID | View Type / Prompt Description | CLIP Score |
|---|---|---|
| `design_sample_1.png` | **[front view]** Chikankari embroidery style on white modern kurta | **0.3178** |
| `design_sample_2.png` | **[detail view]** Macro close-up shot of floral jaali stitch texture | **0.3086** |
| `design_sample_3.png` | **[pattern view]** Seamless repeating Kalamkari tree of life print | **0.3060** |
| `design_sample_4.png` | **[flat garment]** Technical flat layout of Paithani silk saree | **0.3188** |
| **AVERAGE** | **Overall Prompt-Visual Alignment** | **0.3128** (Strong Alignment) |

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

The dataset is scaled to **1,702 total training samples** across three configurations:

| Layer | Configuration Name | Objective | Training Size |
|---|---|---|---|
| **Layer 1** | **`craft_reference`** | Teaches authentic craft aesthetics | **575 Train / 63 Val** |
| **Layer 2** | **`garment_application`** | Teaches regional inpainting & garment preservation | **575 Train / 63 Val** |
| **Layer 3** | **`design_details`** | Teaches macro texture & motif repeatability | **384 Train / 42 Val** |

---

## 💻 Interactive Streamlit Web Application

To launch the interactive testing GUI:

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
diffusion_model/
├── context.md               # Vision document & architectural blueprint
├── requirements.txt         # Dependencies (diffusers, transformers, peft, datasets, hf_transfer, python-dotenv)
├── README.md                # Usage guide & evaluation results
├── app.py                   # Interactive Streamlit Web Application
│
├── data_pipeline/           # Multi-Layer Data Processing Package
│   ├── config.py            # Craft metadata, 3 dataset layer paths, region tags, view types
│   ├── acquisition.py       # Rate-limiting Wikimedia craft image crawler (HTTP 429 handler)
│   ├── cleaning.py          # Integrity check, face detection filter, pHash deduplication
│   ├── captioning.py         # 3-Layer prompt normalizer (Multi-view tags & regional instructions)
│   └── hf_uploader.py       # Multi-config Hugging Face Hub dataset uploader with retries
│
├── scripts/                 # CLI Runners & Preprocessors
│   ├── run_data_pipeline.py # End-to-end dataset cleaning, captioning & HF upload CLI
│   ├── fetch_online_data.py # Wikimedia Commons bulk downloader (handles rate limits)
│   ├── integrate_indofashion.py # IndoFashion preprocessing and integration script
│   ├── train_sdxl_lora.py   # Stage 1 UNet LoRA Trainer (Craft Knowledge & Multi-View)
│   └── train_sdxl_inpainting_lora.py # Stage 2 Inpainting LoRA Trainer (Regional Editing)
│
└── notebooks/               # Interactive Colab / Kaggle / Jupyter Notebooks
    ├── 01_dataset_preparation_and_hf_push.ipynb
    └── 02_sdxl_lora_training.ipynb   # All-in-one standalone training notebook
```

---

## 🧵 Included Indian Craft Categories

- **Chikankari**: White-on-white delicate embroidery & shadow work (Lucknow, UP).
- **Phulkari**: Vibrant silk thread floral geometry embroidery (Punjab).
- **Kalamkari**: Hand-painted or block-printed organic dyes & mythological narrative art (Andhra Pradesh).
- **Ajrakh**: Block-printed geometric motifs with indigo & madder dyes (Kutch, Gujarat).
- **Bandhani**: Tie-and-dye intricate dot patterns (Rajasthan / Gujarat).
- **Kantha**: Running stitch quilted embroidery on cotton/silk (West Bengal).
- **Paithani**: Fine silk saree with gold zari peacock pallu borders (Maharashtra).
- **Ikat**: Resist-dyed warp/weft geometric weave (Telangana / Odisha).
- **Madhubani**: Traditional folk art lines with natural pigments (Mithila, Bihar).
- **Warli**: Tribal geometric figure art (Maharashtra).

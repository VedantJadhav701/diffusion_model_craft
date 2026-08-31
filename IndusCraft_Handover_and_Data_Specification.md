# 🎨 IndusCraft: Handover & Data Specification Guide

This document contains the complete technical specification, dataset architecture, and operational guidelines for the **IndusCraft** traditional Indian textile craft and fashion design AI engine. It is designed to serve as a comprehensive handover guide for team members responsible for data collection, cleaning, processing, and model training.

---

## 📋 1. Project Overview & Quick Reference

* **Project Objective**: An AI-powered fashion design engine that enables students (e.g. Pearl Academy) to generate multi-view traditional Indian craft concepts (Workflow A) and perform regional garment inpainting (Workflow B).
* **Target Model Architecture**: Stable Diffusion XL (SDXL Base 1.0) CONDITION UNetConditionModel with Parameter-Efficient Fine-Tuning (PEFT LoRA) at Rank 32/Alpha 32.
* **Target Hugging Face Dataset**: [vedantjadhav701/induscraft-dataset](https://huggingface.co/datasets/vedantjadhav701/induscraft-dataset)
* **Target GitHub Repository**: [VedantJadhav701/diffusion_model_craft](https://github.com/VedantJadhav701/diffusion_model_craft)

---

## 📊 2. Dataset Size & Structure

The dataset contains **638 high-resolution, cleaned master images** that have been formatted, captioned, and partitioned into **1,702 total training samples** across three configurations:

| Dataset Layer | Hugging Face Configuration | Purpose | Train Size | Val Size |
|---|---|---|---|---|
| **Layer 1** | `craft_reference` | Authentic craft aesthetics & multi-view generation | **575** | **63** |
| **Layer 2** | `garment_application` | Regional inpainting & garment preservation | **575** | **63** |
| **Layer 3** | `design_details` | Macro texture & motif repeatability | **384** | **42** |
| **TOTAL** | | | **1,534** | **168** |

---

## 🧵 3. Traditional Indian Crafts Covered (10 Categories)

The engine covers ten distinct traditional crafts, each indexed in the system using specific keywords and caption tags:

1. **Chikankari**: White-on-white shadow embroidery (Lucknow, Uttar Pradesh).
2. **Phulkari**: Vibrant silk thread floral geometry embroidery (Punjab).
3. **Kalamkari**: Hand-painted/block-printed natural organic dye fabrics (Andhra Pradesh).
4. **Ajrakh**: Double-sided woodblock printing with indigo and crimson madder (Kutch, Gujarat).
5. **Bandhani**: Tie-and-dye intricate dot and grid patterns (Gujarat/Rajasthan).
6. **Kantha**: running-stitch folk motif embroidery on layered textiles (West Bengal).
7. **Paithani**: Luxurious hand-woven silk with gold zari peacock pallu borders (Maharashtra).
8. **Ikat**: Resist-dyed warp/weft geometric weave (Odisha/Telangana).
9. **Madhubani**: Mithila folk line drawings with natural pigments (Bihar).
10. **Warli**: Tribal geometric figure wall art (Maharashtra).

---

## 📁 4. Data Format & Schema

The dataset is stored and pushed to Hugging Face as a partitioned Arrow table inside standard **Parquet shards**. Each row in the dataset adheres to the following metadata schema:

* `id` *(string)*: Unique identifier of the sample (e.g. `chikankari_web_000006`).
* `image` *(image/bytes)*: The raw image byte stream automatically handled by Hugging Face `datasets.Image`.
* `craft` *(string)*: The lowercased craft category key.
* `view_type` *(string)*: Multi-view tagging schema (`[front view]`, `[detail view]`, `[pattern view]`, `[flat garment]`).
* `text` / `prompt` *(string)*: Standardized text description containing view tags, craft identifiers, regional trigger details, and descriptive VLM captions.

---

## 🔍 5. Sample Data Preview (10 Representative Records)

Below is a preview of 10 records from the `craft_reference` dataset layer:

| Record ID | Craft Category | Multi-View Tag | Resolution | Sample Training Prompt |
|---|---|---|---|---|
| `chikankari_web_000006` | Chikankari | `[detail view]` | 700 × 618 | "A high resolution detailed fashion design photograph, **[detail view]**, of authentic chikankari embroidery style, originating from Lucknow, Uttar Pradesh, India, featuring delicate and artful hand embroidery..." |
| `madhubani_web_000044` | Madhubani | `[front view]` | 1280 × 772 | "A high resolution detailed fashion design photograph, **[front view]**, of authentic madhubani painting style, originating from Mithila region, Bihar, India, featuring vibrant folk art style..." |
| `madhubani_web_000102` | Madhubani | `[flat garment]` | 1280 × 1707 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic madhubani painting style, originating from Mithila region, Bihar, India, featuring vibrant folk art style..." |
| `ikat_web_000034` | Ikat | `[front view]` | 600 × 942 | "A high resolution detailed fashion design photograph, **[front view]**, of authentic ikat woven pattern style, originating from Odisha, Telangana (Pochampally), & Gujarat (Patan), India..." |
| `bandhani_web_000073` | Bandhani | `[flat garment]` | 960 × 1416 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic bandhani tie-dye style, originating from Gujarat & Rajasthan, India, featuring ancient tie-dye textile technique..." |
| `madhubani_web_000149` | Madhubani | `[detail view]` | 720 × 1280 | "A high resolution detailed fashion design photograph, **[detail view]**, of authentic madhubani painting style, originating from Mithila region, Bihar, India, featuring vibrant folk art..." |
| `ajrakh_web_000142` | Ajrakh | `[flat garment]` | 960 × 972 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic ajrakh block print style, originating from Kutch, Gujarat & Barmer, Rajasthan, India..." |
| `ajrakh_web_000135` | Ajrakh | `[front view]` | 960 × 1275 | "A high resolution detailed fashion design photograph, **[front view]**, of authentic ajrakh block print style, originating from Kutch, Gujarat & Barmer, Rajasthan, India..." |
| `kantha_web_000111` | Kantha | `[flat garment]` | 960 × 1280 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic kantha embroidery style, originating from West Bengal & Bangladesh, featuring running-stitch..." |
| `warli_web_000001` | Warli | `[detail view]` | 1280 × 960 | "A high resolution detailed fashion design photograph, **[detail view]**, of authentic warli tribal art style, originating from Maharashtra, India, featuring minimalist tribal art..." |

---

## 🔄 6. End-to-End Data Workflow

To scale up and maintain the dataset, follow this step-by-step workflow:

```
[Web Scraper / Local Folders] ──> [Cleaning & Face Filtering] ──> [Multi-View Captioning] ──> [HF Parquet Upload]
```

### Step 1: Ingesting Raw Data
You can either crawl new open-access images from the web or import local high-resolution fashion design folders:
* **Web Crawling**:
  ```bash
  python scripts/fetch_online_data.py --craft all --max-per-craft 300
  ```
  *(This queries Wikimedia Commons with throttled threads and automatic HTTP 429 rate-limiting retries)*
* **Local Ingestion**:
  ```bash
  python scripts/run_data_pipeline.py --source-dir "C:/path/to/your/images" --craft "chikankari"
  ```
  *(This copies local images directly into the raw data workspace)*

### Step 2: Processing and Pushing to Hugging Face
Run the end-to-end processing pipeline to clean, de-duplicate (via perceptual hashing), filter out human faces (using OpenCV), caption, and push to Hugging Face:
```bash
python scripts/run_data_pipeline.py --push-hf "vedantjadhav701/induscraft-dataset" --hf-token "YOUR_HF_TOKEN"
```

---

## ⏱️ 7. Training Time Estimates (Production Scale)

Based on observed metrics on an **NVIDIA A100 80GB GPU** (using bf16 precision, batch size 8, resolution 1024×1024):

* **Current Pilot Run (~1,500 samples)**: ~1 hour total execution time (approx. cost **$1.50 - $2.50**).
* **Full Production Run (~13,500 samples, 20 epochs)**:
  * Stage 1 (Craft Knowledge LoRA): **~6.1 hours**
  * Stage 2 (Regional Inpainting LoRA): **~7.6 hours**
  * Stage 3 (Design Details LoRA): **~2.2 hours**
  * **Total Production Compute Cost**: **~$22 - $45** (approx. 16–18 hours of A100 time).

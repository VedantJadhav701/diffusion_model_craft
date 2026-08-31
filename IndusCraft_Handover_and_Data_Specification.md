# 🎨 IndusCraft: Handover & Data Specification Guide (Starting From Scratch)

This document serves as the comprehensive technical specification, dataset architecture guide, and operational instructions for building the **IndusCraft** traditional Indian textile craft and fashion design AI engine. 

This guide is written from a **start-from-scratch** perspective, outlining how to collect, clean, structure, and upload the entire multi-layer dataset from zero images.

---

## 📋 1. Project Objective & Reference

* **Project Objective**: An AI-powered fashion design engine that enables students to generate multi-view traditional Indian craft concepts (Workflow A) and perform regional garment inpainting (Workflow B).
* **Target Model Architecture**: Stable Diffusion XL (SDXL Base 1.0) condition UNet with Parameter-Efficient Fine-Tuning (PEFT LoRA) at Rank 32/Alpha 32.
* **Hugging Face Repository**: `[INSERT_YOUR_HF_USERNAME/induscraft-dataset]`
* **GitHub Codebase**: `[INSERT_YOUR_GITHUB_REPO_URL]`

---

## 📊 2. Production Dataset Target Size

To train a robust, production-grade model across 10 traditional crafts, the target size for the dataset configurations should meet the following benchmarks:

| Dataset Configuration | Purpose | Target Training Images | Target Validation Images |
|---|---|---|---|
| `craft_reference` (Layer 1) | Authentic craft aesthetics & multi-view generation | **5,000+** | **500+** |
| `garment_application` (Layer 2) | Regional inpainting & garment preservation | **5,000+** | **500+** |
| `design_details` (Layer 3) | Macro texture & motif repeatability | **2,500+** | **250+** |
| **TOTAL TARGET** | | **~12,500** | **~1,250** |

*Note: For a shared LoRA covering 10 distinct craft styles, the pipeline should aim for a minimum of **500 high-quality images per craft** to prevent overfitting or style bleeding.*

---

## 🧵 3. Traditional Indian Crafts Covered (10 Categories)

The data collection and crawling pipeline must gather images specifically for these ten traditional Indian crafts:

1. **Chikankari**: White-on-white delicate shadow embroidery (Lucknow, Uttar Pradesh).
2. **Phulkari**: Vibrant silk thread floral geometry embroidery (Punjab).
3. **Kalamkari**: Hand-painted or block-printed organic dye designs (Andhra Pradesh).
4. **Ajrakh**: Double-sided woodblock printing with indigo and crimson madder (Kutch, Gujarat).
5. **Bandhani**: Tie-and-dye intricate dot and grid patterns (Gujarat/Rajasthan).
6. **Kantha**: running-stitch folk motif embroidery on cotton/silk (West Bengal).
7. **Paithani**: Luxurious hand-woven silk with gold zari peacock pallu borders (Maharashtra).
8. **Ikat**: Resist-dyed warp/weft geometric weave (Odisha/Telangana).
9. **Madhubani**: Mithila folk line drawings with natural pigments (Bihar).
10. **Warli**: Tribal geometric figure wall art (Maharashtra).

---

## 📁 4. Data Schema & Format

The final dataset is compiled, formatted, and pushed directly to Hugging Face as partitioned Arrow tables inside standard **Parquet shards**. The data loader reads the following schema for training:

* `id` *(string)*: Unique identifier of the sample (e.g. `[craft_name]_web_[index]`).
* `image` *(image/bytes)*: The raw image byte stream automatically handled by Hugging Face `datasets.Image`.
* `craft` *(string)*: The lowercased craft category key.
* `view_type` *(string)*: Multi-view tagging schema (`[front view]`, `[detail view]`, `[pattern view]`, `[flat garment]`).
* `text` / `prompt` *(string)*: Standardized text description containing view tags, craft identifiers, and detailed descriptive captions.

---

## 🔍 5. Sample Data Specification Preview

Here is a mockup preview of how the final output records will be formatted inside the `craft_reference` config:

| Record ID | Craft Category | Multi-View Tag | Target Resolution | Sample Target Training Prompt |
|---|---|---|---|---|
| `chikankari_web_000001` | Chikankari | `[detail view]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[detail view]**, of authentic chikankari embroidery style, originating from Lucknow, Uttar Pradesh, India, featuring delicate and artful hand embroidery..." |
| `madhubani_web_000001` | Madhubani | `[front view]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[front view]**, of authentic madhubani painting style, originating from Mithila region, Bihar, India, featuring vibrant folk art style..." |
| `madhubani_web_000002` | Madhubani | `[flat garment]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic madhubani painting style, originating from Mithila region, Bihar, India, featuring vibrant folk art style..." |
| `ikat_web_000001` | Ikat | `[front view]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[front view]**, of authentic ikat woven pattern style, originating from Odisha, Telangana (Pochampally), & Gujarat (Patan), India..." |
| `bandhani_web_000001` | Bandhani | `[flat garment]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic bandhani tie-dye style, originating from Gujarat & Rajasthan, India, featuring ancient tie-dye textile technique..." |
| `madhubani_web_000003` | Madhubani | `[detail view]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[detail view]**, of authentic madhubani painting style, originating from Mithila region, Bihar, India, featuring vibrant folk art..." |
| `ajrakh_web_000001` | Ajrakh | `[flat garment]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic ajrakh block print style, originating from Kutch, Gujarat & Barmer, Rajasthan, India..." |
| `ajrakh_web_000002` | Ajrakh | `[front view]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[front view]**, of authentic ajrakh block print style, originating from Kutch, Gujarat & Barmer, Rajasthan, India..." |
| `kantha_web_000001` | Kantha | `[flat garment]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[flat garment]**, of authentic kantha embroidery style, originating from West Bengal & Bangladesh, featuring running-stitch..." |
| `warli_web_000001` | Warli | `[detail view]` | Minimum 512 × 512 | "A high resolution detailed fashion design photograph, **[detail view]**, of authentic warli tribal art style, originating from Maharashtra, India, featuring minimalist tribal art..." |

---

## 🔄 6. End-to-End Data Workflow (How to Do It)

The pipeline is fully automated and runs sequentially:

```
[Web Scraper / Local Folders] ──> [Cleaning & Face Filtering] ──> [Multi-View Captioning] ──> [HF Parquet Upload]
```

### Step 1: Gathering Raw Images (Data Acquisition)
You can collect raw images from zero using either the automated web crawler or by importing offline folders.

* **Method A: Automated Web Crawler (Wikimedia Commons)**:
  Run the scraper to automatically search, verify, and download open-access images for all 10 crafts:
  ```bash
  python scripts/fetch_online_data.py --craft all --max-per-craft 500
  ```
  *Note: The crawler utilizes throttled threads and an exponential backoff retry handler to prevent HTTP 429 rate-limiting blocks.*

* **Method B: Local Ingestion (E-commerce / Private Folders)**:
  If you have directories of high-resolution images, import them directly:
  ```bash
  python scripts/run_data_pipeline.py --source-dir "C:/path/to/your/images" --craft "[craft_name]"
  ```

---

### Step 2: Cleaning, Filtering, and Captioning
Once the raw images are in place, the pipeline automatically processes the metadata to ensure high-fidelity inputs:

1. **Resolution & Integrity Check**: Deletes corrupted files and enforces a minimum size of **512x512 pixels**.
2. **Face Filtering**: Uses OpenCV's Haar Cascade classifier to scan for human faces and automatically moves them to a rejection folder (to keep focus solely on garments/textiles).
3. **Deduplication**: Runs a perceptual hashing (`pHash`) check to eliminate exact and near-identical duplicate files.
4. **VLM Captioning**: Auto-generates structured fashion prompts containing view-type tags (`[front view]`, `[detail view]`, etc.), craft origin triggers, and descriptive details.
5. **Split Generation**: Automatically partitions the cleaned files into `train` and `validation` subsets (default 90/10 split).

---

### Step 3: Pushing to Hugging Face
To compile the processed folders into Hugging Face Parquet configurations and push them to your target repository, run:

```bash
python scripts/run_data_pipeline.py --push-hf "[YOUR_HF_REP_ID]" --hf-token "[YOUR_HF_WRITE_TOKEN]"
```

---

## ⏱️ 7. Training Time Estimates (Production Scale)

Based on observed throughput benchmarks on a single **NVIDIA A100 80GB GPU** (1024x1024 resolution, bf16 precision, batch size 8):

* **Stage 1 (Craft Knowledge LoRA - 5,000 images, 20 epochs)**: **~6 hours**
* **Stage 2 (Regional Inpainting LoRA - 5,000 images, 20 epochs)**: **~7.5 hours**
* **Stage 3 (Design Details LoRA - 2,500 images, 20 epochs)**: **~2.2 hours**
* **Total Estimated Compute Time**: **~16 hours** of A100 execution.

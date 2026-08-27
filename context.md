# IndusCraft: Indian Traditional Craft & Textile Design System for Fashion Education

## 🎯 Vision & Core Concept
**IndusCraft** is an AI-powered fashion and textile design platform designed specifically for fashion design students and educators (e.g., Pearl Academy). Rather than acting solely as a generic image generator, IndusCraft provides two core workflows tailored to textile design, craft preservation, and garment customization.

```
                                INDUS CRAFT
                       (Indian Design Engine)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   WORKFLOW A: CREATE NEW DESIGN                  WORKFLOW B: APPLY CRAFT
   (Text / Image -> Multi-View)               (Image + Mask/Instruction -> Image)
         │                                                 │
 ┌───────┴────────┐                               ┌────────┴────────┐
 ▼                ▼                               ▼                 ▼
Front View   Detail View                       Upload Blank     Select Region & Craft
 Pattern     Flat Garment                        Garment       (Collar, Sleeves, Cuffs)
```

---

## 🛠️ Workflows Breakdown

### 1. Workflow A — Create New Design (Multi-View Design Exploration)
- **Input**: Text prompt (e.g., *"Create a white kurta with intricate Lucknowi chikankari embroidery, floral jaali pattern, contemporary silhouette"*).
- **Output**: 4–8 design variations featuring comprehensive fashion visualization:
  - **Front View**: Full garment presentation on model or mannequin.
  - **Detail View**: Macro close-up of stitch work, embroidery, or weave texture.
  - **Pattern View**: Seamless, repeatable textile print or motif layout.
  - **Flat Garment View**: Technical flat layout for garment construction.

### 2. Workflow B — Apply Craft to Existing Garment (Regional Inpainting & Garment Preservation)
- **Input**: Student's uploaded blank garment image + selected region (e.g., Collar, Sleeves, Chest, Hemline) + Craft instruction (e.g., *"Apply subtle chikankari embroidery on collar, cuffs and chest"*).
- **Output**: The student's **original garment strictly preserved** in shape, fit, background, and unselected areas, with authentic Indian craft seamlessly applied to the targeted region.

---

## 📊 3-Layer Dataset Architecture

To support both **Design Generation** and **Regional Garment Application**, the dataset is structured into three distinct layers:

| Layer | Dataset Type | Target Volume | Primary Objective | Key Features |
|---|---|---|---|---|
| **Layer 1** | **Craft Reference Knowledge** | 50K – 100K | Teaches authentic craft aesthetics & motifs | Craft imagery + detailed VLM craft captions (`chikankari`, `phulkari`, `kalamkari`, `ajrakh`, `bandhani`, `kantha`, `paithani`, `ikat`, `madhubani`, `warli`). |
| **Layer 2** | **Garment Application Pairs** | 20K – 50K pairs | Teaches regional garment editing & inpainting | `Before Image` (Plain garment) + `Instruction / Mask` + `After Image` (Garment with craft applied). |
| **Layer 3** | **Design Details & Macro Views** | 10K – 30K | Teaches macro texture & pattern repeatability | Close-ups of collars, cuffs, sleeves, necklines, hemlines, embroidery stitches, and repeat textile prints. |

---

## 🏗️ Model Architecture & Multi-Stage Training Strategy

```
               Pretrained Base Diffusion (SDXL / DiT)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   Stage 1: Craft Knowledge LoRA                    Stage 2: Inpainting & Regional Editing
  (Trained on Layer 1 & Layer 3)                   (Trained on Layer 2 Application Pairs)
         │                                                 │
  Generates authentic motifs                     Applies craft while preserving
   & multi-view craft designs                      original garment silhouette
```

### Training Pipeline Stages:
1. **Stage 1 — Craft Specialization (Layer 1 + Layer 3)**:
   - Freeze VAE & Text Encoders; train UNet LoRA on craft reference and macro detail datasets.
2. **Stage 2 — Design Application & Inpainting (Layer 2)**:
   - Train SDXL Inpainting / Instruct-editing LoRA on Before/After pairs + region masks to enforce garment preservation and targeted placement.
3. **Stage 3 — Multi-View & Pattern Synthesis**:
   - Fine-tune multi-view generation triggers (`[front view]`, `[detail view]`, `[pattern view]`, `[flat view]`).

---

## 🎓 Student Interface Modes

1. **Mode 1: Create Design**: Text prompt -> 4-8 multi-view design variations.
2. **Mode 2: Apply Craft**: Upload blank garment -> Select craft (`Chikankari`, `Kalamkari`, etc.) -> Select region (`Collar`, `Sleeves`, `Chest`, `Hemline`) -> Inpaint.
3. **Mode 3: Free Instruction**: Upload garment -> Natural language prompt (*"Apply a contemporary Ajrakh-inspired geometric pattern only to the sleeves, keeping the shirt white"*).
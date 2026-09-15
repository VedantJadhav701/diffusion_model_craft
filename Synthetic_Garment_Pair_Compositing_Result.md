# 🖼️ Synthetic Garment Pair Compositing Execution Result

This document presents the visual output of the **Synthetic Pair Compositing Pipeline** executed on local IndoFashion human apparel photos and authentic IndusCraft craft texture references.

---

## 📊 Visual 3-Panel Output Comparison

![Synthetic Pair Output](file:///C:/Users/HP/.gemini/antigravity-cli/brain/4e475b64-f29b-4602-b396-4a8019b43049/synthetic_pair_output.png)

---

## ⚙️ Execution & Data Flow Summary

1. **Human Photo (Input)**: Loaded from IndoFashion training split ([`c:\Users\HP\projects\diffusion_model\indofashion_dataset\images\train\0.jpeg`](file:///c:/Users/HP/projects/diffusion_model/indofashion_dataset/images/train/0.jpeg)) and resized to $1024 \times 1024$.
2. **Craft Texture (Reference)**: Extracted from IndusCraft Layer 1 craft reference dataset ([`c:\Users\HP\projects\diffusion_model\data\images\chikankari_web_000000.jpg`](file:///c:/Users/HP/projects/diffusion_model/data/images/chikankari_web_000000.jpg)) as an authentic Chikankari embroidery patch resized to $240 \times 200$.
3. **Garment Mask Region**: Defined a bounding box rectangular mask `[420, 220, 660, 420]` covering the chest/garment region of the model.
4. **Synthetic Pair (Target)**: Composited the Chikankari embroidery patch seamlessly into the targeted garment region, producing a paired training sample for regional inpainting SDXL LoRA fine-tuning.

import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# 1. Load real human-wearing photo from IndoFashion
INDOFASHION_DIR = r"c:\Users\HP\projects\diffusion_model\indofashion_dataset"
human_img_path = os.path.join(INDOFASHION_DIR, "images", "train", "0.jpeg")

if not os.path.exists(human_img_path):
    # Fallback to any jpeg in train folder
    train_dir = Path(INDOFASHION_DIR) / "images" / "train"
    for img_p in train_dir.glob("*.*"):
        human_img_path = str(img_p)
        break

print(f"Loading human photo from: {human_img_path}")
human_img = Image.open(human_img_path).convert("RGB").resize((1024, 1024))

# 2. Load real craft texture reference from IndusCraft
craft_img_path = r"c:\Users\HP\projects\diffusion_model\data\images\chikankari_web_000000.jpg"
if not os.path.exists(craft_img_path):
    base_craft = Path(r"c:\Users\HP\projects\diffusion_model\data\images")
    for f in base_craft.glob("*.jpg"):
        craft_img_path = str(f)
        break

print(f"Loading craft texture reference from: {craft_img_path}")
craft_full = Image.open(craft_img_path).convert("RGB")
craft_sample = craft_full.resize((240, 200))

# 3. Define mask region on human photo
mask = Image.new("L", (1024, 1024), 0)
draw = ImageDraw.Draw(mask)
draw.rectangle([420, 220, 660, 420], fill=255)

# 4. Alpha-blend / paste craft texture into masked region
composited = human_img.copy()
craft_patch = craft_sample.resize((660-420, 420-220))
composited.paste(craft_patch, (420, 220))

# Save plot output
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(human_img)
axes[0].set_title("Human photo (input)", fontsize=14, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(craft_sample)
axes[1].set_title("Craft texture (reference)", fontsize=14, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(composited)
axes[2].set_title("Synthetic pair (target)", fontsize=14, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()

art_dir = Path(r"C:\Users\HP\.gemini\antigravity-cli\brain\4e475b64-f29b-4602-b396-4a8019b43049")
art_dir.mkdir(parents=True, exist_ok=True)
out_plot_path = art_dir / "synthetic_pair_output.png"
plt.savefig(out_plot_path, bbox_inches="tight", dpi=150)
print(f"Successfully generated and saved figure to: {out_plot_path}")

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

CRAFT_IMG_DIR = r"c:\Users\HP\projects\diffusion_model\data\images"

# 1. Inspect exact target file
bad_path = os.path.join(CRAFT_IMG_DIR, "chikankari_web_000000.jpg")
if os.path.exists(bad_path):
    bad_img = Image.open(bad_path)
    print(f"File: chikankari_web_000000.jpg | Size: {bad_img.size} | Mode: {bad_img.mode}")
else:
    print(f"File {bad_path} not found.")

# 2. List first 20 chikankari files
all_files = sorted([f for f in os.listdir(CRAFT_IMG_DIR) if f.startswith("chikankari_web_")])
chikankari_files = all_files[:20]
print(f"\nFound {len(all_files)} total chikankari files. First 20:")
for f in chikankari_files:
    print(" -", f)

# 3. Render 4x5 visual grid check plot
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for i, f in enumerate(chikankari_files):
    img_path = os.path.join(CRAFT_IMG_DIR, f)
    img = Image.open(img_path)
    ax = axes[i // 5][i % 5]
    ax.imshow(img)
    ax.set_title(f"{f}\n({img.size[0]}x{img.size[1]})", fontsize=9)
    ax.axis("off")

# Hide any empty subplot axes if len < 20
for j in range(len(chikankari_files), 20):
    axes[j // 5][j % 5].axis("off")

plt.tight_layout()

art_dir = Path(r"C:\Users\HP\.gemini\antigravity-cli\brain\4e475b64-f29b-4602-b396-4a8019b43049")
art_dir.mkdir(parents=True, exist_ok=True)
out_plot_path = art_dir / "chikankari_grid_check.png"
plt.savefig(out_plot_path, bbox_inches="tight", dpi=150)
print(f"\nSuccessfully generated and saved 20-image grid plot to: {out_plot_path}")

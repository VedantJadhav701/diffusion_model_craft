import os
import torch
import streamlit as st
from PIL import Image, ImageDraw

# Load .env file automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import login
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline

# Set Page Config
st.set_page_config(
    page_title="IndusCraft — Fashion & Craft AI Engine",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🎨 IndusCraft AI — Fashion & Textile Design Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pearl Academy AI Fashion Design System: Multi-View Concept Generation & Regional Garment Inpainting</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Model & Generation Controls")

# Read token from .env or OS environment variable
DEFAULT_TOKEN = os.environ.get("HF_TOKEN", "")
HF_TOKEN = st.sidebar.text_input("Hugging Face API Token", type="password", value=DEFAULT_TOKEN)

if HF_TOKEN.strip():
    try:
        login(token=HF_TOKEN.strip())
    except Exception:
        pass

MODEL_SOURCE = st.sidebar.selectbox(
    "Select Workflow A Model Engine",
    [
        "HF Merged Engine Stage 1 (vedantjadhav701/induscraft-sdxl-merged)",
        "Base SDXL + HF Stage 1 LoRA (vedantjadhav701/induscraft-stage1-craft-lora)",
        "Base SDXL + Local Stage 1 LoRA Checkpoint"
    ],
    index=0
)

INPAINT_MODEL_SOURCE = st.sidebar.selectbox(
    "Select Workflow B Inpaint Engine",
    [
        "Base SDXL Inpaint + HF Stage 2 LoRA (vedantjadhav701/induscraft-stage2-inpainting-lora)",
        "HF Merged Engine Stage 2 (vedantjadhav701/induscraft-sdxl-inpaint-merged)",
        "Base SDXL Inpaint + Local Inpaint LoRA Checkpoint"
    ],
    index=0
)

NUM_INFERENCE_STEPS = st.sidebar.slider("Inference Steps", min_value=15, max_value=50, value=30, step=5)
GUIDANCE_SCALE = st.sidebar.slider("Guidance Scale (CFG)", min_value=3.0, max_value=15.0, value=7.5, step=0.5)
IMAGE_RESOLUTION = st.sidebar.selectbox("Resolution", [768, 1024], index=0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

st.sidebar.markdown("---")
st.sidebar.markdown(f"**PyTorch Device**: `{DEVICE.upper()}`")
st.sidebar.markdown(f"**Precision**: `{DTYPE}`")

# Cache Model Loaders
@st.cache_resource
def load_create_pipeline(source_option: str, token: str):
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    merged_stage1 = "vedantjadhav701/induscraft-sdxl-merged"
    hf_lora_stage1 = "vedantjadhav701/induscraft-stage1-craft-lora"
    auth_token = token.strip() if token.strip() else None

    # Option 1: Merged Stage 1
    if "Merged" in source_option:
        try:
            st.info(f"Loading merged Stage 1 pipeline from Hugging Face: {merged_stage1}...")
            pipe = StableDiffusionXLPipeline.from_pretrained(merged_stage1, torch_dtype=DTYPE, token=auth_token, local_files_only=False).to(DEVICE)
            return pipe
        except Exception as e:
            st.warning(f"Could not load online merged model ({e}). Trying Base SDXL + HF Adapter...")

    # Option 2: HF Adapter
    if "HF Stage 1" in source_option or "Merged" in source_option:
        try:
            st.info(f"Loading Base SDXL + HF LoRA Adapter: {hf_lora_stage1}...")
            pipe = StableDiffusionXLPipeline.from_pretrained(base_id, torch_dtype=DTYPE, token=auth_token, local_files_only=False).to(DEVICE)
            pipe.load_lora_weights(hf_lora_stage1, token=auth_token)
            return pipe
        except Exception as e:
            st.warning(f"Could not load online HF adapter ({e}). Checking local checkpoints...")

    # Option 3: Local Fallback
    st.info("Loading Base SDXL Base 1.0 + Local Stage 1 LoRA...")
    pipe = StableDiffusionXLPipeline.from_pretrained(base_id, torch_dtype=DTYPE, token=auth_token, local_files_only=False).to(DEVICE)
    possible_paths = [
        "./induscraft_stage1_craft_lora/induscraft_stage1_final",
        "./outputs/induscraft_sdxl_lora_final"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            pipe.load_lora_weights(p)
            st.success(f"Loaded local Stage 1 LoRA from '{p}'!")
            break
    return pipe

@st.cache_resource
def load_inpaint_pipeline(source_option: str, token: str):
    base_inpaint = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    merged_stage2 = "vedantjadhav701/induscraft-sdxl-inpaint-merged"
    hf_lora_stage2 = "vedantjadhav701/induscraft-stage2-inpainting-lora"
    auth_token = token.strip() if token.strip() else None

    # Option 1: HF Stage 2 Adapter
    if "HF Stage 2" in source_option:
        try:
            st.info(f"Loading Base Inpaint + HF Stage 2 LoRA Adapter: {hf_lora_stage2}...")
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base_inpaint, torch_dtype=DTYPE, token=auth_token, local_files_only=False).to(DEVICE)
            pipe.load_lora_weights(hf_lora_stage2, token=auth_token)
            return pipe
        except Exception as e:
            st.warning(f"Could not load online HF inpaint adapter ({e}). Falling back to local checkpoints...")

    # Option 2: Merged Stage 2
    if "Merged" in source_option:
        try:
            st.info(f"Loading merged Stage 2 Inpainting pipeline: {merged_stage2}...")
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(merged_stage2, torch_dtype=DTYPE, token=auth_token, local_files_only=False).to(DEVICE)
            return pipe
        except Exception as e:
            st.warning(f"Could not load online merged inpaint model ({e}). Falling back to local checkpoints...")

    # Option 3: Local Fallback
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base_inpaint, torch_dtype=DTYPE, token=auth_token, local_files_only=False).to(DEVICE)
    possible_inpaint_paths = [
        "./induscraft_stage2_inpainting_lora/induscraft_stage2_final",
        "./outputs/induscraft_inpainting_lora_final"
    ]
    for p in possible_inpaint_paths:
        if os.path.exists(p):
            pipe.load_lora_weights(p)
            st.success(f"Loaded local Stage 2 Inpainting LoRA from '{p}'!")
            break
    return pipe

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🎨 Workflow A: Create New Design (Multi-View)",
    "🧥 Workflow B: Apply Craft to Garment (Regional Inpainting)",
    "📊 Evaluation & Model Details"
])

# ----------------------------------------------------
# TAB 1: WORKFLOW A — CREATE NEW DESIGN
# ----------------------------------------------------
with tab1:
    st.subheader("Explore Multi-View Fashion Design Concepts")
    st.caption("Generates front view, macro detail view, seamless repeat pattern view, or technical flat view.")

    col1, col2 = st.columns([1, 1])

    with col1:
        CRAFT_STYLE = st.selectbox(
            "Select Indian Craft Category",
            [
                "Chikankari (Lucknow Shadow Work Embroidery)",
                "Phulkari (Punjab Silk Thread Geometry)",
                "Kalamkari (Andhra Organic Dyes & Narrative Art)",
                "Ajrakh (Kutch Indigo/Madder Block Print)",
                "Bandhani (Rajasthan/Gujarat Tie-Dye Dots)",
                "Kantha (Bengal Running Stitch Embroidery)",
                "Paithani (Maharashtra Silk & Gold Zari Peacock)",
                "Ikat (Telangana/Odisha Geometric Weave)",
                "Madhubani (Mithila Folk Line Painting)",
                "Warli (Maharashtra Tribal Figure Art)"
            ]
        )

        VIEW_MODE = st.radio(
            "Select View Mode",
            ["[front view]", "[detail view]", "[pattern view]", "[flat garment]"],
            horizontal=True
        )

        craft_key = CRAFT_STYLE.split(" (")[0].lower()
        default_prompt = f"A high resolution detailed fashion design photograph, {VIEW_MODE}, of authentic {craft_key} embroidery style on a modern kurta silhouette."

        PROMPT = st.text_area("Prompt", value=default_prompt, height=100)

        GENERATE_BTN = st.button("✨ Generate Design Concept", type="primary", use_container_width=True)

    with col2:
        if GENERATE_BTN:
            with st.spinner("Generating fashion design concept..."):
                try:
                    pipe_create = load_create_pipeline(MODEL_SOURCE, HF_TOKEN)
                    result_img = pipe_create(
                        prompt=PROMPT,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE
                    ).images[0]

                    st.image(result_img, caption=f"Generated {VIEW_MODE} — {craft_key.title()}", use_column_width=True)

                    # Download Button
                    os.makedirs("./output_images", exist_ok=True)
                    img_bytes = os.path.join("./output_images", "generated_design.png")
                    result_img.save(img_bytes)

                    with open(img_bytes, "rb") as file:
                        st.download_button(
                            label="📥 Download High-Res Design",
                            data=file,
                            file_name="induscraft_design.png",
                            mime="image/png",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Generation error: {e}")
        else:
            st.info("Click 'Generate Design Concept' to synthesize fashion designs.")

# ----------------------------------------------------
# TAB 2: WORKFLOW B — REGIONAL INPAINTING
# ----------------------------------------------------
with tab2:
    st.subheader("Apply Craft to Existing Student Garment")
    st.caption("Applies craft embroidery/print to selected region while strictly preserving original garment silhouette.")

    col1, col2 = st.columns([1, 1])

    with col1:
        UPLOADED_FILE = st.file_uploader("Upload Blank/Existing Garment Image", type=["png", "jpg", "jpeg"])

        TARGET_REGION = st.selectbox(
            "Targeted Garment Region",
            ["Collar / Neckline", "Sleeves & Cuffs", "Chest / Bodice", "Hemline Border"]
        )

        INPAINT_CRAFT = st.selectbox(
            "Target Craft Style",
            ["Chikankari", "Kalamkari", "Phulkari", "Ajrakh", "Bandhani", "Kantha", "Paithani", "Ikat"]
        )

        INPAINT_PROMPT = st.text_input(
            "Inpainting Instruction",
            value=f"Apply authentic {INPAINT_CRAFT.lower()} embroidery on the {TARGET_REGION.lower()} of the garment."
        )

        APPLY_BTN = st.button("🪄 Apply Regional Craft", type="primary", use_container_width=True)

    with col2:
        if UPLOADED_FILE is not None:
            input_image = Image.open(UPLOADED_FILE).convert("RGB").resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION))
            st.image(input_image, caption="Original Student Garment", use_column_width=True)

            if APPLY_BTN:
                with st.spinner("Inpainting regional craft onto garment..."):
                    try:
                        pipe_inpaint = load_inpaint_pipeline(INPAINT_MODEL_SOURCE, HF_TOKEN)

                        # Synthetic Region Mask Generation
                        mask = Image.new("L", (IMAGE_RESOLUTION, IMAGE_RESOLUTION), 0)
                        draw = ImageDraw.Draw(mask)

                        if "Collar" in TARGET_REGION:
                            draw.polygon([(IMAGE_RESOLUTION*0.3, IMAGE_RESOLUTION*0.1), (IMAGE_RESOLUTION*0.7, IMAGE_RESOLUTION*0.1), (IMAGE_RESOLUTION*0.6, IMAGE_RESOLUTION*0.35), (IMAGE_RESOLUTION*0.4, IMAGE_RESOLUTION*0.35)], fill=255)
                        elif "Sleeves" in TARGET_REGION:
                            draw.rectangle([0, int(IMAGE_RESOLUTION*0.2), int(IMAGE_RESOLUTION*0.25), int(IMAGE_RESOLUTION*0.8)], fill=255)
                            draw.rectangle([int(IMAGE_RESOLUTION*0.75), int(IMAGE_RESOLUTION*0.2), IMAGE_RESOLUTION, int(IMAGE_RESOLUTION*0.8)], fill=255)
                        elif "Chest" in TARGET_REGION:
                            draw.rectangle([int(IMAGE_RESOLUTION*0.25), int(IMAGE_RESOLUTION*0.2), int(IMAGE_RESOLUTION*0.75), int(IMAGE_RESOLUTION*0.55)], fill=255)
                        else:
                            draw.rectangle([int(IMAGE_RESOLUTION*0.15), int(IMAGE_RESOLUTION*0.75), int(IMAGE_RESOLUTION*0.85), IMAGE_RESOLUTION], fill=255)

                        edited_img = pipe_inpaint(
                            prompt=INPAINT_PROMPT,
                            image=input_image,
                            mask_image=mask,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            guidance_scale=GUIDANCE_SCALE
                        ).images[0]

                        st.image(edited_img, caption=f"Craft Applied: {INPAINT_CRAFT} on {TARGET_REGION}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Inpainting error: {e}")
        else:
            st.info("Upload a garment image to test regional inpainting.")

# ----------------------------------------------------
# TAB 3: EVALUATION & MODEL DETAILS
# ----------------------------------------------------
with tab3:
    st.subheader("📊 Model Evaluation & Dataset Architecture")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Stage 1 Loss (15 Epochs)", "0.0463", delta="-0.2579")
    col_m2.metric("Stage 2 Inpaint Loss", "0.0177", delta="-0.0212")
    col_m3.metric("Average CLIP Score", "0.3128", delta="+0.045")

    st.markdown("---")
    st.markdown("""
    ### 🧵 Official Published Hugging Face Resources

    - 📊 **Multimodal Dataset**: [vedantjadhav701/induscraft-dataset](https://huggingface.co/datasets/vedantjadhav701/induscraft-dataset)
    - ⚡ **Stage 1 Craft Knowledge LoRA Adapter**: [vedantjadhav701/induscraft-stage1-craft-lora](https://huggingface.co/vedantjadhav701/induscraft-stage1-craft-lora)
    - ⚡ **Stage 2 Inpainting LoRA Adapter**: [vedantjadhav701/induscraft-stage2-inpainting-lora](https://huggingface.co/vedantjadhav701/induscraft-stage2-inpainting-lora)
    - 🎨 **Stage 1 Merged SDXL Engine**: [vedantjadhav701/induscraft-sdxl-merged](https://huggingface.co/vedantjadhav701/induscraft-sdxl-merged)
    - 🧥 **Stage 2 Merged SDXL Inpaint Engine**: [vedantjadhav701/induscraft-sdxl-inpaint-merged](https://huggingface.co/vedantjadhav701/induscraft-sdxl-inpaint-merged)
    - 💻 **GitHub Repository**: [VedantJadhav701/diffusion_model_craft](https://github.com/VedantJadhav701/diffusion_model_craft)

    ### 📐 Multi-View Tagging Schema
    - `[front view]`: Full garment composition on mannequin or model.
    - `[detail view]`: Macro texture close-up of embroidery stitch work & weave motifs.
    - `[pattern view]`: Seamless repeat layout for textile manufacturing.
    - `[flat garment]`: Technical flat drawing for pattern making.
    """)

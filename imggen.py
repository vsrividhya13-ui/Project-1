import streamlit as st
import torch
from diffusers import DiffusionPipeline

# Page settings
st.set_page_config(page_title="AI Image Generator", page_icon="🎨")

# Title
st.title("🎨 AI Image Generation")
st.write("Enter a prompt and generate an image using AI.")

# Load the AI model
@st.cache_resource
def load_model():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    )

    if device == "cpu":
        pipe.enable_attention_slicing()

    pipe = pipe.to(device)
    return pipe

# Prompt input
prompt = st.text_input(
    "✍️ Enter your prompt:",
    "A superhero sitting in the Himalayas"
)

# Image settings
num_images = st.slider("Number of images", 1, 3, 1)
height = st.slider("Image height", 256, 512, 512, step=64)
width = st.slider("Image width", 256, 512, 512, step=64)

# Generate button
if st.button("🎨 Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a valid prompt!")
    else:
        with st.spinner("🤖 Generating image... Please wait"):
            pipe = load_model()

            result = pipe(
                prompt,
                num_images_per_prompt=num_images,
                height=height,
                width=width
            )

            st.subheader("✨ Generated Image(s)")

            for img in result.images:
                st.image(img, use_container_width=True)

import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍"
)

st.title("🔍 AI Object Detection")
st.write("Upload an image and AI will detect objects with bounding boxes.")

@st.cache_resource
def load_model():
    return pipeline(
        "object-detection",
        model="hustvl/yolos-tiny"
    )

image = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if image:

    img = Image.open(image).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Objects"):

        with st.spinner("🤖 Detecting objects..."):

            detector = load_model()
            results = detector(img)

        # Create a copy to draw boxes
        detected_img = img.copy()
        draw = ImageDraw.Draw(detected_img)

        # Draw bounding boxes around detected objects
        for result in results:
            box = result["box"]

            xmin = box["xmin"]
            ymin = box["ymin"]
            xmax = box["xmax"]
            ymax = box["ymax"]

            label = result["label"]
            score = result["score"]

            # Draw rectangle
            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                outline="red",
                width=3
            )

            # Add object name and confidence
            text = f"{label} {score:.0%}"
            draw.text(
                (xmin, max(0, ymin - 15)),
                text,
                fill="red"
            )

        # Display the same uploaded image with boxes
        st.subheader("🎯 Detected Objects")
        st.image(
            detected_img,
            caption="Objects Detected",
            use_container_width=True
        )

        # Display object details
        for result in results:
            st.write(
                f"**{result['label']}** - "
                f"Confidence: {result['score']:.2%}"
            )

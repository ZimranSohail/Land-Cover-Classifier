
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

from inference import load_model, predict, CONFIG


# ==========================================
# Page setup
# ==========================================
st.set_page_config(
    page_title="Satellite Land Cover Classifier",
    page_icon="🛰️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_images"

CLASS_INFO = {
    "Bare_Sparse_Vegetation": "Bare soil and sparsely vegetated land.",
    "Built_Up": "Urban areas, buildings, and developed surfaces.",
    "Cropland": "Land used for growing crops.",
    "Permanent_Water": "Rivers, lakes, reservoirs, and other permanent water.",
    "Tree_Cover": "Areas predominantly covered by trees.",
}


# ==========================================
# Load model once and reuse it
# ==========================================
@st.cache_resource
def get_model():
    return load_model()


# ==========================================
# Header
# ==========================================
st.title("🛰️ Satellite Land Cover Classifier")

st.write(
    "Explore satellite imagery using a fine-tuned ResNet18 "
    "deep-learning model. Upload an image or try one of the "
    "provided examples."
)

st.caption(
    "Five classes · Sentinel-2 RGB training imagery · "
    "128 × 128 model input"
)

with st.expander("About this project"):
    st.write(
        "This project classifies satellite image patches into "
        "five land-cover categories. The model was fine-tuned "
        "from ImageNet-pretrained ResNet18 weights."
    )
    st.write(
        "The reported test accuracy was 97.00% on 367 held-out "
        "test images. This score measures performance on that "
        "dataset and does not guarantee the same accuracy on "
        "arbitrary uploaded images."
    )


# ==========================================
# Input selection
# ==========================================
st.subheader("1. Choose an image")

input_mode = st.radio(
    "Image source",
    ["Try a sample image", "Upload your own"],
    horizontal=True,
)

selected_image = None
selected_name = None

if input_mode == "Try a sample image":
    sample_files = (
        sorted(SAMPLE_DIR.glob("*.png"))
        + sorted(SAMPLE_DIR.glob("*.jpg"))
        + sorted(SAMPLE_DIR.glob("*.jpeg"))
    ) if SAMPLE_DIR.exists() else []

    if not sample_files:
        st.warning(
            "No sample images found. Add PNG or JPG images "
            "to the sample_images folder beside app.py, "
            "or choose Upload your own."
        )
    else:
        sample_names = [p.name for p in sample_files]

        selected_name = st.selectbox(
            "Select a sample",
            sample_names,
        )

        selected_path = SAMPLE_DIR / selected_name
        selected_image = Image.open(selected_path).convert("RGB")

else:
    uploaded_file = st.file_uploader(
        "Upload a satellite image",
        type=["png", "jpg", "jpeg"],
        help=(
            "RGB images work best when their visual appearance "
            "matches the Sentinel-2 imagery used for training."
        ),
    )

    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file).convert("RGB")
        selected_name = uploaded_file.name


# ==========================================
# Display image and predict
# ==========================================
if selected_image is not None:
    st.divider()
    st.subheader("2. Image preview")

    left, right = st.columns([1, 1])

    with left:
        st.image(
            selected_image,
            caption=selected_name or "Uploaded image",
            use_container_width=True,
        )
        st.caption(
            f"Original dimensions: "
            f"{selected_image.width} × {selected_image.height} pixels"
        )

    with right:
        st.subheader("Prediction")

        try:
            with st.spinner("Analyzing image..."):
                model = get_model()
                result = predict(selected_image, model)

            predicted_class = result["class_name"]
            confidence = result["confidence"]

            st.metric(
                "Predicted land-cover class",
                predicted_class.replace("_", " "),
            )

            st.metric(
                "Model confidence",
                f"{confidence:.2%}",
            )

            st.write(CLASS_INFO.get(predicted_class, ""))

            # Calculate all class probabilities for display.
            # Uses the same preprocessing as inference.py.
            import torch
            from inference import preprocess, DEVICE

            tensor = preprocess(
                selected_image.convert("RGB")
            ).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                probabilities = torch.softmax(
                    model(tensor), dim=1
                )[0].cpu().tolist()

            probability_df = pd.DataFrame({
                "Land-cover class": [
                    name.replace("_", " ")
                    for name in CONFIG["class_names"]
                ],
                "Probability": probabilities,
            }).sort_values("Probability", ascending=False)

            st.subheader("Class probability breakdown")

            st.bar_chart(
                probability_df.set_index("Land-cover class")
            )

            st.caption(
                "Probabilities are model outputs, not calibrated "
                "guarantees that a prediction is correct."
            )

        except Exception as exc:
            st.error(
                "The model could not process this image. "
                "Check that the model files and dependencies "
                "are installed correctly."
            )
            st.exception(exc)


# ==========================================
# Input guidance
# ==========================================
st.divider()

with st.expander("What images should I use?"):
    st.markdown(
        """
        - Use RGB satellite imagery resembling the training data.
        - Sentinel-2-style imagery is preferable.
        - Google Earth screenshots may include labels, borders,
          different dates, or different image processing.
        - Ordinary photographs are outside the model's intended use.
        - A confident prediction can still be incorrect.
        """
    )

st.caption(
    "Educational demonstration · ResNet18 · "
    "Satellite-based land-cover classification"
)

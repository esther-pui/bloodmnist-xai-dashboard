import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="BloodMNIST XAI Dashboard", layout="wide")

# -----------------------------
# CSS styling
# -----------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #fafcfc;
        font-family: 'Montserrat', sans-serif;
    }

    /* Global Text Color */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label, .stApp li {
        color: #383c43 !important;
    }

    /* Target the Streamlit container to act as your card */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1px solid #e6ecec !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        border-radius: 12px;
        padding: 10px;
    }

    /* Fix table text color */
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #383c43 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Blood Cell Classification with Explainable AI 🧬✨")

st.markdown(
    "This dashboard demonstrates how AI predictions on blood cell images can be explained "
    "using visual (Grad-CAM) and counterfactual (DiCE) explanations."
)

# -----------------------------
# Columns
# -----------------------------
col1, col2, col3 = st.columns(3)

mock_classes = [
    "Neutrophil", "Eosinophil", "Basophil", "Lymphocyte",
    "Monocyte", "Immature Granulocyte", "Red Blood Cell", "Platelet"
]

# -----------------------------
# Column 1: Image Upload & Prediction
# -----------------------------
with col1:
    with st.container(border=True): # Native container handles background correctly
        st.subheader("1️⃣ Upload & Prediction 🩸")
        uploaded_file = st.file_uploader("Upload a blood cell microscopy image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Blood Cell Image", use_container_width=True)

            pred_class = np.random.choice(mock_classes)
            pred_prob = np.random.uniform(0.7, 0.99)
            st.markdown(f"### Predicted Cell Type: **{pred_class}**")
            st.progress(pred_prob, text=f"Model Confidence: {pred_prob:.2%}")
            st.info("The model predicts the most likely blood cell type based on learned morphological patterns.")
        else:
            st.info("Please upload an image to start the analysis.")

# -----------------------------
# Column 2: Grad-CAM Visualization
# -----------------------------
with col2:
    with st.container(border=True):
        st.subheader("2️⃣ Grad-CAM: Visual Explanation 🔥")

        if uploaded_file is not None:
            cam_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            st.image(cam_image, caption="Grad-CAM heatmap overlay (Mock)", use_container_width=True)

            st.markdown(
                """
                **How to interpret this image:**
                - Warmer colors (red/yellow) highlight regions that most influenced the prediction.
                - These regions often correspond to important morphological structures.
                """
            )
        else:
            st.info("Upload an image to view the Grad-CAM explanation.")

# -----------------------------
# Column 3: Counterfactuals (DiCE) & Dropdown
# -----------------------------
with col3:
    with st.container(border=True):
        st.subheader("3️⃣ Counterfactual Analysis ⚡")

        if uploaded_file is not None:
            n_cfs = 2
            cf_target_classes = np.random.choice(mock_classes, n_cfs, replace=False)
            change_magnitudes = np.round(np.random.uniform(0.5, 2.5, n_cfs), 2)
            cf_summary_df = pd.DataFrame({
                "Alternative Cell Type": cf_target_classes,
                "Change Magnitude": change_magnitudes
            }).sort_values("Change Magnitude")

            st.table(cf_summary_df)

            closest_cf = cf_summary_df.iloc[0]
            st.warning(f"Similarity: **{closest_cf['Alternative Cell Type']}** is the closest alternative.")

            # Restored the dropdown (Expander)
            with st.expander("Show technical details (embedding-level features)"):
                n_features = 10
                cf_data = np.random.randn(n_cfs, n_features)
                cf_df = pd.DataFrame(cf_data, columns=[f"Embedding_{i}" for i in range(n_features)])
                st.dataframe(cf_df)
        else:
            st.info("Upload an image to generate counterfactual explanations.")
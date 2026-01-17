import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import torch

# Import your real model functions
from model_utils import (
    model, pca, pca_classifier, wrapped_model, preprocess_image, 
    get_gradcam_image, get_counterfactuals_single, label_names, device
)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="BloodMNIST XAI Dashboard", layout="wide")

def st_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -----------------------------
# CSS styling (Preserved exactly)
# -----------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #fafcfc; font-family: 'Montserrat', sans-serif; }
        h1 { font-weight: 400 !important; color: #383c43 !important; }
        .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label, .stApp li { color: #383c43 !important; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff;
            border: 1px solid #e6ecec !important;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
            border-radius: 12px;
            padding: 10px;
        }
        [data-testid="stFileUploader"] section {
            background-color: #f0f4f4 !important;
            border: 1px dashed #c0cccc !important;
        }
        [data-testid="stFileUploader"] button[kind="secondary"] {
            background-color: transparent !important;
            border: 1px solid #c0cccc !important;
            color: #383c43 !important;
            box-shadow: none !important;
        }
        [data-testid="stUploadedFile"] {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        [data-testid="stUploadedFile"] button {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #383c43 !important;
        }
        [data-testid="stFileUploaderFileName"], 
        [data-testid="stFileUploaderFileData"] small {
            color: #383c43 !important;
        }
        div[data-testid="stProgress"] > div > div > div > div {
            background-color: #9babfe !important;
        }
        div[role="progressbar"] > div > div {
            background-color: #cccdda !important;
        }
        [data-testid="stNotification"] {
            background-color: #dbdef4 !important;
            color: #383c43 !important;
            border: none !important;
        }
        [data-testid="stNotification"] svg {
            fill: #9babfe !important;
        }
        [data-testid="stNotification"], 
        [data-testid="stAlert"], 
        [data-testid="stAlert"] > div,
        .stAlert {
            background-color: #dbdef4 !important;
            color: #383c43 !important;
            border: none !important;
            border-radius: 15px !important;
        }
        [data-testid="stNotification"] p, 
        [data-testid="stAlert"] p,
        [data-testid="stNotificationContent"] {
            color: #383c43 !important;
        }
        [data-testid="stNotification"] svg, 
        [data-testid="stAlert"] svg {
            fill: #383c43 !important;
        }
        header {
            opacity: 0;
            transform: translateY(-50px);
            transition: all 0.3s ease-in-out;
            height: 50px !important;
        }
        header:hover {
            opacity: 1;
            transform: translateY(0);
            background-color: rgba(255, 255, 255, 0.9) !important;
        }
        .block-container {
            padding-top: 2rem !important;
            margin-top: -3rem !important;
        }
        [data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"] img {
            max-height: 300px !important;
            width: auto !important;
            object-fit: contain !important;
        }
        .image-frame {
            max-height: 340px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        div[data-testid="stVerticalBlock"] > div:has(div[class*="img-frame"]) {
            background-color: #f8f9fb !important;
            border-radius: 15px !important;
            padding: 8px !important;
            display: flex !important;
            justify-content: center !important;
        }
        table {
            border-collapse: collapse !important;
            width: 100% !important;
            border: 1px solid #dde0f2 !important;
        }
        th {
            background-color: #f8f9fb !important;
            color: #383c43 !important;
            border: 1px solid #dde0f2 !important;
            padding: 10px !important;
            text-align: left !important;
        }
        td {
            border: 1px solid #dde0f2 !important;
            padding: 10px !important;
            color: #383c43 !important;
        }
        [data-testid="stExpander"] details:hover {
            background-color: transparent !important;
        }
        [data-testid="stExpander"] summary:hover {
            background-color: transparent !important;
            color: inherit !important;
        }
        [data-testid="stExpander"] summary:hover span {
            color: #383c43 !important;
        }
        [data-testid="stTable"] {
            background-color: #ffffff !important;
        }
        div[data-testid="stDataFrame"] > div {
            background-color: #ffffff !important;
        }
        [data-testid="stDataFrame"] [role="columnheader"] {
            background-color: #f8f9fb !important;
            color: #383c43 !important;
            font-weight: bold !important;
        }
        [data-testid="stDataFrame"] [role="gridcell"] {
            background-color: #ffffff !important;
            color: #383c43 !important;
            border: 1px solid #f0f2f6 !important;
        }
        [data-testid="stExpander"] div.stHtml {
            overflow-x: auto;
            background-color: #ffffff !important;
        }
        [data-testid="stExpander"] table {
            background-color: #ffffff !important;
            font-size: 0.8rem !important;
        }
        [data-testid="stColumn"] > div {
            background-color: #ffffff !important;
            border-radius: 12px !important;
            padding: 15px !important;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            background-color: transparent !important;
        }
        div[data-testid="stTable"] table tr th:first-child, 
        div[data-testid="stTable"] table tr td:first-child {
            display: none !important;
        }
        [data-testid="stExpander"] details summary:hover {
            background-color: transparent !important;
            color: inherit !important;
        }
        [data-testid="stExpander"] details summary:hover span {
            color: #383c43 !important;
        }
        [data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: none !important;
        }
        [data-testid="stExpander"] details {
            border: 1px solid #dbe9c7 !important;
            border-radius: 8px;
        }
        [data-testid="stExpander"] table td {
            background-color: #ffffff !important;
        }
        [data-testid="stExpander"] details summary {
            background-color: #dbe9c7 !important;
            color: #383c43 !important;
        }
        [data-testid="stExpander"] details summary:focus, 
        [data-testid="stExpander"] details summary:active,
        [data-testid="stExpander"] details[open] summary {
            background-color: #dbe9c7 !important;
            color: #383c43 !important;
            outline: none !important;
            box-shadow: none !important;
        }
        [data-testid="stExpander"] details summary p {
            color: #383c43 !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid #dbe9c7 !important;
            border-radius: 8px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Blood Cell Classification with Explainable AI 🧬")
st.markdown("This dashboard explains AI blood cell predictions using Grad-CAM (visual) and DiCE (counterfactual) explanations.")

# -----------------------------
# Columns
# -----------------------------
col1, col2, col3 = st.columns(3)

# -----------------------------
# Column 1: Image Upload & Prediction
# -----------------------------
with col1:
    with st.container(border=True): 
        st.subheader("1 Upload & Prediction")
        uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            st.markdown(
                f"""
                <div style="background-color: #f8f9fb; height: 250px; border-radius: 15px; display: flex; 
                justify-content: center; align-items: center; overflow: hidden; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{st_image_to_base64(image)}" 
                         style="max-height: 90%; max-width: 90%; object-fit: contain; border-radius: 5px;">
                </div>
                """, 
                unsafe_allow_html=True
            )

            # REAL PREDICTION
            input_tensor = preprocess_image(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)

            pred_class = label_names[pred.item()]
            pred_prob = conf.item()
            
            st.markdown(f"### 🔹 Predicted: **{pred_class}**")
            st.progress(pred_prob, text=f"Confidence: {pred_prob:.2%}")
            st.info("ⓘ The model predicts the cell type based on learned patterns.")
        else:
            st.info("ⓘ Please upload an image to start.")

# -----------------------------
# Column 2: Grad-CAM Visualization
# -----------------------------
with col2:
    with st.container(border=True):
        st.subheader("2 Grad-CAM: Visual Explanation")

        if uploaded_file is not None:
            # REAL GRAD-CAM
            cam_image = get_gradcam_image(input_tensor, model)
            st.image(cam_image, use_container_width=True)

            st.info(
                """
                **ⓘ How to interpret this image**
                - Warmer colors (red/yellow) highlight regions that most influenced the prediction.
                - These regions often correspond to important morphological structures.
                """
            )
        else:
            st.info("ⓘ Upload an image to view the Grad-CAM explanation.")

# -----------------------------
# Column 3: Counterfactuals (DiCE)
# -----------------------------
with col3:
    with st.container(border=True):
        st.subheader("3 Counterfactual Analysis")

        if uploaded_file is not None:
            # 1. Run the DiCE Generation with a spinner
            with st.spinner("Calculating clinical counterfactuals..."):
                try:
                    cf_df = get_counterfactuals_single(image)
                    
                    if cf_df is not None and len(cf_df) >= 1:
                        # 2. Process Real Data for the Table
                        # We take the top 2 closest counterfactuals
                        n_results = min(len(cf_df), 2)
                        
                        # 1. Calculate Similarity with calibrated score (multiplier = 8)
                        similarities = [f"{max(0, 100 - (m * 3)):.1f}%" for m in cf_df['change_magnitude'].iloc[:n_results]]
                        
                        # 2. Dynamic Divergence: Find which PC contributes most to the difference
                        divergence_reasons = []
                        pc_cols = [c for c in cf_df.columns if c.startswith('PC')]

                        for i in range(n_results):
                            # Find column with max absolute value in the counterfactual row
                            top_pc = cf_df.iloc[i][pc_cols].abs().idxmax()
                            divergence_reasons.append(f"Structure ({top_pc})")

                        rank_labels = ["Closest Alternative", "2nd Option", "3rd Option"]

                        cf_summary_df = pd.DataFrame({
                            "Rank": rank_labels[:n_results],
                            "Alternative Type": [label_names[int(l)] for l in cf_df['label'].iloc[:n_results]],
                            "Similarity": similarities,
                            "Main Difference": divergence_reasons
                        })

                        # 3. Apply the bolding/UI styling we created earlier
                        def highlight_best(row):
                            if row.name == 0: # First row in the styled view
                                return ['font-weight: bold; background-color: #f1f8e9'] * len(row)
                            return [''] * len(row)

                        styled_df = (cf_summary_df.style
                            .apply(highlight_best, axis=1)
                            .set_properties(**{'color': '#383c43', 'border-color': '#e6ecec'})
                            .hide())

                        # 4. Display the main Clinical Table
                        st.table(styled_df)

                        # 5. Clinical Guidance Notes
                        best_alt = cf_summary_df.iloc[0]['Alternative Type']
                        st.info(f"**AI Insight:** The model identifies high morphological similarity to **{best_alt}** cells.")
                        st.warning(f"⚠️ **Borderline Case:** Subtle shifts in PCA embeddings would change the diagnosis to **{best_alt}**.")

                        # 6. Technical Details Expander
                        # Technical Details (Top 10 + Red Gradient + White Background)
                        with st.expander("🧬 View Morphological Embeddings"):
                            st.write("Top 10 features driving this counterfactual (Sorted by Impact):")
                            
                            # 1. Identify columns
                            all_pc_cols = [c for c in cf_df.columns if c.startswith('PC')]
                            
                            # 2. Sort by magnitude (Impact)
                            row = cf_df.iloc[0]
                            sorted_cols = sorted(all_pc_cols, key=lambda col: abs(row[col]), reverse=True)
                            
                            # 3. Take Top 10
                            top_changing_cols = sorted_cols[:10]
                            tech_df = cf_df[top_changing_cols].iloc[:n_results]
                            
                            # 4. Red Gradient
                            st.dataframe(
                                tech_df.style
                                       .format("{:.2f}")
                                       .set_properties(**{
                                           'background-color': '#ffffff',  # Base White
                                           'color': '#000000',             # Black Text
                                           'border-color': '#e6e6e6'
                                       })
                                       .background_gradient(cmap='Reds', axis=1), # Gradient applies over the white base
                                use_container_width=True, 
                                hide_index=True,
                                height=120
                            )
                            
                            st.caption(f"Showing the most significant features. Notice how **{top_changing_cols[0]}** is now visible.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("DiCE could not find suitable counterfactuals for this image.")
                
                except Exception as e:
                    st.error(f"XAI Engine Error: {str(e)}")
                    st.info("Ensure the PCA and Classifier models are correctly loaded in the assets folder.")
        else:
            st.info("ⓘ Upload an image to generate counterfactual explanations.")
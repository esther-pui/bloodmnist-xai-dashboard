import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import base64
from io import BytesIO


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="BloodMNIST XAI Dashboard", layout="wide")


def st_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
# -----------------------------
# CSS styling
# -----------------------------
st.markdown(
    """
    <style>
        /* 1. App Background & Header */
        .stApp { background-color: #fafcfc; font-family: 'Montserrat', sans-serif; }
        h1 { font-weight: 400 !important; color: #383c43 !important; }
        .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label, .stApp li { color: #383c43 !important; }

        /* 2. Main Card Styling */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff;
            border: 1px solid #e6ecec !important;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
            border-radius: 12px;
            padding: 10px;
        }

        /* 3. Drag & Drop Box Background */
        [data-testid="stFileUploader"] section {
            background-color: #f0f4f4 !important;
            border: 1px dashed #c0cccc !important;
        }

        /* 4. FIX: Browse Files Button (Remove border/background) */
        [data-testid="stFileUploader"] button[kind="secondary"] {
            background-color: transparent !important;
            border: 1px solid #c0cccc !important; /* Kept a thin border for visibility, change to 'none' if desired */
            color: #383c43 !important;
            box-shadow: none !important;
        }

        /* 5. FIX: Uploaded File Row (Remove Grey Box) */
        [data-testid="stUploadedFile"] {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }

        /* 6. FIX: The 'X' Button (Remove its specific background/border) */
        [data-testid="stUploadedFile"] button {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #383c43 !important;
        }

        /* 7. Ensure text is visible */
        [data-testid="stFileUploaderFileName"], 
        [data-testid="stFileUploaderFileData"] small {
            color: #383c43 !important;
        }

        /* Target the inner moving part of the progress bar */
        div[data-testid="stProgress"] > div > div > div > div {
            background-color: #9babfe !important;
        }

        /* Fallback: Target by the ARIA attribute */
        div[role="progressbar"] > div > div {
            background-color: #cccdda !important;
        }

        /* Change background color for all st.info boxes */
        [data-testid="stNotification"] {
            background-color: #dbdef4 !important;
            color: #383c43 !important; /* Ensures text remains dark and readable */
            border: none !important;
        }

        /* Optional: Change the icon color inside the info box to match your theme */
        [data-testid="stNotification"] svg {
            fill: #9babfe !important;
        }
        
        /* FORCE ST.INFO BACKGROUND */
        [data-testid="stNotification"], 
        [data-testid="stAlert"], 
        [data-testid="stAlert"] > div,
        .stAlert {
            background-color: #dbdef4 !important;
            color: #383c43 !important;
            border: none !important;
            border-radius: 15px !important;
        }

        /* Fix the text inside info/warning boxes specifically */
        [data-testid="stNotification"] p, 
        [data-testid="stAlert"] p,
        [data-testid="stNotificationContent"] {
            color: #383c43 !important;
        }

        /* Style the icons inside the boxes to match the dark text */
        [data-testid="stNotification"] svg, 
        [data-testid="stAlert"] svg {
            fill: #383c43 !important;
        }

        /* Remove top padding from the main container */
        /* 1. Initially hide and move the header off-screen */
        header {
            opacity: 0;
            transform: translateY(-50px);
            transition: all 0.3s ease-in-out;
            height: 50px !important;
        }

        /* 2. Show the header when hovering near the top */
        header:hover {
            opacity: 1;
            transform: translateY(0);
            background-color: rgba(255, 255, 255, 0.9) !important;
        }

        /* 3. Keep the content moved up */
        .block-container {
            padding-top: 2rem !important;
            margin-top: -3rem !important;
        }

        /* Center all images inside Streamlit blocks */
        [data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }

        /* Optional: Specifically center images within containers */
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"] img {
            # display: block;
            # margin-left: auto;
            # margin-right: auto;
            max-height: 300px !important;
            width: auto !important;
            object-fit: contain !important;
        }

        /* Create a class for your image frame */
        .image-frame {
            max-height: 340px; /* Slightly larger to account for padding */
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* This creates the colored box effect */
        div[data-testid="stVerticalBlock"] > div:has(div[class*="img-frame"]) {
            background-color: #f8f9fb !important;
            border-radius: 15px !important;
            padding: 8px !important;
            display: flex !important;
            justify-content: center !important;
        }

        /* Force borders on st.table */
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

        /* Remove hover effect from expanders */
        [data-testid="stExpander"] details:hover {
            background-color: transparent !important;
        }

        [data-testid="stExpander"] summary:hover {
            background-color: transparent !important;
            color: inherit !important;
        }

        /* Optional: remove the border change on hover to make it feel completely static */
        [data-testid="stExpander"] summary:hover span {
            color: #383c43 !important;
        }

        /* 1. Make the dataframe background white and remove the dark header */
        [data-testid="stTable"] {
            background-color: #ffffff !important;
        }

        /* 2. Target the specific dataframe container to remove the black/dark theme */
        div[data-testid="stDataFrame"] > div {
            background-color: #ffffff !important;
        }

        /* 3. Style the dataframe headers and cells to match your tables */
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

        /* Create a scrollable container for tables inside expanders */
        [data-testid="stExpander"] div.stHtml {
            overflow-x: auto;
            background-color: #ffffff !important;
        }

        /* Ensure the table inside the expander is white and compact */
        [data-testid="stExpander"] table {
            background-color: #ffffff !important;
            font-size: 0.8rem !important;
        }

        /* FORCE BACKGROUND FOR ALL 3 MAIN COLUMN CONTAINERS */
        [data-testid="stColumn"] > div {
            background-color: #ffffff !important;
            # border: 1px solid #e6ecec !important;
            border-radius: 12px !important;
            padding: 15px !important;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05) !important;
        }

        /* Remove any duplicate borders if using st.container(border=True) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            background-color: transparent !important;
        }

        /* 1. Target the first column of all tables (the hidden index) */
        div[data-testid="stTable"] table tr th:first-child, 
        div[data-testid="stTable"] table tr td:first-child {
            display: none !important;
        }

        /* 2. Fix the image object-fit typo in your existing code */
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"] img {
            max-height: 300px !important;
            width: auto !important;
            object-fit: contain !important; 
        }

        /* Remove the grey/dark hover from the expander bar */
        [data-testid="stExpander"] details summary:hover {
            background-color: transparent !important;
            color: inherit !important;
        }

        [data-testid="stExpander"] details summary:hover span {
            color: #383c43 !important;
        }

        /* Fix the expander bar background to be white and remove borders */
        [data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: none !important;
        }

        [data-testid="stExpander"] details {
            border: 1px solid #dbe9c7 !important;
            border-radius: 8px;
        }

        /* Specifically target the table inside the expander to be pure white */
        [data-testid="stExpander"] table td {
            background-color: #ffffff !important;
        }

        /* Fix the expander bar turning black after clicking/focusing */
        [data-testid="stExpander"] details summary {
            background-color: #dbe9c7 !important;
            color: #383c43 !important;
        }

        [data-testid="stExpander"] details summary:focus, 
        [data-testid="stExpander"] details summary:active,
        [data-testid="stExpander"] details[open] summary {
            background-color: $dbe9c7 !important;
            color: #dbe9c7 !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* Ensure the text inside the bar stays dark */
        [data-testid="stExpander"] details summary p {
            color: #383c43 !important;
        }

        /* Optional: Add a very light border to the expander so it's visible against the white background */
        [data-testid="stExpander"] {
            border: 1px solid #dbe9c7 !important;
            border-radius: 8px !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Blood Cell Classification with Explainable AI 🧬")

st.markdown(
    "This dashboard explains AI blood cell predictions using Grad-CAM (visual) and DiCE (counterfactual) explanations."
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
    with st.container(border=True): 
        st.subheader("1 Upload & Prediction")
        uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # This single block handles centering, background color, and height constraint
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fb; 
                    height: 250px; 
                    border-radius: 15px; 
                    display: flex; 
                    justify-content: center; 
                    align-items: center;
                    overflow: hidden;
                    margin-bottom: 20px;">
                    <img src="data:image/png;base64,{st_image_to_base64(image)}" 
                         style="max-height: 90%; max-width: 90%; object-size: contain; border-radius: 5px;">
                </div>
                """, 
                unsafe_allow_html=True
            )

            pred_class = np.random.choice(mock_classes)
            pred_prob = np.random.uniform(0.7, 0.99)
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
            cam_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            # st.image(cam_image, caption="Grad-CAM heatmap overlay (Mock)", use_container_width=True)
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

with col3:
    with st.container(border=True):
        st.subheader("3 Counterfactual Analysis")

        if uploaded_file is not None:
            # --- MOCK LOGIC FOR BETTER INSIGHTS ---
            n_cfs = 2
            cf_target_classes = np.random.choice(mock_classes, n_cfs, replace=False)
            
            # Convert Magnitude to "Similarity Score" (0-100%)
            magnitudes = np.random.uniform(0.5, 2.5, n_cfs)
            similarities = [f"{max(0, 100 - (m * 20)):.1f}%" for m in magnitudes]
            
            # Clinical reasoning mock-up
            reasons = ["Cytoplasm texture", "Nuclear lobe count"]
            
            cf_summary_df = pd.DataFrame({
                "Rank": ["Best Match", "Alternative"],
                "Alternative Type": cf_target_classes,
                "Similarity": similarities,
                "Key Divergence": reasons
            })

            # --- STYLING ---
            def highlight_best(row):
                return ['font-weight: bold' if row['Rank'] == "Best Match" else '' for _ in row]

            styled_df = (cf_summary_df.style
                .apply(highlight_best, axis=1)
                .set_properties(**{'color': '#383c43', 'border-color': '#e6ecec'})
                .hide())

            st.table(styled_df)

            # Clinical Guidance Note
            st.info(f"**Doctor's Note:** Minimal changes in **{reasons[0]}** would shift the classification to **{cf_target_classes[0]}**.")
            
            st.warning(f"⚠️ **Borderline Case:** This cell shares significant characteristics with **{cf_target_classes[0]}**.")

            # --- TECHNICAL DETAILS ---
            with st.expander("🧬 View Morphological Embeddings"):
                st.write("Raw feature vector changes required to flip classification:")
                n_features = 8
                cf_data = np.random.randn(n_cfs, n_features)
                feature_names = ["Area", "Circularity", "Granularity", "Nucleus Size", "Intensity", "Elongation", "Fractal", "Symmetry"]
                cf_df = pd.DataFrame(cf_data, columns=feature_names)

                st.markdown('<div style="overflow-x: auto;">', unsafe_allow_html=True)
                st.table(cf_df.style.format("{:.2f}").set_properties(**{
                    'background-color': '#ffffff',
                    'color': '#383c43',
                    'font-size': '11px'
                }).hide())
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("ⓘ Upload an image to generate counterfactual explanations.")
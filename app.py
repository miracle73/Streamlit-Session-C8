"""Streamlit app that serves the trained PyTorch model.

Run locally:  streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import Net

# ---------------------------------------------------------------
# Page setup. Must be the first Streamlit call in the script.
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------
# Load the model once, not on every click.
#
# Streamlit re-runs this whole file every time the user touches a
# widget. @st.cache_resource tells it to keep the loaded model in
# memory instead of reading it from disk again and again.
# ---------------------------------------------------------------
@st.cache_resource
def load_everything():
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = Net(len(meta["feature_names"]))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    return model, scaler, meta


model, scaler, meta = load_everything()
FEATURES = meta["feature_names"]
CLASSES = meta["target_names"]  # ['malignant', 'benign']


def predict(rows):
    """rows: a 2D array of raw, unscaled measurements.

    Returns the probability of class 1 (benign) for each row.
    """
    scaled = scaler.transform(np.asarray(rows, dtype=float))
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).numpy().ravel()

    return probs


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🔬 Breast Cancer Tumour Classifier")
st.caption(
    "A PyTorch neural network trained on the Wisconsin breast cancer "
    "dataset, served live with Streamlit."
)

st.sidebar.header("About this model")
st.sidebar.metric("Test accuracy", f"{meta['accuracy']:.1%}")
st.sidebar.write(
    f"""
- **Inputs:** {len(FEATURES)} measurements taken from a tumour image
- **Output:** benign or malignant
- **Network:** {len(FEATURES)} to 32 to 16 to 1, with ReLU and dropout
- **Loss:** BCEWithLogitsLoss, trained for 60 epochs
"""
)
st.sidebar.warning(
    "Educational demo only. This is not a medical device and must not "
    "be used for diagnosis."
)

tab1, tab2, tab3 = st.tabs(["Single prediction", "Batch from CSV", "The data"])

# ---------------------------------------------------------------
# Tab 1: one patient, adjusted with sliders
# ---------------------------------------------------------------
with tab1:
    st.subheader("Enter the measurements")
    st.write(
        "Adjust the ten main measurements below. The remaining features "
        "are held at their dataset median."
    )

    # Start from the median of every feature, then let the user change
    # the first ten, which are the headline "mean" measurements.
    values = list(meta["medians"])

    columns = st.columns(2)
    for i in range(10):
        target_column = columns[i % 2]
        values[i] = target_column.slider(
            FEATURES[i].title(),
            min_value=float(meta["mins"][i]),
            max_value=float(meta["maxs"][i]),
            value=float(meta["medians"][i]),
            key=f"f{i}",
        )

    if st.button("Predict", type="primary", use_container_width=True):
        prob_benign = predict([values])[0]
        label = CLASSES[int(prob_benign > 0.5)]

        st.divider()
        left, right = st.columns(2)

        if label == "benign":
            left.success(f"Prediction: **{label.upper()}**")
        else:
            left.error(f"Prediction: **{label.upper()}**")

        right.metric("Confidence", f"{max(prob_benign, 1 - prob_benign):.1%}")

        st.write("Probability the tumour is benign")
        st.progress(float(prob_benign))
        st.caption(f"{prob_benign:.2%} benign, {1 - prob_benign:.2%} malignant")

# ---------------------------------------------------------------
# Tab 2: many patients at once
# ---------------------------------------------------------------
with tab2:
    st.subheader("Score a whole file")
    st.write(
        f"Upload a CSV with these {len(FEATURES)} columns, in this order, "
        "one row per patient."
    )

    template = pd.DataFrame([meta["medians"]], columns=FEATURES)
    st.download_button(
        "Download a template CSV",
        template.to_csv(index=False).encode(),
        "template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"These columns are missing: {', '.join(missing[:5])} ...")
        else:
            probs = predict(df[FEATURES].values)

            results = df.copy()
            results["probability_benign"] = probs.round(4)
            results["prediction"] = [CLASSES[int(p > 0.5)] for p in probs]

            st.success(f"Scored {len(results)} rows.")

            counts = results["prediction"].value_counts()
            a, b = st.columns(2)
            a.metric("Benign", int(counts.get("benign", 0)))
            b.metric("Malignant", int(counts.get("malignant", 0)))

            st.dataframe(
                results[["probability_benign", "prediction"]],
                use_container_width=True,
            )

            st.download_button(
                "Download the results",
                results.to_csv(index=False).encode(),
                "predictions.csv",
                "text/csv",
            )

# ---------------------------------------------------------------
# Tab 3: what the model actually looks at
# ---------------------------------------------------------------
with tab3:
    st.subheader("The features")
    st.write(
        "Each row of the dataset holds 30 measurements taken from an "
        "image of a tumour, such as radius, texture and smoothness. "
        "The model learned which combinations point to a benign tumour."
    )

    st.dataframe(
        pd.DataFrame(
            {
                "feature": FEATURES,
                "minimum": np.round(meta["mins"], 3),
                "median": np.round(meta["medians"], 3),
                "maximum": np.round(meta["maxs"], 3),
            }
        ),
        use_container_width=True,
        height=400,
    )

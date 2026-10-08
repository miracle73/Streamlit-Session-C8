import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import Net

st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🔬",
    layout= "wide"
)


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
CLASSES = meta["target_names"]

def predict(rows):
    scaled = scaler.transform(np.asarray(rows, dtype=float))
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).numpy().ravel()
    return probs
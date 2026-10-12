# Breast Cancer Tumour Classifier

A PyTorch neural network trained on the Wisconsin breast cancer dataset
and served as a live web app with Streamlit.

Built for the C8 AI/ML Class: the point is not the
model, it is getting a model out of a notebook and in front of people.

## What it does

- **Single prediction:** move the sliders, get a benign or malignant call with a confidence figure
- **Batch from CSV:** upload a file of patients, score them all, download the results
- **The data:** shows the 30 features the model looks at

## Files

| File | Purpose |
|---|---|
| `model.py` | the network definition, imported by both other scripts |
| `train.py` | trains the model and saves the weights, scaler and metadata |
| `app.py` | the Streamlit app that loads those files and serves predictions |
| `model.pth` | the learned weights |
| `scaler.pkl` | the fitted StandardScaler |
| `meta.pkl` | feature names, ranges and the test accuracy |
| `requirements.txt` | what Streamlit Cloud installs |
| `sample_patients.csv` | 25 rows to try the batch tab with |

## Run it locally

```bash
pip install -r requirements.txt
python train.py          # creates model.pth, scaler.pkl, meta.pkl
streamlit run app.py
```

It opens at http://localhost:8501

## Deploy to Streamlit Cloud

1. Push this folder to a **public** GitHub repo. Include `model.pth`,
   `scaler.pkl` and `meta.pkl`, since the cloud does not train anything.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo, the branch, and `app.py`.
4. Click **Deploy** and wait about two minutes.

You get a public URL you can put in your portfolio.

## Three things to teach from this

**1. Train once, load many times.** Training belongs in `train.py`, which
you run on your own machine. The app only loads the result. If you train
inside the app, it retrains on every click and the deploy will be slow
and expensive.

**2. Save the scaler, not just the model.** The model learned from scaled
numbers. If new data arrives unscaled, the prediction is nonsense. The
scaler is part of the model as far as production is concerned.

**3. `@st.cache_resource` matters.** Streamlit re-runs the entire script
on every widget interaction. Without the cache decorator, the model is
read from disk every single time.

## A note on scope

This is a teaching demo, not a medical device. The single prediction tab
also holds 20 of the 30 features at their median so the interface stays
usable, which means its numbers are illustrative rather than clinical.

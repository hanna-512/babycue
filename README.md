# BabyCue — AI-Powered Infant Cry Classification Platform
*Every little cry has a cue.*

- **Live Web Application:** [https://babycue.streamlit.app](https://babycue.streamlit.app)
- **Source Code Repository:** [https://github.com/hanna-512/babycue](https://github.com/hanna-512/babycue)

## Overview
**BabyCue** is a non-clinical caregiver decision-support web application developed for Final Year Project 2 (FYP2). It analyzes infant cry audio (`0.8s–30s` in `.wav`, `.mp3`, `.ogg`, `.m4a`, or `.3gp` format) and classifies cries into **Hunger**, **Pain / Discomfort**, or **Sleepiness**, paired with developmental stage-specific soothing guidance.

## Machine Learning Pipeline
- **Feature Extraction:** Pre-trained **YAMNet** neural network extracting **2,048-D** deep acoustic embeddings (1,024 mean-pooled + 1,024 max-pooled) from standardized 16 kHz mono audio.
- **Preprocessing & Classifier:** L2 Normalization → `StandardScaler` → **64-component PCA** → **Class-Balanced Logistic Regression**.
- **Validation Performance:** Audited 5-fold cross-validation accuracy of **67.55%** across 638 MD5-deduplicated multi-format recordings (**69.45%** across 586 high-SNR WAV recordings).

## Repository Structure
- `app.py` — Main Streamlit web application, audio preprocessing pipeline, and UI components.
- `models/best_model.pkl` — Trained PCA + Logistic Regression classification pipeline.
- `requirements.txt` — Python package dependencies for cloud and local deployment.
- `.streamlit/config.toml` — Custom light-mode color theme configuration.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

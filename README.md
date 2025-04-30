# ReviewTrust: Scoring Trust in Online Product Reviews

📚 A Trusted AI course project at University of South Carolina (CSCE 581)  
📅 Spring 2025 — by Adel Dghim

---

## 🔍 Project Summary

This project presents a system that assigns **trust scores** (from 0 to 1) to product reviews to assess how likely they are to be manipulative or fake. Instead of using binary classification, it uses a **continuous score** to support nuanced decision-making for consumers and platforms alike.

A Streamlit interface was developed to demonstrate scraping real Amazon product reviews and displaying model predictions with SHAP explanations.  
⚠️ *Note: The demo is not functional outside the author's local setup due to dependencies like ChromeDriver, file paths, and local models.*

---

## 🧠 Method Overview

### Models:
- **TF-IDF + Logistic Regression** (baseline, interpretable)
- **BERT (MiniLM) + Metadata + Random Forest** (robust, semantic)

### Explainability:
- **LIME**: Word-level explanation (TF-IDF) & metadata-level explanation (BERT)
- **SHAP**: Local and global explanations for the BERT model

### Dataset:
Custom dataset built by combining:
- Amazon (verified real)
- YelpZIP
- Ott et al. corpus (2011)
- Manually filtered crawl-stage dataset (Kaggle)
- AI-generated fakes (ChatGPT, Claude, DeepSeek)

**Total**: ~59,500 reviews across 13 categories.  
📦 *Some raw files (e.g., `Electronics_5.json`, `yelpzip.csv`) were too large for GitHub and have been excluded.*

---

## 🖥️ Demo Video

📺 Click below to watch the project demo:  
[![Watch the demo](https://img.youtube.com/vi/eEC36IJWgBY/0.jpg)](https://www.youtube.com/watch?v=eEC36IJWgBY)

---

## 🧪 Project Structure

### 📁 `documents/`
- Final Report (PDF)
- Presentation Slides (PDF)

### 📁 `notebooks/`
- `BERT_model.ipynb` — BERT + RandomForest + SHAP + LIME  
- `TFIDF_model.ipynb` — TF-IDF + Logistic Regression + LIME  
- `data_processing.ipynb` — Loads, filters and merges datasets

### 📁 `streamlit_demo/`
- `app.py` — Streamlit interface with SHAP visualizations *(code only; not executable elsewhere)*

### 📁 `data/`
- Final dataset used at run stage *(CSV, preprocessed)*

---

## 🏁 How to Run

⚠️ The demo **cannot be run as-is** due to local setup and ChromeDriver dependencies.  
Only the interface code (`streamlit_demo/app.py`) is included for reference.

---


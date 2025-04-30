# ReviewTrust: Scoring the Credibility of Online Product Reviews

📚 A Trusted AI course project at University of South Carolina (CSCE 581)  
📅 Spring 2025 — by Adel Dghim

## 🔍 Project Summary

This project presents a system that assigns **trust scores** (from 0 to 1) to product reviews to assess how likely they are to be manipulative or fake. Instead of using binary classification, it uses a **continuous score** to support nuanced decision-making for consumers and platforms alike.

A working demo allows users to paste an Amazon product URL, scrape reviews, and view trust scores with SHAP-based explanations.

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

Total: ~59,500 reviews across 13 categories.

## 🖥️ Demo Video

📺 [Watch Demo on YouTube](https://www.youtube.com/watch?v=eEC36IJWgBY)

## 🧪 Project Structure

### 📁 `documents/`
- Final Report (PDF)
- Presentation Slides (PDF)

### 📁 `notebooks/`
- `BERT_model.ipynb` — BERT + RandomForest + SHAP + LIME
- `TFIDF_model.ipynb` — TF-IDF + Logistic Regression + LIME
- `data_processing.ipynb` — Loads, filters and merges datasets

### 📁 `streamlit_demo/`
- `app.py` — Streamlit interface with SHAP visualizations
- `requirements.txt` — Dependencies list

### 📁 `data/`
- Final CSV dataset used in the run stage (filtered, merged)

## 🏁 How to Run

### 🔧 Install dependencies:

```bash
pip install -r streamlit_demo/requirements.txt
```

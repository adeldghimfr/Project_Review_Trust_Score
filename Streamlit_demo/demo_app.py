import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import pickle
import time
import shap
import matplotlib.pyplot as plt

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE
st.set_page_config(layout="wide")

# ---------------------- Initialisation ----------------------
@st.cache_resource
def load_models():
    with open("run_stage_model.pkl", "rb") as f:
        clf = pickle.load(f)
    bert_model = SentenceTransformer("run_stage_bert_model")
    with open("run_stage_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    explainer = shap.TreeExplainer(clf)
    return clf, bert_model, feature_names, explainer

clf, bert_model, feature_names, explainer = load_models()

# ---------------------- Scraping ----------------------
def scrape_amazon_reviews(url, max_reviews=10):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service("/usr/local/bin/chromedriver"), options=options)
    driver.get(url)
    time.sleep(4)

    try:
        see_all = driver.find_element(By.PARTIAL_LINK_TEXT, "See all reviews")
        see_all.click()
        time.sleep(3)
    except:
        pass

    reviews = []
    elements = driver.find_elements(By.CSS_SELECTOR, ".review")

    for e in elements[:max_reviews]:
        try:
            text = e.find_element(By.CSS_SELECTOR, ".review-text-content span").text.strip()
            stars_text = e.find_element(By.CSS_SELECTOR, ".a-icon-alt").get_attribute("innerHTML")
            stars = float(stars_text.split(" ")[0])
            reviews.append((text, stars))
        except:
            continue

    driver.quit()
    return reviews

# ---------------------- Prédiction ----------------------
def predict_review_batch(reviews, category="electronics"):
    results = []
    X_all = []
    for text, stars in reviews:
        full_input = f"{category} - {text}"
        sentiment = TextBlob(text).sentiment.polarity
        length = len(text.split())
        sentiment_gap = abs(sentiment - (stars / 5))

        emb = bert_model.encode([full_input])
        meta = np.array([[sentiment, length, stars, sentiment_gap]])
        full_features = np.concatenate([emb, meta], axis=1)
        X_all.append(full_features[0])

        pred = clf.predict(full_features)[0]
        proba = clf.predict_proba(full_features)[0][1]

        results.append({
            "review": text,
            "stars": stars,
            "predicted_label": "FAKE" if pred == 1 else "REAL",
            "trust_score": round(proba, 2),
            "sentiment": round(sentiment, 2),
            "length": length
        })

    return pd.DataFrame(results), np.array(X_all)

# ---------------------- Interface Streamlit ----------------------
st.title("🛒 Fake Review Detector - Trusted AI Project")

# Initialisation de l'état de session
if 'df_results' not in st.session_state:
    st.session_state.df_results = None
if 'X_all' not in st.session_state:
    st.session_state.X_all = None

default_url = "https://www.amazon.com/Apple-iPhone-SE-3rd-Midnight/dp/B0BDY71GRG/"
url = st.text_input("🔗 Enter the Amazon product URL:", value=default_url)

if st.button("📥 Analyze Reviews"):
    with st.spinner("Scraping and analyzing..."):
        raw_reviews = scrape_amazon_reviews(url, max_reviews=10)
        if raw_reviews:
            df_results, X_all = predict_review_batch(raw_reviews)
            st.session_state.df_results = df_results
            st.session_state.X_all = X_all
            st.success("✅ Reviews processed!")
        else:
            st.error("❌ No reviews found on this page.")

if st.session_state.df_results is not None:
    # ---------------------- Affichage du tableau ----------------------
    st.subheader("📋 Reviews Analysis Table")
    
    # Ajout d'une colonne d'index pour référence
    display_df = st.session_state.df_results.copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Review #"
    
    st.dataframe(display_df[['review', 'stars', 'predicted_label', 'trust_score', 'sentiment', 'length']])
    
    # ---------------------- Analyse Individuelle ----------------------
    st.subheader("🔍 Detailed Review Analysis")
    
    # Sélection par numéro
    selected_num = st.selectbox(
        "Select a review number to analyze:",
        options=display_df.index,
        help="Select a review number from the table above to see detailed analysis"
    )
    
    # Récupération de l'index réel (0-based)
    selected_index = selected_num - 1
    
    # Affichage des détails de la review
    selected_review = st.session_state.df_results.iloc[selected_index]
    st.markdown(f"**Review #{selected_num}:** {selected_review['review']}")
    st.markdown(f"**Stars:** {selected_review['stars']} ⭐")
    st.markdown(f"**Predicted as:** {selected_review['predicted_label']}")
    st.markdown(f"**Trust Score:** {selected_review['trust_score']} (1 = certain fake, 0 = certain real)")
    st.markdown(f"**Sentiment Score:** {selected_review['sentiment']} (Positive > 0, Negative < 0)")
    st.markdown(f"**Length:** {selected_review['length']} words")
    
    # ---------------------- SHAP Waterfall Plot ----------------------
    st.subheader("📊 SHAP Waterfall Explanation")
    st.markdown("""
    **How to read this chart:**
    - The base value is the model's average prediction
    - Each bar shows how much each feature pushed the prediction from the base value
    - Red bars increase the 'FAKE' probability, blue bars decrease it
    - The final value is the model's prediction for this review
    """)
    
    try:
        # Get SHAP values
        shap_values = explainer.shap_values(st.session_state.X_all[selected_index:selected_index+1])
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values_single = shap_values[1][0]  # For binary classification
            expected_value = explainer.expected_value[1]
        elif len(shap_values.shape) == 3:
            shap_values_single = shap_values[0, :, 1]  # For multi-class
            expected_value = explainer.expected_value[1]
        else:
            shap_values_single = shap_values[0]  # For single output
            expected_value = explainer.expected_value

        # Waterfall plot
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_single,
                base_values=expected_value,
                data=st.session_state.X_all[selected_index],
                feature_names=feature_names
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig1)
        
    except Exception as e:
        st.error(f"❌ SHAP waterfall plot failed: {str(e)}")
        
   # ---------------------- Global Feature Importance (Corrigé) ----------------------
    st.subheader("🌐 Global Feature Importance")
    st.markdown("""
    **How to read this chart:**
    - Shows which features are most important overall in the model's decisions
    - Each point represents a review
    - Color shows the feature value (red=high, blue=low)
    - Position on x-axis shows impact on prediction
    """)
    
    try:
        shap_values = explainer(st.session_state.X_all)
        
        # Get the index of the FAKE class (1)
        class_idx = 1 if len(shap_values.values.shape) == 3 else 0
        
        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values.values[:,:,class_idx],
            st.session_state.X_all,
            feature_names=feature_names,
            max_display=15,
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
        
    except Exception as e:
        st.error(f"❌ Could not generate global feature importance: {str(e)}")
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from recommend import (
    load_content_model,
    recommend_content,
    load_cf_model,
    recommend_cf,
)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.caption("End-to-end ML project — content-based + collaborative filtering")

method = st.radio(
    "Recommendation method",
    ["Content-based (by movie title)", "Collaborative filtering (by user ID)"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

if method.startswith("Content"):
    similarity_path = os.path.join(MODELS_DIR, "content_similarity.npy")
    if not os.path.exists(similarity_path):
        st.warning("No content model found yet. Run `python src/train_content.py` first.")
    else:
        similarity, movies_index = load_content_model()
        title = st.text_input("Type a movie title (or part of one):", "Toy Story")
        top_n = st.slider("How many recommendations?", 5, 20, 10)

        if st.button("Get recommendations"):
            results = recommend_content(title, similarity, movies_index, top_n)
            if results.empty:
                st.error("No matching movie found. Try a different title.")
            else:
                st.subheader(f"Because you liked movies matching '{title}':")
                st.dataframe(results, use_container_width=True, hide_index=True)

else:
    cf_path = os.path.join(MODELS_DIR, "cf_svd_model.joblib")
    if not os.path.exists(cf_path):
        st.warning("No CF model found yet. Run `python src/train_cf.py` first.")
    else:
        bundle, movies_index = load_cf_model()
        user_id = st.number_input("Enter a user ID from the dataset:", min_value=1, value=1, step=1)
        top_n = st.slider("How many recommendations?", 5, 20, 10)

        if st.button("Get recommendations"):
            results = recommend_cf(int(user_id), bundle, movies_index, top_n)
            if results.empty:
                st.error("Unknown user ID — try one that exists in ratings.csv.")
            else:
                st.subheader(f"Top picks for user {user_id}:")
                st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()
st.caption("Built with pandas, scikit-learn, MLflow and Streamlit.")

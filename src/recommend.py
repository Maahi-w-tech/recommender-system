"""
Shared inference functions — load saved model artifacts and produce
top-N recommendations. Used by app.py so the UI stays dumb/thin.
""" 

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ---------- Content-based ----------

def load_content_model():
    similarity = np.load(os.path.join(MODEL_DIR, "content_similarity.npy"))
    movies_index = pd.read_csv(os.path.join(MODEL_DIR, "content_movies_index.csv"))
    return similarity, movies_index


def recommend_content(title: str, similarity, movies_index: pd.DataFrame, top_n: int = 10):
    matches = movies_index[movies_index["title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame(columns=["title", "genres"])

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    rec_indices = [i for i, _ in scores]
    return movies_index.iloc[rec_indices][["title", "genres"]]


# ---------- Collaborative filtering ----------

def load_cf_model():
    bundle = joblib.load(os.path.join(MODEL_DIR, "cf_svd_model.joblib"))
    movies_index = pd.read_csv(os.path.join(MODEL_DIR, "cf_movies_index.csv"))
    return bundle, movies_index


def recommend_cf(user_id: int, bundle: dict, movies_index: pd.DataFrame, top_n: int = 10):
    user_to_idx = bundle["user_to_idx"]
    if user_id not in user_to_idx:
        return pd.DataFrame(columns=["title", "genres"])

    u_idx = user_to_idx[user_id]
    scores = bundle["user_factors"][u_idx] @ bundle["movie_factors"].T
    top_movie_idxs = np.argsort(scores)[::-1][:top_n]

    idx_to_movie = bundle["idx_to_movie"]
    movie_ids = [idx_to_movie[i] for i in top_movie_idxs]
    result = movies_index[movies_index["movieId"].isin(movie_ids)][["title", "genres"]]
    return result

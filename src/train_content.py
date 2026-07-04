"""
Content-based recommender: TF-IDF on genres + cosine similarity.

No user history needed at inference time — great for new users / cold start.
Trains fast, so it's a good first model to get the whole pipeline working.
"""

import os
import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_movies

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train():
    movies = load_movies()

    with mlflow.start_run(run_name="content_based_tfidf"):
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(movies["genres_clean"])

        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        mlflow.log_param("model_type", "tfidf_cosine_similarity")
        mlflow.log_param("vocab_size", len(vectorizer.vocabulary_))
        mlflow.log_param("n_movies", movies.shape[0])
        # simple sanity metric: average similarity a movie has to its "neighbors"
        mlflow.log_metric("avg_top5_similarity", _avg_top_k_similarity(similarity, k=5))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "content_vectorizer.joblib"))
        np.save(os.path.join(MODEL_DIR, "content_similarity.npy"), similarity)
        movies[["movieId", "title", "genres"]].to_csv(
            os.path.join(MODEL_DIR, "content_movies_index.csv"), index=False
        )

        mlflow.log_artifact(os.path.join(MODEL_DIR, "content_movies_index.csv"))
        print(f"Saved content-based model artifacts to {MODEL_DIR}/")


def _avg_top_k_similarity(similarity: np.ndarray, k: int = 5) -> float:
    # exclude self-similarity (diagonal), average the top-k neighbor scores per movie
    sim = similarity.copy()
    np.fill_diagonal(sim, -1)
    top_k = np.sort(sim, axis=1)[:, -k:]
    return float(top_k.mean())


if __name__ == "__main__":
    train()

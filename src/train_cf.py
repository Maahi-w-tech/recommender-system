"""
Collaborative filtering recommender: matrix factorization via TruncatedSVD
on the sparse user-item ratings matrix.

Captures "users who liked X also liked Y" patterns without needing any
content/metadata — but needs existing rating history for a user to work.
"""

import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data_loader import load_ratings, load_movies

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
N_COMPONENTS = 30  # latent factors — tune this


def build_user_item_matrix(ratings: pd.DataFrame):
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

    rows = ratings["userId"].map(user_to_idx)
    cols = ratings["movieId"].map(movie_to_idx)
    matrix = csr_matrix(
        (ratings["rating"], (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )
    return matrix, user_to_idx, movie_to_idx


def train():
    ratings = load_ratings()
    movies = load_movies()

    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    matrix, user_to_idx, movie_to_idx = build_user_item_matrix(train_df)

    with mlflow.start_run(run_name="collaborative_filtering_svd"):
        mlflow.log_param("model_type", "truncated_svd")
        mlflow.log_param("n_components", N_COMPONENTS)
        mlflow.log_param("n_users", matrix.shape[0])
        mlflow.log_param("n_movies", matrix.shape[1])

        svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
        user_factors = svd.fit_transform(matrix)
        movie_factors = svd.components_.T  # (n_movies, n_components)

        rmse = _evaluate(test_df, user_factors, movie_factors, user_to_idx, movie_to_idx)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("explained_variance_ratio", float(svd.explained_variance_ratio_.sum()))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(
            {
                "user_factors": user_factors,
                "movie_factors": movie_factors,
                "user_to_idx": user_to_idx,
                "movie_to_idx": movie_to_idx,
                "idx_to_movie": {v: k for k, v in movie_to_idx.items()},
            },
            os.path.join(MODEL_DIR, "cf_svd_model.joblib"),
        )
        movies[["movieId", "title", "genres"]].to_csv(
            os.path.join(MODEL_DIR, "cf_movies_index.csv"), index=False
        )

        print(f"RMSE on held-out ratings: {rmse:.4f}")
        print(f"Saved CF model artifacts to {MODEL_DIR}/")


def _evaluate(test_df, user_factors, movie_factors, user_to_idx, movie_to_idx) -> float:
    preds, actuals = [], []
    for row in test_df.itertuples():
        u, m, r = row.userId, row.movieId, row.rating
        if u in user_to_idx and m in movie_to_idx:
            pred = np.dot(user_factors[user_to_idx[u]], movie_factors[movie_to_idx[m]])
            preds.append(pred)
            actuals.append(r)
    if not preds:
        return float("nan")
    return float(np.sqrt(mean_squared_error(actuals, preds)))


if __name__ == "__main__":
    train()

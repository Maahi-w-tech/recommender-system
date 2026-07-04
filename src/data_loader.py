"""
Data loading + light cleaning for the MovieLens dataset.

Expects data/movies.csv and data/ratings.csv (MovieLens "latest-small" format).
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_movies() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "movies.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the MovieLens dataset and place "
            "movies.csv in the data/ folder — see README.md."
        )
    df = pd.read_csv(path)
    # genres column looks like "Adventure|Animation|Children" -> normalize to spaces
    # so it plays nicely with TF-IDF
    df["genres_clean"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
    return df


def load_ratings() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ratings.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the MovieLens dataset and place "
            "ratings.csv in the data/ folder — see README.md."
        )
    df = pd.read_csv(path)
    return df


def load_merged() -> pd.DataFrame:
    """Ratings joined with movie titles/genres — handy for EDA."""
    movies = load_movies()
    ratings = load_ratings()
    return ratings.merge(movies, on="movieId", how="left")


if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings()
    print(f"Movies: {movies.shape[0]} rows")
    print(f"Ratings: {ratings.shape[0]} rows, {ratings['userId'].nunique()} users")
    print(movies.head())

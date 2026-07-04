# Movie Recommender — End-to-End ML Project

A weekend-scoped, full pipeline recommender system: data → model → experiment
tracking → Streamlit app → Docker.

## 1. Get the data

Download the **MovieLens Latest Small** dataset (100k ratings, ~9k movies,
no signup needed):

https://grouplens.org/datasets/movielens/latest/

Unzip it and place `movies.csv` and `ratings.csv` into the `data/` folder:

```
recommender-project/
  data/
    movies.csv
    ratings.csv
```

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train the models

Two approaches are included — pick one to start, both are wired up:

**Content-based** (TF-IDF on genres + cosine similarity — fast, simple, no
cold-start problem for new users):

```bash
python src/train_content.py
```

**Collaborative filtering** (matrix factorization via TruncatedSVD on the
user-item ratings matrix — captures taste patterns, needs existing user
history):

```bash
python src/train_cf.py
```

Both scripts save their artifacts into `models/` and log params + metrics to
**MLflow**. View the experiment dashboard with:

```bash
mlflow ui
```
then open http://localhost:5000

## 4. Run the app

```bash
streamlit run app.py
```

Opens at http://localhost:8501 — pick a movie or user ID, get recommendations.

## 5. Docker (optional deploy step)

```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

## Project structure

```
recommender-project/
  data/                  # put movies.csv, ratings.csv here
  models/                # trained artifacts land here (gitignored)
  src/
    data_loader.py        # loading + light cleaning
    train_content.py      # TF-IDF + cosine similarity model
    train_cf.py            # SVD collaborative filtering model
    recommend.py           # shared inference functions used by app.py
  app.py                  # Streamlit UI
  requirements.txt
  Dockerfile
  README.md
```

## Suggested next steps (stretch goals)

- Hybrid: blend content + CF scores
- Evaluate with precision@k / recall@k instead of just RMSE
- Swap TruncatedSVD for `implicit` (ALS) if you move to implicit feedback data
- Deploy to Streamlit Community Cloud or Render for a live shareable link
- Add a `/predict` FastAPI endpoint alongside the Streamlit app for an API-style deployment

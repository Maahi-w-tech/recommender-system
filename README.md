# 🎬 Movie Recommender System

An end-to-end Machine Learning project that recommends movies using both **Content-Based Filtering** and **Collaborative Filtering** techniques.

The project demonstrates the complete ML workflow—from data preprocessing and model training to experiment tracking, deployment with Streamlit.

---

##  Features

- Content-Based Recommendation using **TF-IDF** and **Cosine Similarity**
- Collaborative Filtering using **Truncated SVD**
- Interactive Streamlit web application
- MLflow experiment tracking
- Modular project structure

---

##  Dataset

This project uses the **MovieLens Latest Small** dataset.

Download it here:

https://grouplens.org/datasets/movielens/latest/

After downloading, extract the files and place them inside the `data` directory.

```
recommender-project/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
```

---

##  Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/recommender-system.git
cd recommender-system
```

Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

##  Train the Models

### Content-Based Filtering

Uses TF-IDF vectorization on movie genres followed by cosine similarity.

```bash
python src/train_content.py
```

### Collaborative Filtering

Uses Truncated Singular Value Decomposition (SVD) on the user-item ratings matrix.

```bash
python src/train_cf.py
```

Both models automatically save their trained artifacts inside the `models/` folder.

---

##  MLflow Experiment Tracking

Launch the MLflow dashboard

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

to view parameters, metrics, and experiment runs.

---

##  Run the Application

Start the Streamlit app

```bash
streamlit run app.py
```

The application will be available at

```
http://localhost:8501
```

Users can choose a movie or provide a user ID to receive personalized recommendations.

---



## 📁 Project Structure

```
recommender-project/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── models/
│
├── src/
│   ├── data_loader.py
│   ├── train_content.py
│   ├── train_cf.py
│   └── recommend.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

##  Machine Learning Pipeline

1. Load MovieLens dataset
2. Preprocess movies and ratings
3. Train Content-Based recommender
4. Train Collaborative Filtering recommender
5. Save trained models
6. Log experiments with MLflow
7. Serve recommendations using Streamlit

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Streamlit
  

---

##  Future Improvements

- Hybrid recommendation system
- Precision@K and Recall@K evaluation
- Implicit feedback recommendation (ALS)
- FastAPI REST API
- Cloud deployment (Render / Streamlit Community Cloud)
- User authentication
- Movie posters using TMDB API

---

##  License

This project uses the MovieLens dataset provided by GroupLens Research.

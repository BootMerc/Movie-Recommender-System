A movie recommendation system built with Collaborative Filtering, Content-Based Filtering, and a Hybrid approach. Trained on the MovieLens dataset and served through an interactive Streamlit app.

---

## What it does

CineMatch predicts movies a user will enjoy based on two complementary strategies:

- **Collaborative Filtering** — finds patterns across all users. If you and another user rate the same movies similarly, CineMatch surfaces movies they liked that you haven't seen yet.
- **Content-Based Filtering** — analyses each movie's genres and tags using TF-IDF. If you liked *Inception*, it finds movies with similar content fingerprints.
- **Hybrid** — combines both scores for better overall accuracy, and handles cold-start users gracefully.

---

## Project structure

```
cinematch/
│
├── data/                      # Auto-created on first run
│   ├── ml-latest-small/       # MovieLens dataset (downloaded automatically)
│   └── figures/               # EDA plots saved as PNG
│
├── src/
│   ├── data_loader.py         # Downloads and loads MovieLens CSV files
│   ├── preprocessing.py       # Cleans data, builds user-item matrix, runs EDA
│   ├── collaborative.py       # SVD matrix factorization (collaborative filter)
│   ├── content_based.py       # TF-IDF + cosine similarity (content filter)
│   ├── hybrid.py              # Weighted, Switching, and Cascade hybrid models
│   └── evaluation.py          # RMSE, MAE, Precision@K, Recall@K, NDCG@K
│
├── main.py                    # Runs the full pipeline from the command line
├── app.py                     # Streamlit web application
└── requirements.txt
```

---

## Workflow

```
Download data → Clean & filter → Build features → Train models → Evaluate → Recommend
```

1. **Data** — `data_loader.py` downloads the MovieLens Latest Small dataset (~6 MB) automatically on first run. No manual setup needed.

2. **Preprocessing** — `preprocessing.py` cleans ratings, removes inactive users and rarely-rated movies, extracts release years, builds the user-item rating matrix, and combines genres with user tags into a content string per movie.

3. **Collaborative Filtering** — `collaborative.py` applies Truncated SVD to the rating matrix. It decomposes the matrix into user and movie latent factor matrices, then reconstructs predicted ratings for every (user, movie) pair.

4. **Content-Based Filtering** — `content_based.py` runs TF-IDF on each movie's genre + tag text. It precomputes a full cosine similarity matrix between all movies. For a user, it averages the similarity vectors of their liked movies to surface new titles.

5. **Hybrid** — `hybrid.py` offers three strategies. The Weighted Hybrid blends CF and CB scores linearly. The Switching Hybrid uses CF for users with enough history and CB for new users. The Cascade Hybrid uses CF to recall 60 candidates, then re-ranks them with CB.

6. **Evaluation** — `evaluation.py` measures rating prediction accuracy (RMSE, MAE) and ranking quality (Precision@K, Recall@K, F1@K, NDCG@K) on a held-out test set split by time.

---

## Getting started

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/cinematch.git
cd cinematch

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (trains all models, prints evaluation, shows demo)
python main.py

# 5. Launch the interactive web app
streamlit run app.py
```

The dataset downloads automatically the first time you run either command.

---

## Dataset

**MovieLens Latest Small** — [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/latest/)

| | |
|---|---|
| Ratings | 100,836 |
| Users | 610 |
| Movies | 9,742 |
| Rating scale | 0.5 – 5.0 |

---

## Tech stack

| Purpose | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Matrix factorization | scipy (svds) |
| TF-IDF & similarity | scikit-learn |
| Optional CF | scikit-surprise |
| Web app | streamlit |
| Charts | plotly, seaborn |

---

## Streamlit app tabs

| Tab | What it shows |
|---|---|
| 🎬 Recommendations | Top-N picks for any user, switchable between CF / CB / Hybrid |
| 🔍 Similar Movies | Movies most similar in content to a chosen title |
| 🔎 Keyword Search | Search the catalogue with free-text keywords |
| 📊 EDA Dashboard | Rating distributions, genre popularity, activity heatmap |
| 📐 Evaluation | Live RMSE, Precision@K, NDCG@K comparison across all models |

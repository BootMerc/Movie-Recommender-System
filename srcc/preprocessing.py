import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

FIGURES_DIR = Path(__file__).resolve().parent.parent / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


#  1. CLEANING

def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:

    df = ratings.copy()

    before = len(df)
    df.drop_duplicates(subset=["userId", "movieId"], keep="last", inplace=True)
    df = df[df["rating"].between(0.5, 5.0)]
    df.reset_index(drop=True, inplace=True)

    print(f"[preprocessing] Ratings cleaned: {before:,} → {len(df):,} rows "
          f"(removed {before - len(df):,}).")
    return df


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:

    df = movies.copy()

    # Extract year from title e.g. "Toy Story (1995)" → 1995
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("float")
    df["title"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    df["genres"] = df["genres"].replace("(no genres listed)", "")

    print(f"[preprocessing] Movies cleaned: {len(df):,} rows, "
          f"{df['year'].notna().sum():,} have a release year.")
    return df


def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:

    df = tags.copy()
    df["tag"] = df["tag"].astype(str).str.lower().str.strip()
    df = df[df["tag"].str.len() > 0].reset_index(drop=True)
    print(f"[preprocessing] Tags cleaned: {len(df):,} rows.")
    return df
#  2. FILTERING (activity thresholds)

def filter_by_activity(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 10,
) -> pd.DataFrame:

    before = len(ratings)

    # Iterative filter (ratings can cross-affect each other)
    df = ratings.copy()
    for _ in range(3):
        user_counts  = df.groupby("userId")["movieId"].count()
        movie_counts = df.groupby("movieId")["userId"].count()
        active_users  = user_counts[user_counts  >= min_user_ratings].index
        active_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df["userId"].isin(active_users) & df["movieId"].isin(active_movies)]

    df.reset_index(drop=True, inplace=True)
    print(f"[preprocessing] Activity filter: {before:,} → {len(df):,} ratings | "
          f"{df['userId'].nunique():,} users | {df['movieId'].nunique():,} movies.")
    return df

#  3. USER-ITEM MATRIX
def build_user_item_matrix(ratings: pd.DataFrame) -> tuple[pd.DataFrame, csr_matrix, dict, dict]:

    matrix_df = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )

    user2idx  = {uid: i for i, uid  in enumerate(matrix_df.index)}
    movie2idx = {mid: j for j, mid in enumerate(matrix_df.columns)}

    # Fill NaN with 0 for sparse representation
    sparse_mat = csr_matrix(matrix_df.fillna(0).values)

    sparsity = 1 - (ratings.shape[0] / (matrix_df.shape[0] * matrix_df.shape[1]))
    print(f"[preprocessing] User-item matrix: {matrix_df.shape[0]:,} users × "
          f"{matrix_df.shape[1]:,} movies — sparsity {sparsity:.2%}.")

    return matrix_df, sparse_mat, user2idx, movie2idx

#  4. CONTENT FEATURES
def build_content_features(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:

    df = movies.copy()

    # Genres → space-separated tokens (replace | with space)
    df["genre_list"] = df["genres"].str.replace("|", " ", regex=False).str.lower()

    # Aggregate tags per movie
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda ts: " ".join(ts.tolist()))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )
    df = df.merge(tag_agg, on="movieId", how="left")
    df["tags_text"] = df["tags_text"].fillna("")

    # Soup = genres + tags (genres repeated to boost weight)
    df["content_soup"] = df["genre_list"] + " " + df["genre_list"] + " " + df["tags_text"]
    df["content_soup"] = df["content_soup"].str.strip()

    print(f"[preprocessing] Content features built for {len(df):,} movies.")
    return df

#  5. EXPLORATORY DATA ANALYSIS
def run_eda(ratings: pd.DataFrame, movies: pd.DataFrame, save_plots: bool = True) -> dict:

    summary = {}

    #  Basic statistics 
    summary["n_ratings"]       = len(ratings)
    summary["n_users"]         = ratings["userId"].nunique()
    summary["n_movies"]        = ratings["movieId"].nunique()
    summary["rating_mean"]     = round(ratings["rating"].mean(), 3)
    summary["rating_median"]   = ratings["rating"].median()
    summary["rating_std"]      = round(ratings["rating"].std(), 3)
    summary["sparsity"]        = round(
        1 - len(ratings) / (summary["n_users"] * summary["n_movies"]), 4
    )

    ratings_per_user  = ratings.groupby("userId")["movieId"].count()
    ratings_per_movie = ratings.groupby("movieId")["userId"].count()
    summary["avg_ratings_per_user"]  = round(ratings_per_user.mean(), 2)
    summary["avg_ratings_per_movie"] = round(ratings_per_movie.mean(), 2)

    print("\n" + "═" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("═" * 55)
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("═" * 55 + "\n")

    if not save_plots:
        return summary

    sns.set_theme(style="whitegrid", palette="muted")

    #  1: Rating distribution 
    fig, ax = plt.subplots(figsize=(8, 4))
    ratings["rating"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="#4C72B0", edgecolor="white"
    )
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "rating_distribution.png", dpi=150)
    plt.close(fig)

    #  Plot 2: Ratings per user (log scale) 
    fig, ax = plt.subplots(figsize=(8, 4))
    ratings_per_user.hist(bins=50, ax=ax, color="#55A868", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Ratings per User (log scale)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ratings_per_user.png", dpi=150)
    plt.close(fig)

    #  Plot 3: Top-20 most-rated movies 
    top_movies = (
        ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .merge(movies[["movieId", "title"]], on="movieId")
        .nlargest(20, "count")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_movies, x="count", y="title", ax=ax, palette="Blues_r")
    ax.set_title("Top 20 Most-Rated Movies", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "top20_movies.png", dpi=150)
    plt.close(fig)

    #  Plot 4: Genre popularity 
    genre_counts = (
        movies["genres"]
        .str.split("|")
        .explode()
        .value_counts()
        .drop("(no genres listed)", errors="ignore")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    genre_counts.head(15).plot(kind="bar", ax=ax, color="#C44E52", edgecolor="white")
    ax.set_title("Genre Popularity (# of Movies)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "genre_popularity.png", dpi=150)
    plt.close(fig)

    print(f"[preprocessing] EDA plots saved to '{FIGURES_DIR}'.")
    return summary

#  6. TRAIN / TEST SPLIT  (temporal – mimics real deployment)
def temporal_train_test_split(ratings: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:

    ratings_sorted = ratings.sort_values(["userId", "timestamp"])

    def split_user(grp):
        n_test = max(1, int(len(grp) * test_ratio))
        return grp.iloc[:-n_test], grp.iloc[-n_test:]

    train_parts, test_parts = zip(*[split_user(g) for _, g in ratings_sorted.groupby("userId")])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df  = pd.concat(test_parts).reset_index(drop=True)

    print(f"[preprocessing] Train/test split → train: {len(train_df):,} | test: {len(test_df):,}")
    return train_df, test_df
#  7. NORMALISATION HELPERS

def mean_center_ratings(matrix_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    user_means  = matrix_df.mean(axis=1)
    centered_df = matrix_df.sub(user_means, axis=0)
    return centered_df, user_means

import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

#Project imports 
from src.data_loader   import load_all, download_dataset
from src.preprocessing import (
    clean_ratings, clean_movies, clean_tags, filter_by_activity,
    build_user_item_matrix, build_content_features,
    temporal_train_test_split, run_eda,
)
from src.collaborative  import SVDRecommender
from src.content_based  import ContentBasedRecommender
from src.hybrid         import WeightedHybrid, CascadeHybrid, SwitchingHybrid
from src.evaluation     import (
    evaluate_rating_prediction, evaluate_ranking,
    print_metrics, catalogue_coverage,
)
#  CONFIGURATION
CFG = {
    # Filtering thresholds
    "min_user_ratings"  : 20,
    "min_movie_ratings" : 10,
    # Model hyper-params
    "svd_factors"       : 50,
    "tfidf_max_features": 10_000,
    # Recommendation settings
    "n_recs"            : 10,
    "eval_k"            : 10,
    "relevance_threshold": 4.0,
    # Hybrid weights (CF : CB)
    "cf_weight"         : 0.65,
    "candidate_k"       : 60,
    # Demo
    "demo_user_id"      : None,   # auto-select
    "demo_movie_id"     : None,   # auto-select
}
#  HELPER
def _section(title: str) -> None:
    print(f"  {title}")
    print("-" * 60)
    
def _pick_demo_user(ratings_df: pd.DataFrame, min_ratings: int = 50) -> int:
    """Pick a user with enough ratings for a meaningful demo."""
    counts = ratings_df.groupby("userId")["movieId"].count()
    eligible = counts[counts >= min_ratings].index
    return int(np.random.choice(eligible))

def _pick_demo_movie(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> int:
    """Pick one of the most-rated movies for the similar-items demo."""
    top = ratings_df.groupby("movieId")["rating"].count().nlargest(30)
    mid = int(np.random.choice(top.index))
    return mid

#  MAIN PIPELINE
def main():
    np.random.seed(42)

    # 1. Load data 
    _section("1 │ Loading Data")
    data       = load_all()
    ratings_raw = data["ratings"]
    movies_raw  = data["movies"]
    tags_raw    = data["tags"]
    #2. Preprocess 
    _section("2 │ Preprocessing")
    ratings = clean_ratings(ratings_raw)
    movies  = clean_movies(movies_raw)
    tags    = clean_tags(tags_raw)

    # Activity-based filtering
    ratings = filter_by_activity(
        ratings,
        min_user_ratings  = CFG["min_user_ratings"],
        min_movie_ratings = CFG["min_movie_ratings"],
    )
    #  3. EDA 
    _section("3 │ Exploratory Data Analysis")
    eda_stats = run_eda(ratings, movies, save_plots=True)
    #  4. User-Item Matrix 
    _section("4 │ User-Item Matrix")
    matrix_df, sparse_mat, user2idx, movie2idx = build_user_item_matrix(ratings)

    #  5. Train / Test Split 
    _section("5 │ Train / Test Split")
    train_df, test_df = temporal_train_test_split(ratings, test_ratio=0.2)

    #  6. Content Features 
    _section("6 │ Building Content Features")
    movies_with_features = build_content_features(movies, tags)

    #  7. Train SVD (Collaborative Filtering) 
    _section("7 │ Training Collaborative Filter (SVD)")
    cf_model = SVDRecommender(n_factors=CFG["svd_factors"])
    cf_model.fit(train_df)

    #  8. Train Content-Based Model 
    _section("8 │ Training Content-Based Filter (TF-IDF)")
    cb_model = ContentBasedRecommender(max_features=CFG["tfidf_max_features"])
    cb_model.fit(movies_with_features)

    #  9. Build Hybrid Models 
    _section("9 │ Building Hybrid Recommenders")
    weighted_hybrid  = WeightedHybrid(cf_weight=CFG["cf_weight"]).fit(cf_model, cb_model)
    cascade_hybrid   = CascadeHybrid(candidate_k=CFG["candidate_k"]).fit(cf_model, cb_model)
    switching_hybrid = SwitchingHybrid(warm_threshold=20).fit(cf_model, cb_model)

    #  10. Evaluation 
    _section("10 │ Evaluation")
    # 10a. Rating prediction (RMSE / MAE)
    print("\n▶ Rating Prediction — Collaborative Filter (SVD)")
    rating_metrics_cf = evaluate_rating_prediction(
        test_df, predict_fn=cf_model.predict, sample_n=3_000
    )
    print_metrics("SVD — Rating Prediction", rating_metrics_cf)
    # 10b. Ranking metrics for CF
    print("\n▶ Ranking Metrics — Collaborative Filter (SVD)")
    def cf_rec_fn(uid, n):
        seen = set(train_df[train_df["userId"] == uid]["movieId"])
        return cf_model.recommend(uid, n, already_seen=seen)["movieId"].tolist()

    ranking_metrics_cf = evaluate_ranking(
        test_df,
        recommend_fn        = cf_rec_fn,
        k                   = CFG["eval_k"],
        relevance_threshold = CFG["relevance_threshold"],
        n_users             = 150,
    )
    print_metrics(f"SVD — Ranking @{CFG['eval_k']}", ranking_metrics_cf)

    # 10c. Ranking metrics for CB
    print("\n▶ Ranking Metrics — Content-Based Filter (TF-IDF)")
    def cb_rec_fn(uid, n):
        recs = cb_model.recommend(uid, n, ratings_df=train_df)
        return recs["movieId"].tolist()

    ranking_metrics_cb = evaluate_ranking(
        test_df,
        recommend_fn        = cb_rec_fn,
        k                   = CFG["eval_k"],
        relevance_threshold = CFG["relevance_threshold"],
        n_users             = 150,
    )
    print_metrics(f"Content-Based — Ranking @{CFG['eval_k']}", ranking_metrics_cb)

    # 10d. Ranking metrics for Weighted Hybrid
    print("\n▶ Ranking Metrics — Weighted Hybrid")
    def hybrid_rec_fn(uid, n):
        recs = weighted_hybrid.recommend(uid, n, ratings_df=train_df, movies_df=movies)
        return recs["movieId"].tolist()

    ranking_metrics_hybrid = evaluate_ranking(
        test_df,
        recommend_fn        = hybrid_rec_fn,
        k                   = CFG["eval_k"],
        relevance_threshold = CFG["relevance_threshold"],
        n_users             = 150,
    )
    print_metrics(f"Weighted Hybrid — Ranking @{CFG['eval_k']}", ranking_metrics_hybrid)

    # 10e. Catalogue coverage
    print("\n▶ Catalogue Coverage")
    all_users  = train_df["userId"].unique().tolist()
    all_movies = movies["movieId"].tolist()
    cov = catalogue_coverage(cf_rec_fn, all_users, all_movies, k=CFG["eval_k"], sample_users=200)
    print(f"  Catalogue coverage (CF): {cov:.2%}")

    #  11. Demo: Top-N Recommendations for a User 
    _section("11 │ Demo — Top-N Movie Recommendations for a User")

    demo_user = CFG["demo_user_id"] or _pick_demo_user(train_df)
    seen_by_demo_user = set(train_df[train_df["userId"] == demo_user]["movieId"])

    print(f"\n  Demo User ID : {demo_user}")
    print(f"  Movies rated : {len(seen_by_demo_user)}")

    # User's top-rated movies (history)
    user_history = (
        train_df[train_df["userId"] == demo_user]
        .merge(movies[["movieId", "title"]], on="movieId")
        .nlargest(5, "rating")[["title", "rating"]]
    )
    print("\n  ── User's Highest-Rated Movies (sample) ──")
    print(user_history.to_string(index=False))

    # CF recommendations
    print(f"\n  ── Collaborative Filter — Top {CFG['n_recs']} Recommendations ──")
    cf_recs = cf_model.recommend(
        demo_user, CFG["n_recs"], movies_df=movies, already_seen=seen_by_demo_user
    )
    print(cf_recs[["title", "genres", "predicted_rating"]].to_string(index=False))

    # CB recommendations
    print(f"\n  ── Content-Based Filter — Top {CFG['n_recs']} Recommendations ──")
    cb_recs = cb_model.recommend(demo_user, CFG["n_recs"], ratings_df=train_df)
    print(cb_recs[["title", "genres", "score"]].to_string(index=False))

    # Hybrid recommendations
    print(f"\n  ── Weighted Hybrid — Top {CFG['n_recs']} Recommendations ──")
    hybrid_recs = weighted_hybrid.recommend(
        demo_user, CFG["n_recs"], ratings_df=train_df, movies_df=movies
    )
    print(hybrid_recs[["title", "genres", "hybrid_score"]].to_string(index=False))

    #  12. Demo: Similar Movies 
    _section("12 │ Demo — Similar Movies (Content-Based)")

    demo_movie_id    = CFG["demo_movie_id"] or _pick_demo_movie(movies, train_df)
    demo_movie_title = movies.loc[movies["movieId"] == demo_movie_id, "title"].values
    demo_movie_title = demo_movie_title[0] if len(demo_movie_title) else "Unknown"

    print(f"\n  Query movie : '{demo_movie_title}' (ID: {demo_movie_id})")
    similar = cb_model.similar_movies(demo_movie_id, n=10)
    print(f"\n  ── Top 10 Similar Movies ──")
    print(similar[["title", "genres", "similarity"]].to_string(index=False))

    # 13. Demo: Keyword search 
    _section("13 │ Demo — Movie Search by Keywords")
    keywords = "space sci-fi adventure galaxy"
    print(f"  Query: \"{keywords}\"")
    keyword_recs = cb_model.recommend_by_keywords(keywords, n=8)
    print(keyword_recs[["title", "genres", "similarity"]].to_string(index=False))

    #  Final Summary 
    _section("Pipeline Complete")
    print("\n  Models trained      : SVD (CF) | TF-IDF (CB) | Weighted / Cascade / Switching Hybrid")
    print(f"  EDA plots saved     : data/figures/")
    print(f"  Test RMSE (SVD)     : {rating_metrics_cf['rmse']}")
    k = CFG["eval_k"]
    print(f"  Precision@10 (CF)   : {ranking_metrics_cf[f'precision@{k}']}")
    print(f"  Recall@10 (CF)      : {ranking_metrics_cf[f'recall@{k}']}")
    print(f"  NDCG@10 (CF)        : {ranking_metrics_cf[f'ndcg@{k}']}")
    print(f"  NDCG@10 (Hybrid)    : {ranking_metrics_hybrid[f'ndcg@{k}']}")
    print("\n  Run `streamlit run app.py` to launch the interactive UI.")
    print()
if __name__ == "__main__":
    main()
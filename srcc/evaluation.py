import numpy as np
import pandas as pd
from typing import Callable
#  1. RATING-PREDICTION METRICS
def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:

    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:

    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def evaluate_rating_prediction(
    test_df           : pd.DataFrame,
    predict_fn        : Callable,           # (userId, movieId) → float
    sample_n          : int = 5_000,
) -> dict:
    sample = test_df.sample(min(sample_n, len(test_df)), random_state=42)

    actuals, preds = [], []
    for _, row in sample.iterrows():
        try:
            pred = predict_fn(int(row["userId"]), int(row["movieId"]))
            actuals.append(row["rating"])
            preds.append(pred)
        except Exception:
            pass   # skip cold-start failures

    return {
        "rmse"        : round(rmse(actuals, preds), 4),
        "mae"         : round(mae(actuals, preds),  4),
        "n_evaluated" : len(actuals),
    }
#  2. RANKING METRICS  (Precision@K, Recall@K, F1@K, NDCG@K)

def precision_at_k(recommended: list, relevant: set, k: int) -> float:

    top_k = recommended[:k]
    hits  = sum(1 for mid in top_k if mid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:

    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for mid in top_k if mid in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int) -> float:

    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:

    top_k = recommended[:k]

    # DCG: sum of rel_i / log2(rank + 1)  where rank is 1-based
    dcg = sum(
        1.0 / np.log2(rank + 2)           # +2 because rank is 0-based
        for rank, mid in enumerate(top_k)
        if mid in relevant
    )

    # Ideal DCG: place all relevant items at the top
    ideal_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0
#  3. FULL EVALUATION PIPELINE
def evaluate_ranking(
    test_df              : pd.DataFrame,
    recommend_fn         : Callable,         # (user_id, n) → list[movieId]
    k                    : int   = 10,
    relevance_threshold  : float = 4.0,
    n_users              : int   = 200,
    ratings_df           : pd.DataFrame = None,
) -> dict:
    users = test_df["userId"].unique()
    np.random.seed(42)
    users = np.random.choice(users, min(n_users, len(users)), replace=False)

    precisions, recalls, f1s, ndcgs = [], [], [], []
    skipped = 0

    for uid in users:
        user_test = test_df[test_df["userId"] == uid]
        relevant  = set(
            user_test[user_test["rating"] >= relevance_threshold]["movieId"].tolist()
        )

        if len(relevant) == 0:
            skipped += 1
            continue

        try:
            recommended = recommend_fn(uid, k)        # list of movieIds
            if not recommended:
                skipped += 1
                continue

            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            f1s.append(f1_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))

        except Exception as e:
            skipped += 1

    results = {
        f"precision@{k}"  : round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"recall@{k}"     : round(float(np.mean(recalls)),    4) if recalls    else 0.0,
        f"f1@{k}"         : round(float(np.mean(f1s)),        4) if f1s        else 0.0,
        f"ndcg@{k}"       : round(float(np.mean(ndcgs)),      4) if ndcgs      else 0.0,
        "n_evaluated"     : len(precisions),
        "n_skipped"       : skipped,
    }
    return results
#  4. BEYOND-ACCURACY METRICS

def catalogue_coverage(
    recommend_fn  : Callable,
    all_user_ids  : list,
    all_movie_ids : list,
    k             : int = 10,
    sample_users  : int = 300,
) -> float:
    np.random.seed(42)
    sample = np.random.choice(all_user_ids, min(sample_users, len(all_user_ids)), replace=False)

    recommended_set = set()
    for uid in sample:
        try:
            recs = recommend_fn(uid, k)
            recommended_set.update(recs)
        except Exception:
            pass

    return round(len(recommended_set) / len(all_movie_ids), 4)


def intra_list_diversity(
    recommended_ids : list,
    similarity_fn   : Callable,
) -> float:
    n = len(recommended_ids)
    if n < 2:
        return 0.0

    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                sims.append(similarity_fn(recommended_ids[i], recommended_ids[j]))
            except Exception:
                pass

    avg_sim = np.mean(sims) if sims else 0.0
    return round(1.0 - avg_sim, 4)

#  5. PRETTY PRINT
def print_metrics(label: str, metrics: dict) -> None:
    bar = "─" * 50
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<25} {val:.4f}")
        else:
            print(f"  {key:<25} {val}")
    print(bar)

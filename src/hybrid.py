import numpy as np
import pandas as pd
from typing import Optional

from src.collaborative  import SVDRecommender
from src.content_based  import ContentBasedRecommender
#  1. WEIGHTED HYBRID

class WeightedHybrid:


    def __init__(self, cf_weight: float = 0.6):
        assert 0.0 <= cf_weight <= 1.0, "cf_weight must be in [0, 1]."
        self.cf_weight = cf_weight
        self.cb_weight = 1.0 - cf_weight
        self._cf  = None
        self._cb  = None
        self.is_fitted = False

    def fit(
        self,
        cf_model : SVDRecommender,
        cb_model : ContentBasedRecommender,
    ) -> "WeightedHybrid":
        if not cf_model.is_fitted or not cb_model.is_fitted:
            raise RuntimeError("Both cf_model and cb_model must be fitted first.")
        self._cf = cf_model
        self._cb = cb_model
        self.is_fitted = True
        print(f"[WeightedHybrid] Ready — CF weight={self.cf_weight:.2f}, "
              f"CB weight={self.cb_weight:.2f}.")
        return self

    def recommend(
        self,
        user_id    : int,
        n          : int          = 10,
        ratings_df : pd.DataFrame = None,
        movies_df  : pd.DataFrame = None,
        candidate_k: int          = 100,
    ) -> pd.DataFrame:
        self._check_fitted()

        # ── Determine already-seen movies for this user ────────────────────
        if ratings_df is not None:
            seen = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist())
        else:
            seen = self._cf.get_user_seen_movies(user_id)

        # ── CF candidates ──────────────────────────────────────────────────
        cf_recs = self._cf.recommend(
            user_id      = user_id,
            n            = candidate_k,
            movies_df    = movies_df,
            already_seen = seen,
        )[["movieId", "predicted_rating"]].rename(columns={"predicted_rating": "cf_raw"})

        # ── CB candidates ──────────────────────────────────────────────────
        cb_recs = self._cb.recommend(
            user_id    = user_id,
            n          = candidate_k,
            ratings_df = ratings_df,
        )[["movieId", "score"]].rename(columns={"score": "cb_raw"})

        # ── Merge on union of candidate sets ──────────────────────────────
        merged = pd.merge(cf_recs, cb_recs, on="movieId", how="outer").fillna(0)

        # ── Min-max normalise each score to [0, 1] ────────────────────────
        merged["cf_score"] = _minmax_norm(merged["cf_raw"])
        merged["cb_score"] = _minmax_norm(merged["cb_raw"])

        # ── Weighted blend ────────────────────────────────────────────────
        merged["hybrid_score"] = (
            self.cf_weight * merged["cf_score"] +
            self.cb_weight * merged["cb_score"]
        )

        # ── Top-N ─────────────────────────────────────────────────────────
        result = merged.nlargest(n, "hybrid_score").reset_index(drop=True)

        # Enrich with metadata
        if movies_df is not None:
            result = result.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )

        return result[["movieId", "title", "genres", "cf_score", "cb_score", "hybrid_score"]]


    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
#  2. SWITCHING HYBRID

class SwitchingHybrid:

    def __init__(self, warm_threshold: int = 20):
        self.warm_threshold = warm_threshold
        self._cf  = None
        self._cb  = None
        self.is_fitted = False

    def fit(self, cf_model, cb_model) -> "SwitchingHybrid":
        if not cf_model.is_fitted or not cb_model.is_fitted:
            raise RuntimeError("Both models must be fitted before passing to SwitchingHybrid.")
        self._cf = cf_model
        self._cb = cb_model
        self.is_fitted = True
        print(f"[SwitchingHybrid] Ready — warm threshold = {self.warm_threshold} ratings.")
        return self

    def recommend(
        self,
        user_id    : int,
        n          : int          = 10,
        ratings_df : pd.DataFrame = None,
        movies_df  : pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Recommend movies, switching between CF and CB based on user activity.
        """
        self._check_fitted()

        user_ratings = (
            ratings_df[ratings_df["userId"] == user_id] if ratings_df is not None
            else pd.DataFrame()
        )
        n_rated = len(user_ratings)

        if n_rated >= self.warm_threshold:
            print(f"[SwitchingHybrid] User {user_id}: {n_rated} ratings → using CF.")
            return self._cf.recommend(
                user_id      = user_id,
                n            = n,
                movies_df    = movies_df,
                already_seen = set(user_ratings["movieId"].tolist()),
            )
        else:
            print(f"[SwitchingHybrid] User {user_id}: {n_rated} ratings → using CB (cold-start).")
            return self._cb.recommend(
                user_id    = user_id,
                n          = n,
                ratings_df = ratings_df,
            )

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")


#  3. CASCADE HYBRID

class CascadeHybrid:

    def __init__(self, candidate_k: int = 50):
        self.candidate_k = candidate_k
        self._cf  = None
        self._cb  = None
        self.is_fitted = False

    def fit(self, cf_model, cb_model) -> "CascadeHybrid":
        if not cf_model.is_fitted or not cb_model.is_fitted:
            raise RuntimeError("Both models must be fitted before passing to CascadeHybrid.")
        self._cf = cf_model
        self._cb = cb_model
        self.is_fitted = True
        print(f"[CascadeHybrid] Ready — candidate pool = {self.candidate_k}.")
        return self

    def recommend(
        self,
        user_id    : int,
        n          : int          = 10,
        ratings_df : pd.DataFrame = None,
        movies_df  : pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Two-stage recommendation: CF recall → CB re-rank.
        """
        self._check_fitted()

        seen = (
            set(ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist())
            if ratings_df is not None else set()
        )

        # Stage 1: CF recall
        cf_pool = self._cf.recommend(
            user_id      = user_id,
            n            = self.candidate_k,
            movies_df    = movies_df,
            already_seen = seen,
        )

        # Stage 2: CB score for each candidate
        liked_ids = (
            ratings_df[
                (ratings_df["userId"] == user_id) &
                (ratings_df["rating"] >= self._cb.like_threshold)
            ]["movieId"].tolist()
            if ratings_df is not None else []
        )

        if not liked_ids:
            # Can't re-rank — return CF ranking as-is
            return cf_pool.head(n).reset_index(drop=True)

        # Compute CB scores for the candidate pool
        cb_pref = self._build_user_preference_vector(liked_ids)

        candidate_ids = cf_pool["movieId"].tolist()
        cb_scores = {
            mid: self._score_movie(mid, cb_pref) for mid in candidate_ids
        }

        cf_pool["cb_rerank_score"] = cf_pool["movieId"].map(cb_scores).fillna(0)

        # Re-rank by CB score (break ties with CF predicted_rating if present)
        sort_cols = (
            ["cb_rerank_score", "predicted_rating"]
            if "predicted_rating" in cf_pool.columns
            else ["cb_rerank_score"]
        )
        result = cf_pool.nlargest(n, sort_cols[0]).reset_index(drop=True)

        return result

    def _build_user_preference_vector(self, liked_ids: list) -> np.ndarray:
        """Average TF-IDF vectors of liked movies → user taste vector."""
        cb = self._cb
        vecs = []
        for mid in liked_ids:
            if mid in cb._movie2idx:
                idx = cb._movie2idx[mid]
                vecs.append(cb._tfidf_matrix[idx])
        if not vecs:
            return None
        from scipy.sparse import vstack
        return vstack(vecs).mean(axis=0)   # shape: (1, vocab)

    def _score_movie(self, movie_id: int, user_pref_vec) -> float:
        """Cosine similarity between a movie's TF-IDF vector and the user preference vector."""
        if user_pref_vec is None:
            return 0.0
        cb = self._cb
        if movie_id not in cb._movie2idx:
            return 0.0
        idx  = cb._movie2idx[movie_id]
        mvec = cb._tfidf_matrix[idx]
        from sklearn.metrics.pairwise import cosine_similarity
        return float(cosine_similarity(mvec, user_pref_vec)[0][0])

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")


#  Helper

def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]. Handles degenerate cases."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - mn) / (mx - mn)

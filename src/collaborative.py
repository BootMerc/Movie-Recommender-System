import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
#  1. PURE-SVD RECOMMENDER  (scipy svds – no extra pip install)
class SVDRecommender:

    def __init__(self, n_factors: int = 50):
        self.n_factors  = n_factors
        self.is_fitted  = False

        # Set by fit()
        self._matrix_df  = None   # full user × movie matrix (with NaNs)
        self._predicted  = None   # reconstructed dense matrix (numpy array)
        self._user_enc   = LabelEncoder()
        self._movie_enc  = LabelEncoder()
        self._user_means = None   # shape (n_users,)

    #  Training 
    def fit(self, ratings: pd.DataFrame) -> "SVDRecommender":

        print("[SVDRecommender] Fitting …")

        # Encode user/movie ids to contiguous integers
        ratings = ratings.copy()
        ratings["user_enc"]  = self._user_enc.fit_transform(ratings["userId"])
        ratings["movie_enc"] = self._movie_enc.fit_transform(ratings["movieId"])

        n_users  = ratings["user_enc"].nunique()
        n_movies = ratings["movie_enc"].nunique()

        # Build dense matrix – NaN for unrated
        self._matrix_df = ratings.pivot_table(
            index="user_enc", columns="movie_enc", values="rating"
        )
        # Per-user mean (ignoring NaN)
        self._user_means = self._matrix_df.mean(axis=1).values  # shape (n_users,)

        # Fill NaN with 0 for decomposition (after mean-centering)
        centered = self._matrix_df.sub(self._matrix_df.mean(axis=1), axis=0).fillna(0)

        # Sparse representation
        sparse = csr_matrix(centered.values)

        # Truncated SVD  (k must be < min(n_users, n_movies) - 1)
        k = min(self.n_factors, min(n_users, n_movies) - 1)
        U, sigma, Vt = svds(sparse, k=k)

        # Sort by singular values descending
        idx    = np.argsort(sigma)[::-1]
        U      = U[:, idx]
        sigma  = sigma[idx]
        Vt     = Vt[idx, :]

        # Reconstruct and add user means
        self._predicted = np.dot(np.dot(U, np.diag(sigma)), Vt) + self._user_means.reshape(-1, 1)

        # Clip to valid rating range
        self._predicted = np.clip(self._predicted, 0.5, 5.0)

        self.is_fitted = True
        print(f"[SVDRecommender] Fitted — {n_users:,} users × {n_movies:,} movies "
              f"(k={k} factors).")
        return self
    #  Prediction 
    def predict(self, user_id: int, movie_id: int) -> float:
        self._check_fitted()
        try:
            u = self._user_enc.transform([user_id])[0]
            m = self._movie_enc.transform([movie_id])[0]
            return float(self._predicted[u, m])
        except (ValueError, IndexError):
            # Cold-start fallback: return global mean
            return float(np.nanmean(self._predicted))
    #  Recommendation 
    def recommend(
        self,
        user_id       : int,
        n             : int             = 10,
        movies_df     : pd.DataFrame    = None,
        already_seen  : set             = None,
    ) -> pd.DataFrame:
        self._check_fitted()
        try:
            u = self._user_enc.transform([user_id])[0]
        except ValueError:
            raise ValueError(f"User {user_id} was not seen during training (cold-start).")

        user_preds = self._predicted[u, :]   # shape: (n_movies,)

        # Build a result frame aligned with encoded movie indices
        movie_ids = self._movie_enc.inverse_transform(np.arange(len(user_preds)))
        result_df = pd.DataFrame({
            "movieId"         : movie_ids,
            "predicted_rating": user_preds,
        })
        # Exclude already-seen movies
        if already_seen:
            result_df = result_df[~result_df["movieId"].isin(already_seen)]

        result_df = result_df.nlargest(n, "predicted_rating").reset_index(drop=True)
        # Enrich with movie metadata
        if movies_df is not None:
            result_df = result_df.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )
        return result_df
    def get_user_seen_movies(self, user_id: int) -> set:
        """Return the set of movieIds rated by user_id during training."""
        self._check_fitted()
        try:
            u = self._user_enc.transform([user_id])[0]
        except ValueError:
            return set()
        row = self._matrix_df.iloc[u]
        return set(self._movie_enc.inverse_transform(row[row.notna()].index.tolist()))

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
#  2. SURPRISE-BASED SVD  (scikit-surprise)
class SurpriseRecommender:

    def __init__(
        self,
        n_factors : int   = 100,
        n_epochs  : int   = 20,
        lr_all    : float = 0.005,
        reg_all   : float = 0.02,
    ):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr_all    = lr_all
        self.reg_all   = reg_all
        self.is_fitted  = False
        self._algo      = None
        self._trainset  = None
    def fit(self, ratings: pd.DataFrame) -> "SurpriseRecommender":
        try:
            from surprise import SVD, Dataset, Reader
        except ImportError:
            raise ImportError(
                "scikit-surprise is required. Install it with: pip install scikit-surprise"
            )
        print("[SurpriseRecommender] Fitting …")
        reader = Reader(rating_scale=(0.5, 5.0))
        data   = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
        self._trainset = data.build_full_trainset()

        self._algo = SVD(
            n_factors = self.n_factors,
            n_epochs  = self.n_epochs,
            lr_all    = self.lr_all,
            reg_all   = self.reg_all,
            verbose   = False,
        )
        self._algo.fit(self._trainset)
        self.is_fitted = True
        print(f"[SurpriseRecommender] Fitted — {self._trainset.n_users:,} users "
              f"× {self._trainset.n_items:,} movies.")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for (user_id, movie_id)."""
        self._check_fitted()
        return self._algo.predict(user_id, movie_id).est

    def recommend(
        self,
        user_id      : int,
        n            : int          = 10,
        movies_df    : pd.DataFrame = None,
        already_seen : set          = None,
    ) -> pd.DataFrame:
        """Top-N recommendations. Mirrors SVDRecommender.recommend()."""
        self._check_fitted()

        # All movies in training set
        all_items = list(self._trainset._raw2inner_id_items.keys())

        preds = []
        for mid in all_items:
            if already_seen and mid in already_seen:
                continue
            est = self._algo.predict(user_id, mid).est
            preds.append({"movieId": mid, "predicted_rating": est})
        result_df = (
            pd.DataFrame(preds)
            .nlargest(n, "predicted_rating")
            .reset_index(drop=True)
        )
        if movies_df is not None:
            result_df = result_df.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )

        return result_df
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
#  3. USER-BASED / ITEM-BASED MEMORY CF  (cosine similarity, for reference)
class MemoryCFRecommender:

    def __init__(self, mode: str = "user", n_neighbors: int = 30):
        assert mode in ("user", "item"), "mode must be 'user' or 'item'"
        self.mode        = mode
        self.n_neighbors = n_neighbors
        self.is_fitted   = False
        self._matrix_df  = None

    def fit(self, ratings: pd.DataFrame) -> "MemoryCFRecommender":
        """Build the user-item matrix and precompute similarity."""
        from sklearn.metrics.pairwise import cosine_similarity

        print(f"[MemoryCFRecommender] Fitting ({self.mode}-based) …")

        self._matrix_df = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

        if self.mode == "user":
            self._sim_matrix = cosine_similarity(self._matrix_df.values)
            self._sim_df = pd.DataFrame(
                self._sim_matrix,
                index=self._matrix_df.index,
                columns=self._matrix_df.index,
            )
        else:  # item-based
            self._sim_matrix = cosine_similarity(self._matrix_df.values.T)
            self._sim_df = pd.DataFrame(
                self._sim_matrix,
                index=self._matrix_df.columns,
                columns=self._matrix_df.columns,
            )

        self.is_fitted = True
        print(f"[MemoryCFRecommender] Fitted — similarity matrix "
              f"{self._sim_matrix.shape}.")
        return self

    def recommend(
        self,
        user_id      : int,
        n            : int          = 10,
        movies_df    : pd.DataFrame = None,
        already_seen : set          = None,
    ) -> pd.DataFrame:
        """Top-N recommendations using nearest-neighbour aggregation."""
        self._check_fitted()

        if self.mode != "user":
            raise NotImplementedError("Use item-based mode only for similar_items().")

        # Find k most similar users
        sim_row  = self._sim_df.loc[user_id].drop(user_id)
        top_k    = sim_row.nlargest(self.n_neighbors)
        neighbors = top_k.index.tolist()

        # Weighted average of neighbors' ratings
        scores = {}
        for mid in self._matrix_df.columns:
            if already_seen and mid in already_seen:
                continue
            num, denom = 0.0, 0.0
            for uid, sim in top_k.items():
                r = self._matrix_df.loc[uid, mid]
                if r > 0:
                    num   += sim * r
                    denom += abs(sim)
            if denom > 0:
                scores[mid] = num / denom

        result_df = (
            pd.DataFrame(list(scores.items()), columns=["movieId", "predicted_rating"])
            .nlargest(n, "predicted_rating")
            .reset_index(drop=True)
        )

        if movies_df is not None:
            result_df = result_df.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )

        return result_df

    def similar_items(self, movie_id: int, n: int = 10, movies_df: pd.DataFrame = None) -> pd.DataFrame:
        """Return the n most similar movies to movie_id."""
        self._check_fitted()
        if self.mode != "item":
            raise NotImplementedError("Call fit() with mode='item' first.")

        sim_row  = self._sim_df.loc[movie_id].drop(movie_id)
        top_n    = sim_row.nlargest(n).reset_index()
        top_n.columns = ["movieId", "similarity"]

        if movies_df is not None:
            top_n = top_n.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="left")

        return top_n

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

class ContentBasedRecommender:

    def __init__(
        self,
        max_features   : int   = 10_000,
        ngram_range    : tuple = (1, 2),
        like_threshold : float = 3.5,
    ):
        self.max_features   = max_features
        self.ngram_range    = ngram_range
        self.like_threshold = like_threshold
        self.is_fitted      = False

        # Set by fit()
        self._movies_df    = None        # full movies DataFrame (with content_soup)
        self._tfidf_matrix = None        # sparse (n_movies, vocab)
        self._cosine_sim   = None        # dense  (n_movies, n_movies)  [computed lazily]
        self._movie2idx    = {}          # movieId → row index in matrices
        self._idx2movie    = {}          # row index → movieId
        self._vectorizer   = None

    #  Fitting 

    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedRecommender":
        print("[ContentBasedRecommender] Fitting TF-IDF …")
        self._movies_df = movies_df.reset_index(drop=True).copy()

        # Index maps
        self._movie2idx = {mid: i for i, mid in enumerate(self._movies_df["movieId"])}
        self._idx2movie = {i: mid  for mid, i in self._movie2idx.items()}

        # TF-IDF vectorisation
        self._vectorizer = TfidfVectorizer(
            max_features = self.max_features,
            ngram_range  = self.ngram_range,
            stop_words   = "english",
            min_df       = 2,
            sublinear_tf = True,           # apply 1 + log(tf) scaling
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(
            self._movies_df["content_soup"].fillna("")
        )

        print(f"[ContentBasedRecommender] TF-IDF matrix: "
              f"{self._tfidf_matrix.shape[0]:,} movies × "
              f"{self._tfidf_matrix.shape[1]:,} features.")

        # Pre-compute full cosine similarity (fast linear_kernel for TF-IDF)
        # NOTE: For very large datasets, compute on-the-fly per query instead.
        print("[ContentBasedRecommender] Computing cosine similarity matrix …")
        self._cosine_sim = linear_kernel(self._tfidf_matrix, self._tfidf_matrix)

        self.is_fitted = True
        print("[ContentBasedRecommender] Done.")
        return self

    #  Similar movies 

    def similar_movies(
        self,
        movie_id   : int,
        n          : int = 10,
        include_scores : bool = True,
    ) -> pd.DataFrame:
        self._check_fitted()

        if movie_id not in self._movie2idx:
            raise ValueError(f"movieId {movie_id} not found in training data.")

        idx      = self._movie2idx[movie_id]
        sim_row  = self._cosine_sim[idx]                 # shape: (n_movies,)

        # Sort descending; skip self (index 0 after sorting = the query itself)
        top_indices = np.argsort(sim_row)[::-1][1 : n + 1]

        result = self._movies_df.iloc[top_indices][["movieId", "title", "genres"]].copy()

        if include_scores:
            result["similarity"] = sim_row[top_indices]

        return result.reset_index(drop=True)

    #  User recommendations 

    def recommend(
        self,
        user_id      : int,
        n            : int          = 10,
        ratings_df   : pd.DataFrame = None,
        liked_movie_ids : list      = None,
    ) -> pd.DataFrame:
        self._check_fitted()

        # Determine liked movies
        if liked_movie_ids is not None:
            liked = {mid: 5.0 for mid in liked_movie_ids}   # assume max rating
            seen  = set(liked_movie_ids)
        elif ratings_df is not None:
            user_ratings = ratings_df[ratings_df["userId"] == user_id]
            liked = (
                user_ratings[user_ratings["rating"] >= self.like_threshold]
                .set_index("movieId")["rating"]
                .to_dict()
            )
            seen  = set(user_ratings["movieId"].tolist())
        else:
            raise ValueError("Provide either ratings_df or liked_movie_ids.")

        if not liked:
            print(f"[ContentBasedRecommender] User {user_id} has no liked movies above threshold.")
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

        # Aggregate similarity scores (weighted by rating)
        agg_scores = np.zeros(len(self._movies_df))
        total_weight = 0.0

        for mid, rating in liked.items():
            if mid not in self._movie2idx:
                continue
            idx    = self._movie2idx[mid]
            weight = rating / 5.0           # normalise to [0,1]
            agg_scores += weight * self._cosine_sim[idx]
            total_weight += weight

        if total_weight > 0:
            agg_scores /= total_weight      # average

        # Build result frame
        result = self._movies_df[["movieId", "title", "genres"]].copy()
        result["score"] = agg_scores

        # Exclude already-seen movies
        result = result[~result["movieId"].isin(seen)]

        return result.nlargest(n, "score").reset_index(drop=True)

    #  Query by text 

    def recommend_by_keywords(self, keywords: str, n: int = 10) -> pd.DataFrame:
        self._check_fitted()

        query_vec = self._vectorizer.transform([keywords.lower()])
        scores    = linear_kernel(query_vec, self._tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:n]

        result = self._movies_df.iloc[top_indices][["movieId", "title", "genres"]].copy()
        result["similarity"] = scores[top_indices]

        return result.reset_index(drop=True)

    #  Genre filter 

    def get_movies_by_genre(self, genre: str, n: int = 20, movies_df: pd.DataFrame = None) -> pd.DataFrame:
        self._check_fitted()
        df = self._movies_df if movies_df is None else movies_df
        mask = df["genres"].str.contains(genre, case=False, na=False)
        return df[mask].head(n)[["movieId", "title", "genres"]].reset_index(drop=True)

    #  Internals 

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    @property
    def vocabulary_size(self) -> int:
        """Number of TF-IDF features learned."""
        self._check_fitted()
        return len(self._vectorizer.vocabulary_)

    @property
    def n_movies(self) -> int:
        """Number of movies in the model."""
        self._check_fitted()
        return len(self._movies_df)

import warnings
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

#  Page config (MUST be first Streamlit call) 
st.set_page_config(
    page_title = " Movie Recommender",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

#  Custom CSS 
st.markdown("""
<style>
    .main-title   { font-size: 2.5rem; font-weight: 800; color: #E50914; text-align: center; }
    .sub-title    { font-size: 1.1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
    .metric-card  { background: #1e1e2e; border-radius: 12px; padding: 1rem; text-align: center; }
    .movie-card   { background: #16213e; border-left: 4px solid #E50914;
                    border-radius: 8px; padding: 0.75rem 1rem; margin: 0.4rem 0; }
    .badge        { display: inline-block; background: #E50914; color: white;
                    border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; margin: 2px; }
    .score-pill   { display: inline-block; background: #0f3460; color: #e94560;
                    border-radius: 20px; padding: 2px 10px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

#  CACHED DATA & MODEL LOADING
@st.cache_data(show_spinner=" Loading MovieLens dataset …")
def load_data():
    from src.data_loader   import load_all
    from src.preprocessing import (
        clean_ratings, clean_movies, clean_tags,
        filter_by_activity, build_content_features,
        temporal_train_test_split,
    )
    data    = load_all()
    ratings = clean_ratings(data["ratings"])
    movies  = clean_movies(data["movies"])
    tags    = clean_tags(data["tags"])

    ratings = filter_by_activity(ratings, min_user_ratings=20, min_movie_ratings=10)
    train_df, test_df = temporal_train_test_split(ratings, test_ratio=0.2)
    movies_feat = build_content_features(movies, tags)

    return ratings, train_df, test_df, movies, movies_feat, tags

@st.cache_resource(show_spinner=" Training models …")
def train_models(train_df, movies_feat):
    from src.collaborative import SVDRecommender
    from src.content_based import ContentBasedRecommender
    from src.hybrid        import WeightedHybrid, CascadeHybrid

    cf_model = SVDRecommender(n_factors=50).fit(train_df)
    cb_model = ContentBasedRecommender(max_features=8_000).fit(movies_feat)
    wh_model = WeightedHybrid(cf_weight=0.65).fit(cf_model, cb_model)
    ch_model = CascadeHybrid(candidate_k=60).fit(cf_model, cb_model)

    return cf_model, cb_model, wh_model, ch_model
#  RENDERING HELPERS

def render_movie_card(title: str, genres: str, score: float = None, score_label: str = "Score"):
    genre_badges = ""
    if isinstance(genres, str):
        for g in genres.split("|")[:4]:
            genre_badges += f'<span class="badge">{g.strip()}</span> '
    score_html = ""
    if score is not None:
        score_html = f'<span class="score-pill"> {score_label}: {score:.3f}</span>'
    st.markdown(f"""
    <div class="movie-card">
        <strong style="font-size:1rem;">{title}</strong><br>
        <div style="margin: 4px 0;">{genre_badges}</div>
        {score_html}
    </div>
    """, unsafe_allow_html=True)

def render_recommendations(recs: pd.DataFrame, score_col: str, score_label: str):
    if recs.empty:
        st.warning("No recommendations found for this user.")
        return
    for _, row in recs.iterrows():
        render_movie_card(
            title       = row.get("title", f"Movie {row['movieId']}"),
            genres      = row.get("genres", ""),
            score       = row.get(score_col),
            score_label = score_label,
        )
#  MAIN APP

def main():
    #  Header 
    st.markdown('<div class="main-title"> Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Collaborative Filtering · Content-Based · Hybrid | MovieLens Dataset</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    #  Load data & models 
    with st.spinner("Initialising system …"):
        ratings, train_df, test_df, movies, movies_feat, tags = load_data()
        cf_model, cb_model, wh_model, ch_model = train_models(train_df, movies_feat)

    #  Sidebar 
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=80)
        st.title("  Controls")

        all_users = sorted(train_df["userId"].unique().tolist())
        demo_user = st.selectbox("👤 Select User ID", all_users, index=0)

        model_choice = st.selectbox(
            "Recommendation Model",
            ["Weighted Hybrid", "Collaborative Filter (SVD)", "Content-Based (TF-IDF)", "Cascade Hybrid"],
        )

        n_recs = st.slider("Number of Recommendations", 5, 20, 10)

        genre_filter = st.selectbox(
            " Filter by Genre",
            ["All"] + sorted({
                g for gs in movies["genres"].dropna()
                for g in gs.split("|") if g != "(no genres listed)"
            }),
        )

        st.divider()
        st.markdown("**Dataset Stats**")
        col1, col2 = st.columns(2)
        col1.metric("Users",   f"{ratings['userId'].nunique():,}")
        col2.metric("Movies",  f"{ratings['movieId'].nunique():,}")
        col1.metric("Ratings", f"{len(ratings):,}")
        col2.metric("Sparsity", f"{1 - len(ratings)/(ratings['userId'].nunique()*ratings['movieId'].nunique()):.1%}")

    #  Tabs 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Recommendations",
        "Similar Movies",
        "Keyword Search",
        "EDA Dashboard",
        "Evaluation",
    ])

    #  TAB 1 – PERSONALISED RECOMMENDATIONS
    with tab1:
        st.subheader(f"Top {n_recs} Recommendations for User {demo_user}")
        # User history
        user_history = (
            train_df[train_df["userId"] == demo_user]
            .merge(movies[["movieId", "title", "genres"]], on="movieId")
            .nlargest(5, "rating")
        )
        seen_ids = set(train_df[train_df["userId"] == demo_user]["movieId"])
        col_hist, col_recs = st.columns([1, 2])
        with col_hist:
            st.markdown("####  User Watch History (top-rated)")
            for _, row in user_history.iterrows():
                render_movie_card(row["title"], row["genres"], row["rating"], "Rating")
        with col_recs:
            st.markdown(f"####  {model_choice}")

            with st.spinner("Generating recommendations …"):
                try:
                    if model_choice == "Collaborative Filter (SVD)":
                        recs = cf_model.recommend(demo_user, n_recs, movies_df=movies, already_seen=seen_ids)
                        score_col, score_label = "predicted_rating", "Pred. Rating"

                    elif model_choice == "Content-Based (TF-IDF)":
                        recs = cb_model.recommend(demo_user, n_recs, ratings_df=train_df)
                        recs = recs.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
                        score_col, score_label = "score", "CB Score"

                    elif model_choice == "Weighted Hybrid":
                        recs = wh_model.recommend(demo_user, n_recs, ratings_df=train_df, movies_df=movies)
                        score_col, score_label = "hybrid_score", "Hybrid Score"

                    else:  # Cascade Hybrid
                        recs = ch_model.recommend(demo_user, n_recs, ratings_df=train_df, movies_df=movies)
                        score_col = "predicted_rating" if "predicted_rating" in recs.columns else recs.columns[-1]
                        score_label = "Score"

                    # Genre filter
                    if genre_filter != "All":
                        recs = recs[recs["genres"].str.contains(genre_filter, na=False)]

                    render_recommendations(recs, score_col, score_label)

                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")

    #  TAB 2 – SIMILAR MOVIES
    with tab2:
        st.subheader("Find Similar Movies")

        all_movie_titles = (
            movies.sort_values("title")[["movieId", "title"]]
            .assign(label=lambda df: df["title"] + " (ID: " + df["movieId"].astype(str) + ")")
        )
        chosen_label = st.selectbox(
            "Select a movie:",
            all_movie_titles["label"].tolist(),
        )
        chosen_movie_id = int(all_movie_titles.loc[all_movie_titles["label"] == chosen_label, "movieId"].values[0])
        chosen_title    = all_movie_titles.loc[all_movie_titles["label"] == chosen_label, "title"].values[0]

        n_similar = st.slider("Number of similar movies", 5, 20, 10, key="sim_slider")

        if st.button("Find Similar Movies", type="primary"):
            with st.spinner("Computing similarity …"):
                try:
                    similar = cb_model.similar_movies(chosen_movie_id, n=n_similar)
                    st.markdown(f"#### Movies similar to **{chosen_title}**")
                    for _, row in similar.iterrows():
                        render_movie_card(row["title"], row["genres"], row["similarity"], "Similarity")
                except Exception as e:
                    st.error(f"Error: {e}")
    #  TAB 3 – KEYWORD SEARCH
    with tab3:
        st.subheader("🔎 Search Movies by Keywords")
        st.markdown("Enter genres, themes, or descriptive words to find matching movies.")

        example_queries = [
            "space sci-fi adventure",
            "romantic comedy love",
            "dark thriller crime mystery",
            "animated family children",
            "historical war drama",
        ]
        st.markdown("**Example queries:** " + " · ".join(f"`{q}`" for q in example_queries))

        keyword_input = st.text_input("Enter keywords:", placeholder="e.g. sci-fi space adventure")
        n_kw = st.slider("Results to show", 5, 20, 10, key="kw_slider")

        if st.button("🔎 Search", type="primary") and keyword_input.strip():
            with st.spinner("Searching …"):
                results = cb_model.recommend_by_keywords(keyword_input.strip(), n=n_kw)
                if results.empty:
                    st.warning("No results found. Try different keywords.")
                else:
                    for _, row in results.iterrows():
                        render_movie_card(row["title"], row["genres"], row["similarity"], "Match Score")
    #  TAB 4 – EDA DASHBOARD
    with tab4:
        st.subheader("Exploratory Data Analysis")

        # Row 1: Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Ratings",   f"{len(ratings):,}")
        c2.metric("Unique Users",    f"{ratings['userId'].nunique():,}")
        c3.metric("Unique Movies",   f"{ratings['movieId'].nunique():,}")
        c4.metric("Avg Rating",      f"{ratings['rating'].mean():.2f}")
        c5.metric("Rating Std Dev",  f"{ratings['rating'].std():.2f}")

        st.divider()
        col_a, col_b = st.columns(2)

        # Rating distribution
        with col_a:
            st.markdown("#### Rating Distribution")
            rating_counts = ratings["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["Rating", "Count"]
            fig = px.bar(
                rating_counts, x="Rating", y="Count",
                color="Count", color_continuous_scale="Reds",
                title="How Users Rate Movies",
            )
            fig.update_layout(showlegend=False, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        # Ratings per user distribution
        with col_b:
            st.markdown("#### Ratings per User")
            rpu = ratings.groupby("userId")["movieId"].count().reset_index()
            rpu.columns = ["userId", "n_ratings"]
            fig2 = px.histogram(
                rpu, x="n_ratings", nbins=40,
                title="How Many Movies Users Rate",
                color_discrete_sequence=["#E50914"],
            )
            fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)

        # Top 20 movies
        with col_c:
            st.markdown("#### Top 20 Most-Rated Movies")
            top_mov = (
                ratings.groupby("movieId")["rating"]
                .agg(count="count", mean_rating="mean")
                .reset_index()
                .merge(movies[["movieId", "title"]], on="movieId")
                .nlargest(20, "count")
            )
            fig3 = px.bar(
                top_mov, x="count", y="title", orientation="h",
                color="mean_rating", color_continuous_scale="RdYlGn",
                title="Most-Rated Movies (color = avg rating)",
            )
            fig3.update_layout(
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Genre popularity
        with col_d:
            st.markdown("#### Genre Popularity")
            genre_counts = (
                movies["genres"]
                .str.split("|")
                .explode()
                .value_counts()
                .drop("(no genres listed)", errors="ignore")
                .head(15)
                .reset_index()
            )
            genre_counts.columns = ["Genre", "Count"]
            fig4 = px.bar(
                genre_counts, x="Genre", y="Count",
                color="Count", color_continuous_scale="Blues",
                title="Number of Movies per Genre",
            )
            fig4.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Rating over time heatmap
        st.markdown("#### Rating Activity Over Time")
        ratings_time = ratings.copy()
        ratings_time["year"]  = pd.to_datetime(ratings_time["timestamp"], unit="s").dt.year
        ratings_time["month"] = pd.to_datetime(ratings_time["timestamp"], unit="s").dt.month
        heatmap_data = (
            ratings_time.groupby(["year", "month"])["rating"]
            .count()
            .reset_index()
            .pivot(index="year", columns="month", values="rating")
            .fillna(0)
        )
        fig5 = px.imshow(
            heatmap_data,
            color_continuous_scale="Reds",
            title="Number of Ratings by Year × Month",
            labels={"x": "Month", "y": "Year", "color": "Ratings"},
            aspect="auto",
        )
        fig5.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig5, use_container_width=True)

    #  TAB 5 – EVALUATION
    with tab5:
        st.subheader("Model Evaluation Metrics")

        st.info(
            "📌 **Evaluation is computed lazily** on a sample of the test set for speed.\n"
            "Click the button below to run the full evaluation pipeline."
        )

        k_eval = st.slider("Evaluation cutoff K", 5, 20, 10)

        if st.button("▶ Run Evaluation", type="primary"):
            from src.evaluation import (
                evaluate_rating_prediction, evaluate_ranking, catalogue_coverage
            )

            with st.spinner("Computing RMSE / MAE …"):
                rating_metrics = evaluate_rating_prediction(
                    test_df, predict_fn=cf_model.predict, sample_n=2_000
                )

            with st.spinner(f"Computing ranking metrics @{k_eval} …"):
                seen_cache = {
                    uid: set(train_df[train_df["userId"] == uid]["movieId"])
                    for uid in train_df["userId"].unique()
                }

                def cf_fn(uid, n):
                    return cf_model.recommend(uid, n, already_seen=seen_cache.get(uid, set()))["movieId"].tolist()

                def cb_fn(uid, n):
                    return cb_model.recommend(uid, n, ratings_df=train_df)["movieId"].tolist()

                def hy_fn(uid, n):
                    return wh_model.recommend(uid, n, ratings_df=train_df, movies_df=movies)["movieId"].tolist()

                r_cf = evaluate_ranking(test_df, cf_fn, k=k_eval, n_users=100)
                r_cb = evaluate_ranking(test_df, cb_fn, k=k_eval, n_users=100)
                r_hy = evaluate_ranking(test_df, hy_fn, k=k_eval, n_users=100)

            # Display results
            st.markdown("#### Rating Prediction Quality (SVD)")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rating_metrics['rmse']:.4f}", help="Lower is better")
            col2.metric("MAE",  f"{rating_metrics['mae']:.4f}",  help="Lower is better")
            col3.metric("Evaluated Pairs", f"{rating_metrics['n_evaluated']:,}")

            st.divider()
            st.markdown(f"#### Ranking Metrics @{k_eval}")

            models = ["CF (SVD)", "Content-Based", "Weighted Hybrid"]
            results = [r_cf, r_cb, r_hy]
            comparison = pd.DataFrame({
                "Model"             : models,
                f"Precision@{k_eval}": [r[f"precision@{k_eval}"] for r in results],
                f"Recall@{k_eval}"  : [r[f"recall@{k_eval}"]    for r in results],
                f"F1@{k_eval}"      : [r[f"f1@{k_eval}"]        for r in results],
                f"NDCG@{k_eval}"    : [r[f"ndcg@{k_eval}"]      for r in results],
            })
            st.dataframe(comparison.set_index("Model"), use_container_width=True)

            # Radar / bar comparison chart
            fig = go.Figure()
            metrics_cols = [f"Precision@{k_eval}", f"Recall@{k_eval}", f"NDCG@{k_eval}"]
            colors = ["#E50914", "#00b4d8", "#06d6a0"]
            for i, (model, res) in enumerate(zip(models, results)):
                fig.add_trace(go.Bar(
                    name=model,
                    x=metrics_cols,
                    y=[res[f"precision@{k_eval}"], res[f"recall@{k_eval}"], res[f"ndcg@{k_eval}"]],
                    marker_color=colors[i],
                ))
            fig.update_layout(
                barmode="group",
                title=f"Model Comparison @{k_eval}",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                legend=dict(bgcolor="#1e1e2e"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metric explanations
            with st.expander("📖 What do these metrics mean?"):
                st.markdown("""
| Metric | Definition | Ideal |
|---|---|---|
| **RMSE** | Root Mean Square Error of rating predictions. Penalises large errors more. | Lower ↓ |
| **MAE** | Mean Absolute Error of rating predictions. Easier to interpret. | Lower ↓ |
| **Precision@K** | Of the K items recommended, what fraction did the user like? | Higher ↑ |
| **Recall@K** | Of all items the user likes, what fraction were in the top-K? | Higher ↑ |
| **F1@K** | Harmonic mean of Precision@K and Recall@K. | Higher ↑ |
| **NDCG@K** | Rewards placing relevant items earlier in the ranking. 1.0 = perfect. | Higher ↑ |
                """)
if __name__ == "__main__":
    main()
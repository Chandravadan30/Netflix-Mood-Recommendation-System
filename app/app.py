# Streamlit explorer for the Netflix Mood Map (Spark + Parquet backend)

import streamlit as st
from pyspark.sql import SparkSession, functions as F
import altair as alt  # charts


st.set_page_config(
    page_title="Netflix Mood Explorer",
    layout="wide",
    
)

st.title("Netflix Mood-Based Recommendation Explorer")
st.markdown("Explore clusters of movies grouped by sentiment, genre, and release period.")

# Spark session

@st.cache_resource(show_spinner=False)
def get_spark():
    return (
        SparkSession.builder
        .appName("Netflix_Mood_Explorer")
        .master("local[*]")
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .getOrCreate()
    )

spark = get_spark()

# Load clustered predictions
@st.cache_data(show_spinner=True)
def load_dataframe():
    df = spark.read.parquet("/projects/netflix_mood/models/kmeans_predictions")
    cols = ["nf_title", "nf_description", "mood", "main_cat",
            "release_year", "cluster", "compound"]
    existing = [c for c in cols if c in df.columns]
    df = df.select(*existing)
    df = df.filter(F.col("nf_title").isNotNull())
    pdf = df.toPandas()

    if "release_year" in pdf.columns:
        pdf["release_year"] = pdf["release_year"].fillna(0).astype("int64")
    if "compound" in pdf.columns:
        pdf["compound"] = pdf["compound"].astype(float)
    if "main_cat" in pdf.columns:
        pdf["main_cat"] = pdf["main_cat"].fillna("Unknown")

    sort_cols = [c for c in ["main_cat", "release_year", "compound"]
                 if c in pdf.columns]
    if sort_cols:
        pdf = pdf.sort_values(sort_cols).reset_index(drop=True)
    return pdf


pdf = load_dataframe()

if pdf.empty:
    st.warning(
        "No data found at /projects/netflix_mood/models/kmeans_predictions. "
        "Run the training pipeline first (04_build_features.py ➜ 10_train_kmeans.py)."
    )
    st.stop()

if "main_cat" not in pdf.columns:
    st.error(
        "Fatal Error: The 'main_cat' column is missing from the Parquet file. "
        "This column is required for the Genre filter."
    )
    st.stop()


# Sidebar controller

with st.sidebar:
    st.subheader("Select Genre")

    genre_list = sorted(pdf["main_cat"].dropna().unique().tolist())
    selected_genre = st.selectbox(" ", genre_list, index=0)

    st.markdown(f"**Movies in this genre:** "
                f"{(pdf['main_cat'] == selected_genre).sum():,}")

    st.divider()
    st.subheader("Filters")

    if "release_year" in pdf.columns and pdf["release_year"].gt(0).any():
        yr_min = int(pdf.loc[pdf["release_year"] > 0, "release_year"].min())
        yr_max = int(pdf["release_year"].max())
        r_lo, r_hi = st.slider(
            "Release Year",
            min_value=yr_min,
            max_value=yr_max,
            value=(yr_min, yr_max),
            step=1,
        )
    else:
        r_lo, r_hi = (0, 3000)

    title_query = st.text_input("Search title contains", value="").strip()

    st.subheader("Ranking")
    top_n = st.slider("Top N", 10, 500, 50, step=10)

# Genre filter 
def filtered_genre_frame():
    dfc = pdf[pdf["main_cat"] == selected_genre].copy()

    if "release_year" in dfc.columns:
        dfc = dfc[(dfc["release_year"] >= r_lo) & (dfc["release_year"] <= r_hi)]

    if title_query:
        dfc = dfc[dfc["nf_title"].str.contains(title_query, case=False, na=False)]

    if "compound" in dfc.columns:
        dfc["abs_comp"] = dfc["compound"].abs().fillna(0.0)
    else:
        dfc["abs_comp"] = 0.0

    if "release_year" not in dfc.columns:
        dfc["release_year"] = 0

    dfc = dfc.sort_values(
        ["abs_comp", "release_year"],
        ascending=[False, False]
    ).head(top_n)

    return dfc


genre_movies = filtered_genre_frame()

# Header & Key metrics

st.markdown(f"### {selected_genre} — Representative Movies")

kpi_cols = st.columns(3)
with kpi_cols[0]:
    st.metric("Rows shown", f"{len(genre_movies):,}")
with kpi_cols[1]:
    if "compound" in genre_movies.columns and len(genre_movies):
        st.metric("Avg sentiment", f"{genre_movies['compound'].mean():0.3f}")
    else:
        st.metric("Avg sentiment", "—")
with kpi_cols[2]:
    if "release_year" in genre_movies.columns and genre_movies["release_year"].gt(0).any():
        st.metric("Newest year", int(genre_movies["release_year"].max()))
    else:
        st.metric("Newest year", "—")

st.divider()  # Visual separator

st.markdown("### Visual Insights")

if genre_movies.empty:
    st.info("No data to display for the selected filters.")
else:
    viz_cols = st.columns(2)
    with viz_cols[0]:
        st.markdown("#### Sentiment Distribution")
        if "compound" in genre_movies.columns:
            c = alt.Chart(genre_movies).mark_bar().encode(
                alt.X("compound", bin=alt.Bin(maxbins=20), title="Sentiment Score"),
                alt.Y("count()", title="Number of Movies"),
                tooltip=[alt.X("compound", bin=alt.Bin(maxbins=20)), "count()"]
            ).interactive()
            st.altair_chart(c, use_container_width=True)
        else:
            st.caption("No sentiment data available.")

    with viz_cols[1]:
        st.markdown("#### Movies by Release Year")
        if "release_year" in genre_movies.columns and genre_movies["release_year"].gt(0).any():
            chart_data = genre_movies[genre_movies["release_year"] > 0]
            c = alt.Chart(chart_data).mark_bar().encode(
                alt.X("release_year", bin=alt.Bin(maxbins=20), title="Year"),
                alt.Y("count()", title="Number of Movies"),
                tooltip=[alt.X("release_year", bin=alt.Bin(maxbins=20)), "count()"]
            ).interactive()
            st.altair_chart(c, use_container_width=True)
        else:
            st.caption("No release year data available.")

st.divider()  # Visual separator

st.markdown("### Filtered Movie List")

col_config = {
    "nf_title": st.column_config.TextColumn(
        "Title",
        width="large",
        help="The title of the movie/show"
    ),
    "main_cat": st.column_config.TextColumn("Genre"),
    "release_year": st.column_config.NumberColumn("Year", format="%d"),
    "mood": st.column_config.TextColumn("Mood"),
    "compound": st.column_config.ProgressColumn(
        "Sentiment Score",
        help="VADER sentiment score from -1 (Negative) to 1 (Positive)",
        format="%.3f",
        min_value=-1.0,
        max_value=1.0,
    ),
}

final_col_config = {k: v for k, v in col_config.items() if k in genre_movies.columns}
display_cols_order = ["nf_title", "main_cat", "release_year", "mood", "compound"]
final_display_cols = [c for c in display_cols_order if c in genre_movies.columns]

st.dataframe(
    genre_movies[final_display_cols],
    use_container_width=True,
    column_config=final_col_config,
    height=500,
    hide_index=True,
)


st.caption(
    "Notes: **compound** is the VADER sentiment score in [-1, 1] (negative → positive). "
)


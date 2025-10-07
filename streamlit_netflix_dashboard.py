import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Page config and CSS
# -------------------------
st.set_page_config(page_title="Netflix Analytics — Insight Dashboard",
                   page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* Light theme */
    .main { background-color: #ffffff; color: #222222; }
    h1,h2,h3,h4 { color: #e50914; font-family: "Segoe UI", Roboto, sans-serif; }
    [data-testid="stSidebar"] { background-color: #f7f7f8; border-right: 1px solid #e6e6e6; }
    .metric-card { background: #fff; border-radius: 10px; padding: 14px; border: 1px solid #eee; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
    .kpi { font-size: 22px; font-weight: 700; color: #111; }
    .kpi-sub { font-size: 12px; color: #666; }
    .small { font-size:12px; color:#666; }
    .stButton>button { background-color: #e50914; color: white; border-radius: 8px; border: none; padding: 6px 12px; }
    .stButton>button:hover { background-color:#b00710; }
    .dataframe { border: 1px solid #eee; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Utility: load & preprocess
# -------------------------
@st.cache_data
def load_data(path="Netflix Dataset.csv"):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # unify common names
    # find date column
    date_col = None
    for cand in ["Release_Date", "Release date", "Date", "Date added", "date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Release_Year"] = df[date_col].dt.year
        df["Release_Month"] = df[date_col].dt.month
    else:
        # fallback to 'Year' column
        if "Year" in df.columns:
            df["Release_Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
            df["Release_Month"] = pd.NA
        else:
            df["Release_Year"] = pd.NA
            df["Release_Month"] = pd.NA

    # Category detection (Movie / TV Show)
    if "Category" not in df.columns:
        if "Type" in df.columns and df["Type"].str.contains("Movie|TV Show", na=False).any():
            df.rename(columns={"Type": "Category"}, inplace=True)
        elif "Show" in df.columns and "Movie" in df.columns:
            # unlikely, but try to infer
            df["Category"] = df.get("Category", "Unknown")
        else:
            # fill with Unknown
            df["Category"] = df.get("Category", "Unknown")

    # Ensure Title column
    if "Title" not in df.columns:
        possible = [c for c in df.columns if "title" in c.lower() or "show" in c.lower()]
        if possible:
            df.rename(columns={possible[0]: "Title"}, inplace=True)
        else:
            df["Title"] = df["Title"] if "Title" in df.columns else np.nan

    # Fill key text columns
    for col in ["Director", "Cast", "Country", "Rating", "Genre", "Type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    # Primary country
    if "Country" in df.columns:
        df["Primary_Country"] = df["Country"].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown")
    else:
        df["Primary_Country"] = "Unknown"

    # Genre parsing: prefer 'Genre' then 'Type'
    if "Genre" in df.columns:
        df["Genre_List"] = df["Genre"].astype(str).apply(lambda s: [g.strip() for g in s.split(",")] if s and s != "nan" else [])
    elif "Type" in df.columns:
        df["Genre_List"] = df["Type"].astype(str).apply(lambda s: [g.strip() for g in s.split(",")] if s and s != "nan" else [])
    else:
        df["Genre_List"] = [[] for _ in range(len(df))]
    df["Primary_Genre"] = df["Genre_List"].apply(lambda lst: lst[0] if lst else "Unknown")

    # Duration numeric extraction
    if "Duration" in df.columns:
        import re
        def parse_duration(v):
            if pd.isna(v): return np.nan
            m = re.search(r'(\d+)', str(v))
            return float(m.group(1)) if m else np.nan
        df["Duration_Min"] = df["Duration"].apply(parse_duration)
    else:
        df["Duration_Min"] = np.nan

    # Flags
    df["Is_Movie"] = df["Category"].str.contains("Movie", na=False)
    df["Is_TV"] = df["Category"].str.contains("TV Show|TV", na=False)

    # Decade column
    try:
        df["Decade"] = (df["Release_Year"] // 10 * 10).astype("Int64")
    except Exception:
        df["Decade"] = pd.NA

    return df

# -------------------------
# Load data
# -------------------------
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    data_loaded = False

if not data_loaded:
    st.stop()

# -------------------------
# Sidebar: filters
# -------------------------
st.sidebar.header("Filters & Search")
min_year = int(df["Release_Year"].dropna().min()) if df["Release_Year"].notna().any() else 2000
max_year = int(df["Release_Year"].dropna().max()) if df["Release_Year"].notna().any() else datetime.now().year
year_range = st.sidebar.slider("Release year range", min_year, max_year, (min_year, max_year))

categories = df["Category"].dropna().unique().tolist() if "Category" in df.columns else ["Movie", "TV Show"]
sel_categories = st.sidebar.multiselect("Category", options=categories, default=categories)

top_genres = df["Primary_Genre"].value_counts().head(40).index.tolist()
sel_genres = st.sidebar.multiselect("Primary genres", options=top_genres, default=top_genres[:6])

top_countries = df["Primary_Country"].value_counts().head(80).index.tolist()
sel_countries = st.sidebar.multiselect("Countries (primary)", options=top_countries, default=top_countries[:6])

search_title = st.sidebar.text_input("Search title contains")
search_director = st.sidebar.text_input("Search director contains")

top_n = st.sidebar.slider("Top N", 5, 50, 12)

# -------------------------
# Apply filters
# -------------------------
df_filtered = df.copy()
if "Release_Year" in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered["Release_Year"].fillna(0) >= year_range[0]) &
                              (df_filtered["Release_Year"].fillna(max_year) <= year_range[1])]
if sel_categories:
    df_filtered = df_filtered[df_filtered["Category"].isin(sel_categories)]
if sel_genres:
    df_filtered = df_filtered[df_filtered["Primary_Genre"].isin(sel_genres)]
if sel_countries:
    df_filtered = df_filtered[df_filtered["Primary_Country"].isin(sel_countries)]
if search_title:
    df_filtered = df_filtered[df_filtered["Title"].str.contains(search_title, case=False, na=False)]
if search_director:
    if "Director" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Director"].str.contains(search_director, case=False, na=False)]

# -------------------------
# Top KPI row
# -------------------------
st.title("🎬 Netflix Content Analytics — Insight Dashboard")
st.markdown("**Focused, unique visualizations to surface strategic content insights.**")
st.markdown("---")

k1, k2, k3, k4 = st.columns([1.5, 1, 1, 1])
with k1:
    st.markdown("<div class='metric-card'><div class='kpi'>{:,}</div><div class='kpi-sub'>Total titles (filtered)</div></div>", unsafe_allow_html=True)
with k2:
    movies = int(df_filtered["Is_Movie"].sum()) if "Is_Movie" in df_filtered.columns else (df_filtered["Category"]=="Movie").sum()
    st.markdown(f"<div class='metric-card'><div class='kpi'>{movies:,}</div><div class='kpi-sub'>Movies</div></div>", unsafe_allow_html=True)
with k3:
    tvs = int(df_filtered["Is_TV"].sum()) if "Is_TV" in df_filtered.columns else (df_filtered["Category"]=="TV Show").sum()
    st.markdown(f"<div class='metric-card'><div class='kpi'>{tvs:,}</div><div class='kpi-sub'>TV Shows</div></div>", unsafe_allow_html=True)
with k4:
    uniq_c = int(df_filtered["Primary_Country"].nunique())
    st.markdown(f"<div class='metric-card'><div class='kpi'>{uniq_c}</div><div class='kpi-sub'>Countries</div></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Row 1: Trend + Seasonality heatmap
# -------------------------
row1_col1, row1_col2 = st.columns((2,1))

with row1_col1:
    st.subheader("Yearly Trend — Movies vs TV")
    if "Release_Year" in df_filtered.columns and "Category" in df_filtered.columns:
        trend = df_filtered.groupby(["Release_Year", "Category"]).size().unstack(fill_value=0).sort_index()
        if trend.shape[0] == 0:
            st.info("No data for selected years")
        else:
            fig = go.Figure()
            for c in trend.columns:
                fig.add_trace(go.Scatter(x=trend.index, y=trend[c], mode="lines+markers", name=str(c)))
            fig.update_layout(title="Titles released per year by Category", xaxis_title="Year", yaxis_title="Count", height=420)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Missing Release_Year or Category.")

with row1_col2:
    st.subheader("Seasonality: Releases (Month × Year)")
    if "Release_Month" in df_filtered.columns and "Release_Year" in df_filtered.columns:
        pivot = df_filtered.pivot_table(index="Release_Month", columns="Release_Year", values="Title", aggfunc="count").fillna(0)
        if pivot.size == 0:
            st.info("Not enough date data.")
        else:
            # reorder months
            pivot = pivot.reindex(index=range(1,13), fill_value=0)
            fig = px.imshow(pivot,
                            labels=dict(x="Year", y="Month", color="Count"),
                            x=pivot.columns, y=pivot.index,
                            aspect="auto",
                            title="Heatmap: Titles by Month (rows) vs Year (columns)",
                            height=420)
            fig.update_yaxes(tickmode="array", tickvals=list(range(1,13)), ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Release month/year not available.")

st.markdown("---")

# -------------------------
# Row 2: Genre Country Sunburst + Treemap
# -------------------------
r2c1, r2c2 = st.columns((1.4,1))
with r2c1:
    st.subheader("Genre × Country — Hierarchical Contribution (Sunburst)")
    # prepare sunburst df: Primary_Genre -> Primary_Country -> Count
    sun = df_filtered.groupby(["Primary_Genre", "Primary_Country"]).size().reset_index(name="Count")
    if sun.shape[0] == 0:
        st.info("No genre/country data for selection.")
    else:
        fig = px.sunburst(sun, path=["Primary_Genre", "Primary_Country"], values="Count",
                          title="Primary Genre → Country: Composition", height=520)
        st.plotly_chart(fig, use_container_width=True)

with r2c2:
    st.subheader("Top Genres by Share (Treemap)")
    genre_counts = df_filtered["Primary_Genre"].value_counts().reset_index()
    genre_counts.columns = ["Genre", "Count"]
    if genre_counts.shape[0] == 0:
        st.info("No genre data.")
    else:
        treemap_top = genre_counts.head(50)
        fig = px.treemap(treemap_top, path=["Genre"], values="Count", title="Treemap: Genre share (top)", height=520)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -------------------------
# Row 3: Directors bubble + Duration violin
# -------------------------
r3c1, r3c2 = st.columns((1.2,1))
with r3c1:
    st.subheader(f"Top {top_n} Directors (bubble = #titles, color = avg duration)")
    if "Director" in df_filtered.columns:
        dir_stats = df_filtered.groupby("Director").agg(
            titles=("Title","count"),
            avg_duration=("Duration_Min","mean")
        ).reset_index().sort_values("titles", ascending=False).head(top_n)
        if dir_stats.shape[0] == 0:
            st.info("No director metadata.")
        else:
            # fill avg_duration for plotting
            dir_stats["avg_duration"] = dir_stats["avg_duration"].fillna(dir_stats["avg_duration"].median())
            fig = px.scatter(dir_stats, x="titles", y="Director", size="titles", color="avg_duration",
                             color_continuous_scale="reds", title="Director leaderboard (size=#titles)", height=420,
                             labels={"titles":"# Titles", "avg_duration":"Avg duration"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Director column missing.")

with r3c2:
    st.subheader("Duration distribution by Category (Violin)")
    if df_filtered["Duration_Min"].notna().sum() > 0:
        # keep categories with enough samples
        vc = df_filtered["Category"].value_counts()
        cats = vc[vc >= 5].index.tolist()
        if not cats:
            st.info("Not enough samples per category for violin plot.")
        else:
            plot_df = df_filtered[df_filtered["Category"].isin(cats)]
            fig = px.violin(plot_df, x="Category", y="Duration_Min", box=True, points="outliers",
                            title="Duration distribution (numeric part) by Category", height=420)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric duration data.")

st.markdown("---")

# -------------------------
# Row 4: Rating insights + Rating × Genre stacked bar
# -------------------------
r4c1, r4c2 = st.columns((1,1))
with r4c1:
    st.subheader("Rating distribution (Top ratings)")
    rating_counts = df_filtered["Rating"].value_counts().head(12).reset_index()
    rating_counts.columns = ["Rating", "Count"]
    if rating_counts.shape[0] == 0:
        st.info("No rating data.")
    else:
        fig = px.bar(rating_counts, x="Count", y="Rating", orientation="h", title="Top Ratings by count", height=420)
        st.plotly_chart(fig, use_container_width=True)

with r4c2:
    st.subheader("Rating × Genre (stacked) — top genres")
    # stacked counts for top genres
    topg = df_filtered["Primary_Genre"].value_counts().head(8).index.tolist()
    if len(topg) == 0:
        st.info("No genre data.")
    else:
        stacked = df_filtered[df_filtered["Primary_Genre"].isin(topg)].groupby(["Primary_Genre", "Rating"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for r in stacked.columns:
            fig.add_trace(go.Bar(x=stacked.index, y=stacked[r], name=str(r)))
        fig.update_layout(barmode="stack", title="Rating composition across top genres", xaxis_title="Genre", yaxis_title="Count", height=420)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -------------------------
# Numeric correlations
# -------------------------
st.subheader("Numeric Feature Relationships (Correlation Heatmap)")
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 2:
    corr = df_filtered[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=False)
    ax.set_title("Correlation matrix (numeric features)")
    st.pyplot(fig)
else:
    st.info("Not enough numeric fields to compute correlations.")

st.markdown("---")

# -------------------------
# Data preview & download
# -------------------------
st.subheader("Filtered dataset preview and export")
st.write(f"Rows: {len(df_filtered):,} — Columns: {df_filtered.shape[1]}")
st.dataframe(df_filtered.head(200))

def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8")

st.download_button("Download filtered CSV", to_csv_bytes(df_filtered), file_name="netflix_filtered.csv", mime="text/csv")

st.markdown("---")

# -------------------------
# Automated high-level insights
# -------------------------
st.header("🔎 Strategic Insights (high-level)")
insights = []
total = len(df_filtered)
if total > 0:
    movies_pct = (movies / total) * 100 if total else 0
    tv_pct = (tvs / total) * 100 if total else 0
    insights.append(f"Content mix — Movies: {movies_pct:.1f}% | TV Shows: {tv_pct:.1f}% of filtered catalog.")

    if "Primary_Genre" in df_filtered.columns:
        top_gen = df_filtered["Primary_Genre"].value_counts().idxmax()
        top_gen_share = df_filtered["Primary_Genre"].value_counts().max() / total * 100
        insights.append(f"Top genre: **{top_gen}** ({top_gen_share:.1f}% of filtered titles).")

    top_country = df_filtered["Primary_Country"].value_counts().idxmax() if df_filtered["Primary_Country"].notna().any() else None
    if top_country:
        top_country_share = df_filtered["Primary_Country"].value_counts().max() / total * 100
        insights.append(f"Dominant production country: **{top_country}** ({top_country_share:.1f}%). Consider investing in underrepresented regions.")

    # Duration insight
    if df_filtered["Duration_Min"].notna().sum() > 0:
        median_dur = df_filtered["Duration_Min"].median()
        insights.append(f"Median numeric duration = {median_dur:.0f} (units extracted from 'Duration' column).")

for it in insights:
    st.markdown(f"- {it}")

st.caption("This dashboard focuses on distinct, interpretable visualizations — replace field names in `load_data()` if your CSV uses different column headers.")

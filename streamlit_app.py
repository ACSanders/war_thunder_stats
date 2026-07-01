from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

import features


# ============================================================
# App config
# ============================================================

st.set_page_config(
    page_title="War Thunder Stats",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Larger, slightly heavier main tab labels (scoped to the tab list only so it
# does not affect buttons, dropdowns, or other widgets).
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button p {
        font-size: 1.15rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_URL = (
    "https://raw.githubusercontent.com/ACSanders/war_thunder_stats/"
    "main/data/processed/ground_realistic_30_days_latest.csv"
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data(ttl=60 * 60)
def load_data(url: str) -> pd.DataFrame:
    """Fetch the raw CSV from GitHub. Parse the date so date-range readouts work;
    all deeper typing / cleaning happens in features.clean_daily."""
    df = pd.read_csv(url)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# Cached wrappers around the pure helpers in features.py. Caching is kept here
# (Streamlit-specific); the data logic lives in the importable module.

@st.cache_data(ttl=60 * 60)
def get_cleaned_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    """cleaned_daily_df: one typed, backfilled row per vehicle per day."""
    return features.clean_daily(raw_df)


@st.cache_data(ttl=60 * 60)
def get_recent_daily(cleaned_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Most recent 30-day window of cleaned_daily_df."""
    return features.recent_window(cleaned_daily_df, days=features.RECENT_WINDOW_DAYS)


@st.cache_data(ttl=60 * 60)
def get_vehicle_agg(recent_daily_df: pd.DataFrame) -> pd.DataFrame:
    """vehicle_agg_df: one row per vehicle with the Combat Effectiveness Score,
    broad BR range fields, and quality flags."""
    vehicle_df = features.build_vehicle_agg(recent_daily_df)
    vehicle_df = features.add_quality_flags(vehicle_df)
    vehicle_df = features.add_combat_effectiveness(vehicle_df)
    vehicle_df = features.add_combat_effectiveness_legacy(vehicle_df)
    vehicle_df = features.add_br_ranges(vehicle_df)
    return vehicle_df


@st.cache_data(ttl=60 * 60)
def get_clusters(vehicle_df: pd.DataFrame, min_sample_battles: int):
    """Cluster the given (already filtered) vehicle slice and attach friendly
    archetype labels. Cached on the slice contents + control value."""
    clustered, meta = features.build_vehicle_clusters(
        vehicle_df, min_sample_battles=min_sample_battles
    )
    if not clustered.empty and "cluster_id" in clustered.columns:
        clustered = features.label_clusters(clustered)
    return clustered, meta


# Data preparation, vehicle aggregation, scoring, and nation/BR aggregates now
# live in features.py (pure pandas, reusable by offline scripts). The cached
# wrappers above expose them to the app.


# ============================================================
# Image rendering
# ============================================================

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_image_bytes(image_url):
    """
    Fetch image bytes server-side.

    This is more reliable than embedding ThunderSkill image URLs directly,
    because the Streamlit app serves the fetched image instead of asking the
    browser to hotlink it.
    """
    if pd.isna(image_url) or not str(image_url).startswith("http"):
        return None, "Missing or invalid image URL"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://thunderskill.com/",
        "Connection": "keep-alive",
    }

    try:
        response = requests.get(
            str(image_url),
            headers=headers,
            timeout=15,
            allow_redirects=True,
        )

        content_type = response.headers.get("Content-Type", "")
        content_length = len(response.content) if response.content else 0

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}; content-type={content_type}; bytes={content_length}"

        if "image" not in content_type.lower():
            return None, f"Not an image; content-type={content_type}; bytes={content_length}"

        if content_length == 0:
            return None, "Image response had 0 bytes"

        return response.content, f"OK; content-type={content_type}; bytes={content_length}"

    except requests.RequestException as e:
        return None, f"Request failed: {e}"


# ============================================================
# Load and prepare data
# ============================================================

try:
    raw_df = load_data(DATA_URL)
except Exception as e:
    st.error("Could not load the latest data from GitHub.")
    st.exception(e)
    st.stop()

cleaned_daily_df = get_cleaned_daily(raw_df)
recent_df = get_recent_daily(cleaned_daily_df)
vehicle_30d_df = get_vehicle_agg(recent_df)

if vehicle_30d_df.empty:
    st.error("No vehicle data available after recent-date filtering.")
    st.stop()


# ============================================================
# Header
# ============================================================

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "war_thunder_stats_logo1.png"

logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=88)
with title_col:
    st.title("War Thunder Stats")
    st.caption("Realistic Ground Forces meta, driven by player performance.")


# ============================================================
# Top filter deck  (replaces the old sidebar)
# ============================================================

def _br_range_options(df: pd.DataFrame) -> list:
    sub = (
        df.dropna(subset=["br_range_min", "br_range_label"])
        [["br_range_min", "br_range_label"]]
        .drop_duplicates()
        .sort_values("br_range_min")
    )
    return list(sub["br_range_label"])


br_range_options = _br_range_options(vehicle_30d_df)
country_options = sorted(vehicle_30d_df["country"].dropna().unique())
type_options = sorted(vehicle_30d_df["vehicle_type"].dropna().unique())

with st.container(border=True):
    st.markdown("**Filters** · empty pill groups mean *all*")

    deck_left, deck_right = st.columns([3, 2], gap="large")

    with deck_left:
        selected_ranges = st.pills(
            "BR range",
            options=br_range_options,
            selection_mode="multi",
            key="flt_br_ranges",
        )
        selected_countries = st.pills(
            "Nation",
            options=country_options,
            selection_mode="multi",
            key="flt_nations",
        )
        selected_types = st.pills(
            "Vehicle type",
            options=type_options,
            selection_mode="multi",
            key="flt_types",
        )

    with deck_right:
        premium_filter = st.segmented_control(
            "Premium",
            options=["All", "Non-premium", "Premium"],
            default="Non-premium",
            key="flt_premium",
        )

        tog_a, tog_b = st.columns(2)
        with tog_a:
            show_squadron = st.toggle("Squadron", value=True, key="flt_squadron")
            show_pack = st.toggle("Pack", value=True, key="flt_pack")
        with tog_b:
            show_marketplace = st.toggle("Marketplace", value=True, key="flt_marketplace")

# Normalize widget return values (empty multi -> all; None single -> All).
selected_ranges = selected_ranges or []
selected_countries = selected_countries or []
selected_types = selected_types or []
if premium_filter is None:
    premium_filter = "All"


def apply_vehicle_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if selected_ranges and "br_range_label" in out.columns:
        out = out[out["br_range_label"].isin(selected_ranges)]

    if selected_countries:
        out = out[out["country"].isin(selected_countries)]

    if selected_types:
        out = out[out["vehicle_type"].isin(selected_types)]

    if premium_filter == "Non-premium" and "is_premium" in out.columns:
        out = out[out["is_premium"] == False]

    if premium_filter == "Premium" and "is_premium" in out.columns:
        out = out[out["is_premium"] == True]

    if not show_squadron and "is_squadron" in out.columns:
        out = out[out["is_squadron"] == False]

    if not show_pack and "is_pack" in out.columns:
        out = out[out["is_pack"] == False]

    if not show_marketplace and "on_marketplace" in out.columns:
        out = out[out["on_marketplace"] == False]

    return out


filtered_vehicle_df = apply_vehicle_filters(vehicle_30d_df)
valid_vehicle_slugs = filtered_vehicle_df["vehicle_slug"].unique()
filtered_recent_df = recent_df[recent_df["vehicle_slug"].isin(valid_vehicle_slugs)].copy()
br_30d_df = features.build_br_aggregate(filtered_vehicle_df)


# ============================================================
# Summary metric cards
# ============================================================

n_vehicles = filtered_vehicle_df["vehicle_slug"].nunique()
total_battles = int(filtered_vehicle_df["battles"].fillna(0).sum())
n_nations = filtered_vehicle_df["country"].nunique()
avg_ce = filtered_vehicle_df["combat_effectiveness"].mean()
avg_ce_str = f"{avg_ce:.1f}" if pd.notna(avg_ce) else "N/A"

# Median vehicle-level win rate (win_rate is 0-100; not battle-weighted).
median_wr = filtered_vehicle_df["win_rate"].median(skipna=True)
median_wr_str = f"{median_wr:.1f}%" if pd.notna(median_wr) else "—"

if not recent_df.empty and recent_df["date"].notna().any():
    window_min = recent_df["date"].min()
    window_max = recent_df["date"].max()
    window_days = int((window_max - window_min).days) + 1
    window_value = f"{window_days}d"
    window_help = f"{window_min.date()} → {window_max.date()}"
else:
    window_value = "N/A"
    window_help = None

card_row1 = st.columns(3)
card_row1[0].metric("Vehicles", f"{n_vehicles:,}", border=True)
card_row1[1].metric("30-day sample battles", f"{total_battles:,}", border=True)
card_row1[2].metric("Nations", f"{n_nations}", border=True)

card_row2 = st.columns(3)
card_row2[0].metric("Avg CE Score", avg_ce_str, border=True)
card_row2[1].metric("Median WR", median_wr_str, border=True)
card_row2[2].metric("Rolling window", window_value, help=window_help, border=True)


# ============================================================
# Tabs
# ============================================================

tab_nation, tab_rankings, tab_clusters, tab_trends, tab_data = st.tabs(
    [
        "Nation Meta",
        "Vehicle Rankings",
        "Performance Clusters",
        "Trends",
        "Data Notes",
    ]
)


# ============================================================
# Nation Meta tab
# ============================================================

with tab_nation:
    if filtered_vehicle_df.empty:
        st.warning("No vehicles match the current filters.")
    else:
        # --- 1. Nation x BR Range heatmap ---
        st.subheader("Nation × BR Range — battle-weighted win rate")
        st.caption(
            "Cell value is the battle-weighted win rate over the rolling window. "
            "Color is centered on 50%."
        )

        heat = features.build_nation_br_heatmap(filtered_vehicle_df)

        if heat.empty:
            st.info("Not enough data for the heatmap under the current filters.")
        else:
            x_ranges = (
                heat.sort_values("br_range_min")["br_range_label"].drop_duplicates().tolist()
            )
            y_nations = sorted(heat["country"].unique())

            def _pivot(col):
                return (
                    heat.pivot(index="country", columns="br_range_label", values=col)
                    .reindex(index=y_nations, columns=x_ranges)
                )

            z = _pivot("win_rate_bw")
            customdata = np.dstack(
                [
                    _pivot("battles").values,
                    _pivot("vehicles").values,
                    _pivot("avg_kd").values,
                    _pivot("avg_fpb").values,
                    _pivot("avg_ce").values,
                ]
            )
            zv = z.values
            text = [
                [f"{v:.1f}" if np.isfinite(v) else "" for v in row]
                for row in zv
            ]

            heat_fig = go.Figure(
                go.Heatmap(
                    z=zv,
                    x=x_ranges,
                    y=y_nations,
                    customdata=customdata,
                    colorscale="RdYlGn",
                    zmid=50,
                    zmin=42,
                    zmax=58,
                    colorbar=dict(title="Win %"),
                    text=text,
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    xgap=2,
                    ygap=2,
                    hoverongaps=False,
                    hovertemplate=(
                        "<b>%{y} · %{x}</b><br>"
                        "Win rate: %{z:.1f}%<br>"
                        "Sample battles: %{customdata[0]:,.0f}<br>"
                        "Vehicles: %{customdata[1]:.0f}<br>"
                        "Avg K/D: %{customdata[2]:.2f}<br>"
                        "Avg frags/battle: %{customdata[3]:.2f}<br>"
                        "Avg CE Score: %{customdata[4]:.1f}"
                        "<extra></extra>"
                    ),
                )
            )
            heat_fig.update_layout(
                xaxis_title="BR range",
                yaxis_title="Nation",
                height=max(380, 42 * len(y_nations) + 140),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            heat_fig.update_xaxes(side="bottom")
            st.plotly_chart(heat_fig, width="stretch")

            st.caption(
                "Heatmap reflects the active filters. Blank cells mean either no "
                "vehicles passed the current filters, no win-rate signal was "
                "available, or some source rows return N/A for BR metadata and are "
                "omitted from BR-range cells."
            )

        st.divider()

        # --- 2. Nation daily trend ---
        st.subheader("Nation daily trend")

        trend_label_map = {
            "Win rate": "win_rate",
            "K/D": "ground_frags_per_death",
            "Frags per battle": "ground_frags_per_battle",
            "Sample battles": "battles",
        }

        tc1, tc2 = st.columns([3, 2], gap="large")
        with tc1:
            trend_choice = st.pills(
                "Metric",
                options=list(trend_label_map.keys()),
                default="Win rate",
                selection_mode="single",
                key="nation_trend_metric",
            )
        with tc2:
            trend_roll = st.toggle(
                "3-day rolling average", value=True, key="nation_trend_roll"
            )

        trend_choice = trend_choice or "Win rate"
        trend_metric = trend_label_map[trend_choice]

        nation_trend = features.build_nation_daily_trend(filtered_recent_df)

        if nation_trend.empty:
            st.info("No daily trend rows under the current filters.")
        else:
            plot_df = nation_trend.sort_values(["country", "date"]).copy()
            if trend_roll:
                plot_df[trend_metric] = (
                    plot_df.groupby("country")[trend_metric]
                    .transform(lambda s: s.rolling(3, min_periods=1).mean())
                )

            trend_fig = px.line(
                plot_df,
                x="date",
                y=trend_metric,
                color="country",
            )
            if trend_metric == "win_rate":
                trend_fig.add_hline(
                    y=50, line_dash="dot", opacity=0.5, annotation_text="50%"
                )
            trend_fig.update_layout(
                xaxis_title="Date",
                yaxis_title=trend_choice,
                height=460,
                margin=dict(l=10, r=10, t=20, b=10),
                legend_title_text="Nation",
            )
            st.plotly_chart(trend_fig, width="stretch")

        st.divider()

        # --- 3. Nation distribution ---
        st.subheader("Nation distribution")

        dist_label_map = {
            "CE Score": "combat_effectiveness",
            "K/D": "ground_frags_per_death",
        }
        dist_choice = st.pills(
            "Distribution metric",
            options=list(dist_label_map.keys()),
            default="CE Score",
            selection_mode="single",
            key="nation_dist_metric",
        )
        dist_choice = dist_choice or "CE Score"
        dist_metric = dist_label_map[dist_choice]

        dist_df = filtered_vehicle_df.dropna(subset=[dist_metric, "country"]).copy()

        if dist_df.empty:
            st.info("No vehicles with this metric under the current filters.")
        else:
            order = (
                dist_df.groupby("country")[dist_metric]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            dist_fig = px.box(
                dist_df,
                x="country",
                y=dist_metric,
                color="country",
                points="outliers",
                category_orders={"country": order},
                hover_data=[
                    c
                    for c in [
                        "vehicle_name",
                        "realistic_br",
                        "battles",
                        "combat_effectiveness",
                    ]
                    if c in dist_df.columns
                ],
            )
            dist_fig.update_layout(
                xaxis_title="Nation",
                yaxis_title=dist_choice,
                height=480,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(dist_fig, width="stretch")

        st.divider()

        # --- 4. Nation strength (dumbbell + summary table) ---
        st.subheader("Nation strength")

        nation_summary = features.build_nation_summary(filtered_vehicle_df)
        if nation_summary.empty:
            st.info("No nation summary available under the current filters.")
        else:
            # Dumbbell: Median CE -> Avg CE per nation, sorted by Avg CE desc.
            dumbbell = nation_summary.dropna(subset=["avg_ce"]).copy()
            if dumbbell.empty:
                st.info("No scored nations to plot under the current filters.")
            else:
                # Ascending order so the highest Avg CE sits at the top of the chart.
                dumbbell = dumbbell.sort_values("avg_ce", ascending=True)
                nations = dumbbell["country"].tolist()
                customdata = np.column_stack(
                    [
                        dumbbell["median_ce"].to_numpy(),
                        dumbbell["avg_ce"].to_numpy(),
                        dumbbell["top_br_range"].fillna("—").astype(str).to_numpy(),
                        dumbbell["vehicles"].to_numpy(),
                        dumbbell["battles"].to_numpy(),
                    ]
                )
                hover = (
                    "<b>%{y}</b><br>"
                    "Avg CE: %{customdata[1]:.1f}<br>"
                    "Median CE: %{customdata[0]:.1f}<br>"
                    "Top BR range: %{customdata[2]}<br>"
                    "Vehicles: %{customdata[3]:.0f}<br>"
                    "Sample battles: %{customdata[4]:,.0f}"
                    "<extra></extra>"
                )

                db_fig = go.Figure()
                # Thin connecting segments (one per nation).
                for _, r in dumbbell.iterrows():
                    db_fig.add_trace(
                        go.Scatter(
                            x=[r["median_ce"], r["avg_ce"]],
                            y=[r["country"], r["country"]],
                            mode="lines",
                            line=dict(color="#5A6472", width=3),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                db_fig.add_trace(
                    go.Scatter(
                        x=dumbbell["median_ce"], y=nations, mode="markers",
                        name="Median CE",
                        marker=dict(color="#6EA8FF", size=11, symbol="circle"),
                        customdata=customdata, hovertemplate=hover,
                    )
                )
                db_fig.add_trace(
                    go.Scatter(
                        x=dumbbell["avg_ce"], y=nations, mode="markers",
                        name="Avg CE",
                        marker=dict(color="#E8743B", size=13, symbol="diamond"),
                        customdata=customdata, hovertemplate=hover,
                    )
                )
                db_fig.update_layout(
                    xaxis_title="Combat Effectiveness Score",
                    yaxis_title=None,
                    height=max(320, 34 * len(nations) + 120),
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(db_fig, width="stretch")

            st.markdown("**Nation strength summary**")
            st.dataframe(
                nation_summary,
                width="stretch",
                hide_index=True,
                column_config={
                    "country": "Nation",
                    "vehicles": st.column_config.NumberColumn("Vehicles", format="%d"),
                    "battles": st.column_config.NumberColumn("Sample battles", format="%d"),
                    "avg_ce": st.column_config.NumberColumn("Avg CE", format="%.1f"),
                    "median_ce": st.column_config.NumberColumn("Median CE", format="%.1f"),
                    "top_br_range": "Top BR range",
                    "best_vehicle": "Best vehicle",
                    "best_vehicle_ce": st.column_config.NumberColumn("Best CE", format="%.1f"),
                },
            )

        with st.expander("BR meta curve (avg CE by BR)"):
            if br_30d_df.empty:
                st.info("No BR data under the current filters.")
            else:
                br_fig = px.line(
                    br_30d_df,
                    x="realistic_br",
                    y="avg_combat_effectiveness",
                    markers=True,
                    hover_data=["vehicles", "battles", "avg_win_rate"],
                )
                br_fig.update_layout(
                    xaxis_title="Realistic BR",
                    yaxis_title="Average Combat Effectiveness Score",
                    height=400,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(br_fig, width="stretch")


# ============================================================
# Vehicle Rankings tab  (rankings + folded-in vehicle detail)
# ============================================================

with tab_rankings:
    st.subheader("Top performers by Combat Effectiveness Score")
    st.caption(
        "The Combat Effectiveness Score measures how much a vehicle overperforms "
        "the average for its exact BR (K/D, frags per battle, win rate, and a "
        "small data-confidence term). 50 is roughly average for the BR."
    )

    if filtered_vehicle_df.empty:
        st.warning("No vehicles match the current filters.")
    else:
        # Unique display label per vehicle (slug stays the internal key) so
        # duplicate names from different nations do not collide in charts/selectbox.
        vdf = filtered_vehicle_df.copy()

        def _fmt_br(br):
            return f"BR {br:.1f}" if pd.notna(br) else "BR N/A"

        vdf["display_label"] = [
            f"{n} — {c if pd.notna(c) else 'Unknown'}, {_fmt_br(b)}"
            for n, c, b in zip(vdf["vehicle_name"], vdf["country"], vdf["realistic_br"])
        ]

        # Percentile features among currently filtered vehicles (for the radar),
        # plus log1p battles used by the radar and by Similar vehicles.
        vdf["_wr_pct"] = vdf["win_rate"].rank(pct=True) * 100
        vdf["_kd_pct"] = vdf["ground_frags_per_death"].rank(pct=True) * 100
        vdf["_fpb_pct"] = vdf["ground_frags_per_battle"].rank(pct=True) * 100
        vdf["_logbattles"] = np.log1p(vdf["total_battles_30d"].fillna(0))
        vdf["_ss_pct"] = vdf["_logbattles"].rank(pct=True) * 100

        ranking_cols = [
            "vehicle_name",
            "country",
            "vehicle_type",
            "rank",
            "realistic_br",
            "battles",
            "days_observed",
            "win_rate",
            "ground_frags_per_battle",
            "ground_frags_per_death",
            "efficiency",
            "combat_effectiveness",
        ]
        ranking_cols = [c for c in ranking_cols if c in vdf.columns]

        rankings = (
            vdf.sort_values(["combat_effectiveness", "battles"], ascending=[False, False])
            .head(25)
            .copy()
        )

        # --- CE bar chart (full-width; unique label per bar; x fixed 0-100) ---
        st.markdown("**Top vehicles by CE Score**")
        bar_df = rankings.sort_values("combat_effectiveness", ascending=True)
        bar_fig = px.bar(
            bar_df,
            x="combat_effectiveness",
            y="display_label",
            orientation="h",
            color="country",
            hover_data=[
                c
                for c in [
                    "realistic_br",
                    "battles",
                    "days_observed",
                    "win_rate",
                    "ground_frags_per_battle",
                    "ground_frags_per_death",
                ]
                if c in bar_df.columns
            ],
        )
        bar_fig.update_layout(
            xaxis_title="Combat Effectiveness Score",
            yaxis_title=None,
            height=max(360, 26 * len(bar_df) + 120),
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="Nation",
        )
        bar_fig.update_xaxes(range=[0, 100])
        st.plotly_chart(bar_fig, width="stretch")

        # --- Top performer table (collapsed, below the bar) ---
        with st.expander("Show top performer table"):
            st.dataframe(
                rankings[ranking_cols],
                width="stretch",
                hide_index=True,
                height=520,
                column_config={
                    "vehicle_name": "Vehicle",
                    "country": "Nation",
                    "vehicle_type": "Type",
                    "realistic_br": st.column_config.NumberColumn("BR", format="%.1f"),
                    "battles": st.column_config.NumberColumn("Sample battles", format="%d"),
                    "days_observed": st.column_config.NumberColumn("Days", format="%d"),
                    "win_rate": st.column_config.NumberColumn("Win rate", format="%.2f%%"),
                    "ground_frags_per_battle": st.column_config.NumberColumn("Frags / battle", format="%.2f"),
                    "ground_frags_per_death": st.column_config.NumberColumn("Frags / death", format="%.2f"),
                    "efficiency": st.column_config.NumberColumn("Efficiency", format="%.1f"),
                    "combat_effectiveness": st.column_config.NumberColumn("CE Score", format="%.1f"),
                },
            )

        # --- Daily K/D Stability Plot (top 12 by 30-day aggregate K/D) ---
        st.markdown("**Daily K/D stability — top 12 by 30-day K/D**")
        st.caption(
            "Thick bar = middle 50% of daily K/D (25th–75th pct); thin line = "
            "10th–90th pct; dot = median daily K/D; diamond = 30-day aggregate "
            "K/D. Tight bars mean consistent performance; a diamond far from the "
            "median means spike-driven K/D."
        )
        topkd = (
            vdf.dropna(subset=["ground_frags_per_death"])
            .sort_values("ground_frags_per_death", ascending=False)
            .head(12)
        )
        if topkd.empty:
            st.info("No K/D data under the current filters.")
        else:
            daily_kd = filtered_recent_df[
                filtered_recent_df["vehicle_slug"].isin(topkd["vehicle_slug"])
            ].dropna(subset=["ground_frags_per_death"])
            grp = daily_kd.groupby("vehicle_slug")["ground_frags_per_death"]
            stab = pd.DataFrame(
                {
                    "p10": grp.quantile(0.10),
                    "p25": grp.quantile(0.25),
                    "median": grp.quantile(0.50),
                    "p75": grp.quantile(0.75),
                    "p90": grp.quantile(0.90),
                    "obs_days": grp.count(),
                }
            ).reset_index()
            stab = stab.merge(
                topkd[
                    [
                        "vehicle_slug",
                        "display_label",
                        "country",
                        "realistic_br",
                        "ground_frags_per_death",
                        "total_battles_30d",
                    ]
                ],
                on="vehicle_slug",
                how="left",
            ).rename(columns={"ground_frags_per_death": "agg_kd"})
            stab = stab.sort_values("agg_kd", ascending=True)  # highest at top

            palette = px.colors.qualitative.Set2
            nats = sorted(stab["country"].fillna("Unknown").unique())
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(nats)}

            stab_fig = go.Figure()
            for _, r in stab.iterrows():
                col = color_map.get(r["country"] if pd.notna(r["country"]) else "Unknown", "#888888")
                y = r["display_label"]
                br_txt = f"{r['realistic_br']:.1f}" if pd.notna(r["realistic_br"]) else "N/A"
                sb_txt = f"{int(r['total_battles_30d']):,}" if pd.notna(r["total_battles_30d"]) else "0"
                hover = (
                    f"<b>{r['display_label']}</b><br>"
                    f"Nation: {r['country']}<br>"
                    f"BR: {br_txt}<br>"
                    f"Aggregate K/D: {r['agg_kd']:.2f}<br>"
                    f"Median daily K/D: {r['median']:.2f}<br>"
                    f"25th–75th: {r['p25']:.2f}–{r['p75']:.2f}<br>"
                    f"10th–90th: {r['p10']:.2f}–{r['p90']:.2f}<br>"
                    f"Observed days: {int(r['obs_days'])}<br>"
                    f"Sample battles: {sb_txt}"
                    "<extra></extra>"
                )
                stab_fig.add_trace(go.Scatter(
                    x=[r["p10"], r["p90"]], y=[y, y], mode="lines",
                    line=dict(color=col, width=2), opacity=0.4,
                    hoverinfo="skip", showlegend=False,
                ))
                stab_fig.add_trace(go.Scatter(
                    x=[r["p25"], r["p75"]], y=[y, y], mode="lines",
                    line=dict(color=col, width=9),
                    hovertemplate=hover, showlegend=False,
                ))
                stab_fig.add_trace(go.Scatter(
                    x=[r["median"]], y=[y], mode="markers",
                    marker=dict(color=col, size=10, symbol="circle",
                                line=dict(color="#0F1216", width=1)),
                    hovertemplate=hover, showlegend=False,
                ))
                stab_fig.add_trace(go.Scatter(
                    x=[r["agg_kd"]], y=[y], mode="markers",
                    marker=dict(color="#E8743B", size=12, symbol="diamond",
                                line=dict(color="#0F1216", width=1)),
                    hovertemplate=hover, showlegend=False,
                ))
            stab_fig.update_layout(
                xaxis_title="K/D (ground frags per death)",
                yaxis_title=None,
                height=max(360, 40 * len(stab) + 120),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(stab_fig, width="stretch")

        st.divider()

        # --- folded-in vehicle detail (was the Vehicle Explorer tab) ---
        st.subheader("Vehicle detail")

        detail_opts = vdf.sort_values("display_label")["vehicle_slug"].tolist()
        slug_to_label = dict(zip(vdf["vehicle_slug"], vdf["display_label"]))

        selected_slug = st.selectbox(
            "Select a vehicle",
            options=detail_opts,
            format_func=lambda s: slug_to_label.get(s, s),
            key="rankings_vehicle_select",
        )

        vehicle_row = vdf[vdf["vehicle_slug"] == selected_slug].head(1)

        if not vehicle_row.empty:
            row = vehicle_row.iloc[0]

            st.markdown(f"### {row.get('vehicle_name', 'Unknown vehicle')}")

            if not bool(row.get("has_realistic_br", True)):
                st.warning(
                    "No Realistic BR is available from ThunderSkill for this "
                    "vehicle. It is kept in the dataset because sample performance "
                    "data exists, but it is omitted from BR-normalized views such "
                    "as CE Score, BR heatmaps, and BR grouping."
                )

            # Link buttons side by side, kept compact (stack on mobile).
            pic_url = row.get("pic")
            ts_url = row.get("vehicle_url")
            btn_a, btn_b, _btn_spacer = st.columns([2, 2, 3])
            with btn_a:
                if pd.notna(pic_url) and str(pic_url).startswith("http"):
                    st.link_button("Open vehicle image", str(pic_url), width="stretch")
            with btn_b:
                if pd.notna(ts_url):
                    st.link_button("Open ThunderSkill page", str(ts_url), width="stretch")

            # Stat cards.
            d1, d2 = st.columns(2)
            d1.metric("Nation", row.get("country", "N/A"))
            d2.metric("Type", row.get("vehicle_type", "N/A"))

            d3, d4 = st.columns(2)
            d3.metric("BR", f"{row.get('realistic_br', np.nan):.1f}")
            d4.metric("Rank", f"{row.get('rank', 'N/A')}")

            d5, d6 = st.columns(2)
            d5.metric("30-day sample battles", f"{int(row.get('battles', 0)):,}")
            d6.metric("Days observed", f"{int(row.get('days_observed', 0))}")

            d7, d8 = st.columns(2)
            d7.metric("Win rate", f"{row.get('win_rate', np.nan):.2f}%")
            d8.metric("CE Score", f"{row.get('combat_effectiveness', np.nan):.1f}")

            d9, d10 = st.columns(2)
            d9.metric("Frags / battle", f"{row.get('ground_frags_per_battle', np.nan):.2f}")
            d10.metric("Frags / death", f"{row.get('ground_frags_per_death', np.nan):.2f}")

            # --- Performance radar (0-100; percentiles among filtered peers) ---
            st.markdown("**Performance radar**")
            radar_axes = {
                "CE Score": row.get("combat_effectiveness"),
                "Win Rate Score": row.get("_wr_pct"),
                "K/D Score": row.get("_kd_pct"),
                "Frags/Battle Score": row.get("_fpb_pct"),
                "Sample Size": row.get("_ss_pct"),
            }
            missing_axes = [k for k, v in radar_axes.items() if pd.isna(v)]
            theta = list(radar_axes.keys())
            r_vals = [0.0 if pd.isna(v) else float(v) for v in radar_axes.values()]
            radar_fig = go.Figure()
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=r_vals + [r_vals[0]],
                    theta=theta + [theta[0]],
                    fill="toself",
                    line=dict(color="#E8743B"),
                    name=str(row.get("vehicle_name", "")),
                )
            )
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
                showlegend=False,
                height=380,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(radar_fig, width="stretch")
            radar_cap = (
                "Win Rate / K/D / Frags-per-battle / Sample Size are percentiles "
                "among the currently filtered vehicles; CE Score is the 0-100 "
                "Combat Effectiveness Score."
            )
            if missing_axes:
                radar_cap += " Unavailable metrics shown as 0: " + ", ".join(missing_axes) + "."
            if len(vdf) < 3:
                radar_cap += " Percentiles are unreliable with so few vehicles in view."
            st.caption(radar_cap)

            # --- Similar vehicles (simple standardized nearest-neighbor) ---
            st.markdown("**Similar vehicles by K-nearest neighbors**")
            st.caption(
                "Compares the selected vehicle with currently filtered vehicles "
                "using standardized BR, CE Score, win rate, K/D, frags per battle, "
                "and log sample battles. The search starts with same vehicle type "
                "and nearby BR, then broadens if too few peers are available."
            )
            sim_feats = [
                "realistic_br",
                "combat_effectiveness",
                "win_rate",
                "ground_frags_per_death",
                "ground_frags_per_battle",
                "_logbattles",
            ]
            if pd.isna(row.get("realistic_br")) or pd.isna(row.get("combat_effectiveness")):
                st.info(
                    "Similar vehicles need a Realistic BR and CE Score, which are "
                    "unavailable for this vehicle."
                )
            else:
                base = vdf.dropna(
                    subset=[
                        "realistic_br",
                        "combat_effectiveness",
                        "win_rate",
                        "ground_frags_per_death",
                        "ground_frags_per_battle",
                    ]
                ).copy()
                base = base[base["vehicle_slug"] != selected_slug]

                # Candidate pool: same type & BR within +-1.0, broadening if sparse.
                cand = base[
                    (base["vehicle_type"] == row.get("vehicle_type"))
                    & ((base["realistic_br"] - row["realistic_br"]).abs() <= 1.0)
                ]
                if len(cand) < 3:
                    cand = base[base["vehicle_type"] == row.get("vehicle_type")]
                if len(cand) < 3:
                    cand = base

                if cand.empty:
                    st.info("No similar vehicles under the current filters.")
                else:
                    sel_feats = pd.Series({f: row[f] for f in sim_feats}, dtype="float64")
                    scale_src = pd.concat(
                        [cand[sim_feats], sel_feats.to_frame().T], ignore_index=True
                    )
                    mu = scale_src.mean()
                    sd = scale_src.std(ddof=0).replace(0, 1.0)
                    cz = (cand[sim_feats] - mu) / sd
                    sz = (sel_feats - mu) / sd
                    dist = np.sqrt(((cz - sz) ** 2).sum(axis=1))
                    sim = cand.assign(_dist=dist).sort_values("_dist").head(3)
                    sim["similarity"] = 1.0 / (1.0 + sim["_dist"])

                    sim_cols = [
                        "vehicle_name",
                        "country",
                        "realistic_br",
                        "vehicle_type",
                        "combat_effectiveness",
                        "ground_frags_per_death",
                        "ground_frags_per_battle",
                        "total_battles_30d",
                        "similarity",
                    ]
                    st.dataframe(
                        sim[sim_cols],
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "vehicle_name": "Vehicle",
                            "country": "Nation",
                            "realistic_br": st.column_config.NumberColumn("BR", format="%.1f"),
                            "vehicle_type": "Type",
                            "combat_effectiveness": st.column_config.NumberColumn("CE Score", format="%.1f"),
                            "ground_frags_per_death": st.column_config.NumberColumn("K/D", format="%.2f"),
                            "ground_frags_per_battle": st.column_config.NumberColumn("Frags / battle", format="%.2f"),
                            "total_battles_30d": st.column_config.NumberColumn("Sample battles", format="%d"),
                            "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                        },
                    )

                    # Similarity ladder (compact lollipop of the top-3 matches).
                    ladder = sim.sort_values("similarity", ascending=True)
                    ladder_fig = px.bar(
                        ladder,
                        x="similarity",
                        y="display_label",
                        orientation="h",
                        color="country",
                        hover_data={
                            "display_label": False,
                            "realistic_br": ":.1f",
                            "vehicle_type": True,
                            "combat_effectiveness": ":.1f",
                            "ground_frags_per_death": ":.2f",
                            "ground_frags_per_battle": ":.2f",
                            "total_battles_30d": ":,.0f",
                            "similarity": ":.3f",
                        },
                    )
                    ladder_fig.update_layout(
                        xaxis_title="Similarity",
                        yaxis_title=None,
                        xaxis=dict(range=[0, 1]),
                        height=max(220, 60 * len(ladder) + 80),
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend_title_text="Nation",
                    )
                    st.plotly_chart(ladder_fig, width="stretch")

            # --- Trend chart + raw observations (last) ---
            vehicle_trend_df = filtered_recent_df[
                filtered_recent_df["vehicle_slug"] == selected_slug
            ].sort_values("date")

            if vehicle_trend_df.empty:
                st.info("No trend data available for this vehicle.")
            else:
                trend_options = [
                    c
                    for c in [
                        "win_rate",
                        "ground_frags_per_battle",
                        "ground_frags_per_death",
                        "battles",
                    ]
                    if c in vehicle_trend_df.columns
                ]
                _default_trend = "ground_frags_per_battle"
                trend_index = (
                    trend_options.index(_default_trend)
                    if _default_trend in trend_options
                    else 0
                )
                detail_metric = st.selectbox(
                    "Trend metric",
                    options=trend_options,
                    index=trend_index,
                    key="rankings_detail_metric",
                )

                detail_fig = px.line(
                    vehicle_trend_df,
                    x="date",
                    y=detail_metric,
                    markers=True,
                    title=f"{row.get('vehicle_name', '')}: {detail_metric.replace('_', ' ').title()}",
                )
                detail_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=detail_metric.replace("_", " ").title(),
                    height=420,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(detail_fig, width="stretch")

                with st.expander("Show raw 30-day vehicle observations"):
                    st.dataframe(
                        vehicle_trend_df[
                            [
                                c
                                for c in [
                                    "date",
                                    "battles",
                                    "win_rate",
                                    "ground_frags_per_battle",
                                    "ground_frags_per_death",
                                    "efficiency",
                                ]
                                if c in vehicle_trend_df.columns
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                    )


# ============================================================
# Performance Clusters tab
# ============================================================

with tab_clusters:
    st.subheader("Performance Clusters")
    st.caption(
        "Find vehicle archetypes by BR-relative strength, combat efficiency, and "
        "sample confidence."
    )

    with st.expander("How these clusters work", expanded=False):
        st.markdown(
            "HDBSCAN is a density-based clustering method. Instead of forcing every "
            "vehicle into a fixed number of groups, it looks for natural dense "
            "pockets in the data and can mark uncertain vehicles as noise/outliers. "
            "That makes it useful for messy game-performance data where some "
            "vehicles form clear groups and others are weird one-offs.\n\n"
            "Clusters here use **only three signals**: **CE Score** (BR-relative "
            "strength), **K/D** (combat efficiency), and **log1p(sample battles)** "
            "(evidence strength). Win rate and frags per battle are shown as "
            "**context only** — they do not form the clusters. Clustering runs on the "
            "currently filtered slice, so archetypes change as you adjust the top "
            "filters."
        )

    min_sb = st.slider(
        "Minimum sample battles",
        min_value=0,
        max_value=500,
        value=int(features.CLUSTER_MIN_SAMPLE_BATTLES),
        step=25,
        key="clusters_min_sample_battles",
        help="Vehicles below this 30-day sample-battle count are excluded from clustering.",
    )

    # Missing-BR vehicles are excluded inside build_vehicle_clusters.
    clustered_df, cluster_meta = get_clusters(filtered_vehicle_df, min_sb)

    if not cluster_meta["available"]:
        st.error(
            "Clustering needs scikit-learn, which is not installed in this "
            "environment. Add `scikit-learn>=1.5` to requirements.txt."
        )
    elif cluster_meta["reason"] in ("empty", "too_few") or clustered_df.empty:
        st.warning(
            f"Not enough vehicles to cluster under the current filters "
            f"({cluster_meta['n_vehicles']} with a Realistic BR and ≥ {min_sb} "
            f"sample battles; need at least {features.CLUSTER_MIN_VEHICLES}). "
            "Widen the filters or lower the minimum sample battles."
        )
    else:
        cdf = clustered_df.copy()

        def _cluster_fmt_br(br):
            return f"BR {br:.1f}" if pd.notna(br) else "BR N/A"

        cdf["display_label"] = [
            f"{n} — {c if pd.notna(c) else 'Unknown'}, {_cluster_fmt_br(b)}"
            for n, c, b in zip(cdf["vehicle_name"], cdf["country"], cdf["realistic_br"])
        ]

        # --- 2. Quality cards ---
        sil = cluster_meta["silhouette"]
        qcard = st.columns(4)
        qcard[0].metric("Vehicles clustered", f"{cluster_meta['n_vehicles']:,}", border=True)
        qcard[1].metric("Clusters found", f"{cluster_meta['n_clusters']}", border=True)
        qcard[2].metric("Outliers", f"{cluster_meta['noise_pct']:.0f}%", border=True)
        qcard[3].metric(
            "Cluster quality",
            cluster_meta["quality_label"],
            help=(f"Silhouette {sil:.2f} (rough guide only)" if sil is not None else "Silhouette n/a"),
            border=True,
        )

        if cluster_meta["n_clusters"] == 0:
            st.info(
                "HDBSCAN did not find dense archetypes in this slice — every vehicle "
                "is an outlier. This is common for small or very uniform slices. Try "
                "a wider BR range or a lower minimum sample battles."
            )
        elif cluster_meta["n_clusters"] > 20 or (
            cluster_meta["n_vehicles"] > 250 and cluster_meta["n_clusters"] > 25
        ):
            st.info(
                "This slice produced many small archetypes. For a cleaner cluster "
                "map, narrow the BR range or select a vehicle type."
            )

        # Canonical label order (+ any disambiguated suffixes appended).
        label_order = [
            "Core Meta",
            "Underplayed Meta",
            "Solid Picks",
            "Popular Strugglers",
            "Niche Signals",
            "Off-Meta",
            "Outliers",
        ]
        labels_present = set(cdf["friendly_label"].dropna())
        present = [l for l in label_order if l in labels_present]
        present += sorted(labels_present - set(present))

        # --- 3. 3D cluster space ---
        st.markdown("**Cluster feature space (3D)** — the exact features the model uses")
        fig3d = px.scatter_3d(
            cdf,
            x="combat_effectiveness",
            y="ground_frags_per_death",
            z="log1p_sample_battles",
            color="friendly_label",
            category_orders={"friendly_label": present},
            hover_name="display_label",
            hover_data={
                "display_label": False,
                "country": True,
                "realistic_br": ":.1f",
                "vehicle_type": True,
                "combat_effectiveness": ":.1f",
                "ground_frags_per_death": ":.2f",
                "total_battles_30d": ":,.0f",
                "win_rate": ":.1f",
                "ground_frags_per_battle": ":.2f",
            },
        )
        fig3d.update_traces(marker=dict(size=4))
        fig3d.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            legend_title_text="Archetype",
            scene=dict(
                xaxis_title="CE Score",
                yaxis_title="K/D",
                zaxis_title="log1p(sample battles)",
            ),
        )
        st.plotly_chart(fig3d, width="stretch")

        # --- 4. 2D CE vs K/D nation map (most readable) ---
        st.markdown("**Archetype map — CE Score vs K/D** (point size = sample battles)")
        fig2d = px.scatter(
            cdf,
            x="combat_effectiveness",
            y="ground_frags_per_death",
            size="total_battles_30d",
            size_max=28,
            color="friendly_label",
            category_orders={"friendly_label": present},
            opacity=0.8,
            hover_name="display_label",
            hover_data={
                "display_label": False,
                "friendly_label": True,
                "country": True,
                "realistic_br": ":.1f",
                "vehicle_type": True,
                "combat_effectiveness": ":.1f",
                "ground_frags_per_death": ":.2f",
                "total_battles_30d": ":,.0f",
                "win_rate": ":.1f",
                "ground_frags_per_battle": ":.2f",
            },
        )
        fig2d.update_layout(
            height=560,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="CE Score",
            yaxis_title="K/D (ground frags per death)",
            legend_title_text="Archetype",
        )
        st.plotly_chart(fig2d, width="stretch")

        # --- 5. Cluster profile summary ---
        st.markdown("**Cluster profiles** (win rate & frags/battle are context only)")
        prof = (
            cdf.groupby("friendly_label")
            .agg(
                vehicles=("vehicle_slug", "nunique"),
                avg_ce=("combat_effectiveness", "mean"),
                avg_kd=("ground_frags_per_death", "mean"),
                median_sample_battles=("total_battles_30d", "median"),
                avg_win_rate=("win_rate", "mean"),
                avg_frags_per_battle=("ground_frags_per_battle", "mean"),
            )
            .reset_index()
        )
        top_by_ce = (
            cdf.sort_values("combat_effectiveness", ascending=False)
            .groupby("friendly_label")["vehicle_name"].first()
            .rename("top_vehicle")
        )
        prof = prof.merge(top_by_ce, on="friendly_label", how="left")
        prof["_ord"] = prof["friendly_label"].map({l: i for i, l in enumerate(present)}).fillna(999)
        prof = prof.sort_values("_ord").drop(columns="_ord")
        st.dataframe(
            prof,
            width="stretch",
            hide_index=True,
            column_config={
                "friendly_label": "Archetype",
                "vehicles": st.column_config.NumberColumn("Vehicles", format="%d"),
                "avg_ce": st.column_config.NumberColumn("Avg CE", format="%.1f"),
                "avg_kd": st.column_config.NumberColumn("Avg K/D", format="%.2f"),
                "median_sample_battles": st.column_config.NumberColumn("Median sample battles", format="%d"),
                "avg_win_rate": st.column_config.NumberColumn("Avg win rate (ctx)", format="%.1f%%"),
                "avg_frags_per_battle": st.column_config.NumberColumn("Avg frags/battle (ctx)", format="%.2f"),
                "top_vehicle": "Top vehicle by CE",
            },
        )

        # --- 6. Cluster drilldown ---
        st.markdown("**Explore an archetype**")
        drill = st.selectbox("Archetype", options=present, key="clusters_drill")
        sub = cdf[cdf["friendly_label"] == drill]
        if sub.empty:
            st.info("No vehicles in this archetype.")
        else:
            st.caption(
                f"**{drill}** — {sub['vehicle_slug'].nunique()} vehicles · "
                f"avg CE {sub['combat_effectiveness'].mean():.1f} · "
                f"avg K/D {sub['ground_frags_per_death'].mean():.2f} · "
                f"median {int(sub['total_battles_30d'].median()):,} sample battles."
            )
            drill_a, drill_b = st.columns([3, 2], gap="large")
            with drill_a:
                st.markdown("Top vehicles by CE Score")
                st.dataframe(
                    sub.sort_values("combat_effectiveness", ascending=False)[
                        [
                            "display_label",
                            "combat_effectiveness",
                            "ground_frags_per_death",
                            "total_battles_30d",
                        ]
                    ].head(10),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "display_label": "Vehicle",
                        "combat_effectiveness": st.column_config.NumberColumn("CE Score", format="%.1f"),
                        "ground_frags_per_death": st.column_config.NumberColumn("K/D", format="%.2f"),
                        "total_battles_30d": st.column_config.NumberColumn("Sample battles", format="%d"),
                    },
                )
            with drill_b:
                st.markdown("Nation composition")
                comp = (
                    sub["country"].value_counts().rename_axis("Nation").reset_index(name="Vehicles")
                )
                st.dataframe(comp, width="stretch", hide_index=True)

        # --- 7. Full clustered table (collapsed) ---
        with st.expander("Full clustered vehicle table"):
            full_cols = [
                "display_label",
                "vehicle_name",
                "friendly_label",
                "cluster_id",
                "country",
                "realistic_br",
                "vehicle_type",
                "combat_effectiveness",
                "ground_frags_per_death",
                "total_battles_30d",
                "win_rate",
                "ground_frags_per_battle",
            ]
            full_cols = [c for c in full_cols if c in cdf.columns]
            st.dataframe(
                cdf[full_cols].sort_values(
                    ["friendly_label", "combat_effectiveness"], ascending=[True, False]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "display_label": "Vehicle (label)",
                    "vehicle_name": "Name",
                    "friendly_label": "Archetype",
                    "cluster_id": st.column_config.NumberColumn("Cluster ID", format="%d"),
                    "country": "Nation",
                    "realistic_br": st.column_config.NumberColumn("BR", format="%.1f"),
                    "vehicle_type": "Type",
                    "combat_effectiveness": st.column_config.NumberColumn("CE Score", format="%.1f"),
                    "ground_frags_per_death": st.column_config.NumberColumn("K/D", format="%.2f"),
                    "total_battles_30d": st.column_config.NumberColumn("Sample battles", format="%d"),
                    "win_rate": st.column_config.NumberColumn("Win rate (ctx)", format="%.1f%%"),
                    "ground_frags_per_battle": st.column_config.NumberColumn("Frags/battle (ctx)", format="%.2f"),
                },
            )


# ============================================================
# Trends tab
# ============================================================

with tab_trends:
    st.subheader("Daily trends")
    st.caption(
        "Daily observations after the current filters. Metrics are simple "
        "daily means across the selected grouping."
    )

    if filtered_recent_df.empty:
        st.warning("No trend rows match the current filters.")
    else:
        trend_label_map = {
            "Win rate": "win_rate",
            "K/D": "ground_frags_per_death",
            "Frags per battle": "ground_frags_per_battle",
            "Sample battles": "battles",
        }

        tcol1, tcol2 = st.columns([3, 2], gap="large")
        with tcol1:
            global_trend_choice = st.pills(
                "Trend metric",
                options=list(trend_label_map.keys()),
                default="Win rate",
                selection_mode="single",
                key="global_trend_metric",
            )
        with tcol2:
            trend_group = st.segmented_control(
                "Group by",
                options=["Nation", "BR"],
                default="Nation",
                key="global_trend_group",
            )

        global_trend_choice = global_trend_choice or "Win rate"
        trend_group = trend_group or "Nation"
        global_metric = trend_label_map[global_trend_choice]

        group_col = "country" if trend_group == "Nation" else "realistic_br"

        grouped = (
            filtered_recent_df
            .groupby(["date", group_col], as_index=False)
            .agg(value=(global_metric, "mean"))
            .sort_values("date")
        )

        global_fig = px.line(
            grouped,
            x="date",
            y="value",
            color=group_col,
            markers=True,
        )
        global_fig.update_layout(
            xaxis_title="Date",
            yaxis_title=global_trend_choice,
            height=520,
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text=trend_group,
        )
        st.plotly_chart(global_fig, width="stretch")


# ============================================================
# Data Notes tab
# ============================================================

with tab_data:
    st.subheader("Dataset notes")

    st.write(
        "The app loads the latest automated ThunderSkill CSV from GitHub. "
        "The scraper collects up to 30 chart observations per Realistic Ground vehicle."
    )

    st.subheader("About battle counts")
    st.info(
        "Battle counts here come from ThunderSkill's **tracked-user sample** — "
        "players who have linked their account — not global War Thunder battle "
        "totals. Absolute numbers are therefore small, and low counts for rare, "
        "event, or minor-nation vehicles are expected. They remain useful as a "
        "**sample-size / confidence weight**: the Combat Effectiveness Score "
        "uses them to pull low-sample vehicles toward their BR average. "
        "(`index_battles` is essentially the same window sample, pre-summed.)"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{len(raw_df):,}")
    c2.metric("Raw vehicles", f"{raw_df['vehicle_slug'].nunique():,}")
    c3.metric("Recent rows", f"{len(recent_df):,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("30-day vehicle rows", f"{len(vehicle_30d_df):,}")
    c5.metric("Filtered vehicles", f"{len(filtered_vehicle_df):,}")
    c6.metric("Filtered sample battles", f"{int(filtered_vehicle_df['battles'].fillna(0).sum()):,}")

    if "date" in raw_df.columns:
        st.write("Raw date range:", raw_df["date"].min(), "to", raw_df["date"].max())
        st.write("App recent date range:", recent_df["date"].min(), "to", recent_df["date"].max())

    st.subheader("Combat Effectiveness Score")

    st.write(
        "The Combat Effectiveness Score measures how much a vehicle overperforms "
        "the average for its **exact BR**. Each metric is smoothed toward the BR "
        "average using an empirical-Bayes weight "
        "(reliability = battles / (battles + 100)), so low-battle vehicles are "
        "pulled toward average and do not dominate. Smoothed metrics are then "
        "compared to the BR's distribution with a robust (median / MAD) z-score "
        "and combined:"
    )

    st.code(
        """
z_total =
  0.40 × z(K/D, ground frags per death)
+ 0.40 × z(ground frags per battle)
+ 0.15 × z(win rate)
+ 0.05 × z(data confidence = log1p(battles))

Combat Effectiveness Score = clip(50 + 15 × z_total, 0, 100)
        """.strip()
    )

    st.write(
        "K/D and frags per battle are log1p-transformed before smoothing because "
        "they are skewed. 50 is roughly BR-average, ~65 is a strong step above, "
        "and ~95+ is exceptional. The top vehicle in a BR is not automatically "
        "100. Efficiency is intentionally excluded (it is already a composite). "
        "Vehicles without a Realistic BR are not scored."
    )

    st.subheader("Vehicles without a Realistic BR")
    st.info(
        "ThunderSkill returns **N/A** for BR metadata (country / type / battle "
        "rating) on a subset of vehicles, so they have no Realistic BR here. "
        "These rows are **kept** because their performance / sample data may "
        "still be real, but they are **omitted from BR-normalized views** — the "
        "Combat Effectiveness Score, the Nation × BR Range heatmap, and BR "
        "grouping — to avoid assigning them an incorrect BR."
    )

    excluded_no_br = vehicle_30d_df[~vehicle_30d_df["has_realistic_br"]].copy()
    st.metric("Excluded — missing BR metadata", f"{len(excluded_no_br):,}")

    if not excluded_no_br.empty:
        with st.expander("Show vehicles excluded for missing BR (top by sample battles)"):
            excluded_cols = [
                c for c in [
                    "vehicle_name",
                    "vehicle_slug",
                    "country",
                    "rank",
                    "total_battles_30d",
                    "observed_days",
                ]
                if c in excluded_no_br.columns
            ]
            st.dataframe(
                excluded_no_br[excluded_cols]
                .sort_values("total_battles_30d", ascending=False)
                .head(30),
                width="stretch",
                hide_index=True,
                column_config={
                    "vehicle_name": "Vehicle",
                    "vehicle_slug": "Slug",
                    "country": "Nation",
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "total_battles_30d": st.column_config.NumberColumn("Sample battles", format="%d"),
                    "observed_days": st.column_config.NumberColumn("Days", format="%d"),
                },
            )

    st.subheader("Current filtered vehicle dataframe")

    preview_cols = [
        "vehicle_name",
        "country",
        "vehicle_type",
        "realistic_br",
        "br_range_label",
        "battles",
        "days_observed",
        "win_rate",
        "ground_frags_per_battle",
        "ground_frags_per_death",
        "efficiency",
        "combat_effectiveness",
        "combat_effectiveness_legacy",
        "is_analysis_ready",
    ]
    preview_cols = [c for c in preview_cols if c in filtered_vehicle_df.columns]

    st.dataframe(
        filtered_vehicle_df[preview_cols]
        .sort_values("combat_effectiveness", ascending=False)
        .head(100),
        width="stretch",
        hide_index=True,
        height=500,
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Independent data science project by Adam Sanders / "
    "[War Thunder Stats](https://www.youtube.com/@warthunderstats). "
    "Data source: ThunderSkill."
)

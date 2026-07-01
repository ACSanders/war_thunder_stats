from io import BytesIO

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


def render_vehicle_image(image_url, caption: str = "", width: int = 260) -> None:
    """
    ThunderSkill blocks embedded/programmatic image rendering for many vehicle images.
    For now, show a clean placeholder and link to the image instead of a broken image.
    """
    st.info("Vehicle image available on ThunderSkill.")

    if image_url and str(image_url).startswith("http"):
        st.link_button("Open vehicle image", str(image_url))


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

st.title("War Thunder Stats")
st.caption(
    "Rolling 30-day Realistic Ground meta from automated ThunderSkill data. "
    "Scores describe observed player performance, not intrinsic vehicle strength."
)


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
max_battles = int(max(100, vehicle_30d_df["battles"].max()))

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

        with st.popover("More filters", width="stretch"):
            min_battles = st.slider(
                "Minimum 30-day sample battles",
                min_value=0,
                max_value=max_battles,
                value=50,
                step=10,
                key="flt_min_battles",
            )
            min_days_observed = st.slider(
                "Minimum days observed",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key="flt_min_days",
            )

# Normalize widget return values (empty multi -> all; None single -> All).
selected_ranges = selected_ranges or []
selected_countries = selected_countries or []
selected_types = selected_types or []
if premium_filter is None:
    premium_filter = "All"


def apply_vehicle_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "battles" in out.columns:
        out = out[out["battles"].fillna(0) >= min_battles]

    if "days_observed" in out.columns:
        out = out[out["days_observed"].fillna(0) >= min_days_observed]

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
n_ready = int(filtered_vehicle_df.get("is_analysis_ready", pd.Series(dtype=bool)).sum())

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
card_row2[1].metric("Analysis-ready", f"{n_ready:,}", border=True)
card_row2[2].metric("Rolling window", window_value, help=window_help, border=True)


# ============================================================
# Tabs
# ============================================================

tab_nation, tab_rankings, tab_trends, tab_data = st.tabs(
    [
        "Nation Meta",
        "Vehicle Rankings",
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

        # --- 4. Nation summary ---
        st.subheader("Nation strength summary")

        nation_summary = features.build_nation_summary(filtered_vehicle_df)
        if nation_summary.empty:
            st.info("No nation summary available under the current filters.")
        else:
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
        ranking_cols = [c for c in ranking_cols if c in filtered_vehicle_df.columns]

        rankings = (
            filtered_vehicle_df[ranking_cols]
            .sort_values(["combat_effectiveness", "battles"], ascending=[False, False])
            .head(25)
            .copy()
        )

        st.dataframe(
            rankings,
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

        rank_left, rank_right = st.columns(2, gap="large")

        with rank_left:
            st.markdown("**Top vehicles by CE Score**")
            top_chart_df = rankings.sort_values("combat_effectiveness", ascending=True)
            bar_fig = px.bar(
                top_chart_df,
                x="combat_effectiveness",
                y="vehicle_name",
                orientation="h",
                hover_data=[
                    c
                    for c in [
                        "country",
                        "realistic_br",
                        "battles",
                        "days_observed",
                        "win_rate",
                        "ground_frags_per_battle",
                        "ground_frags_per_death",
                    ]
                    if c in top_chart_df.columns
                ],
            )
            bar_fig.update_layout(
                xaxis_title="Combat Effectiveness Score",
                yaxis_title=None,
                height=520,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(bar_fig, width="stretch")

        with rank_right:
            st.markdown("**Sample battles vs CE Score**")
            scatter_df = filtered_vehicle_df.dropna(
                subset=["battles", "combat_effectiveness", "realistic_br"]
            ).copy()
            scatter_fig = px.scatter(
                scatter_df,
                x="battles",
                y="combat_effectiveness",
                color="country",
                size="realistic_br",
                hover_name="vehicle_name",
                hover_data=[
                    c
                    for c in [
                        "vehicle_type",
                        "realistic_br",
                        "days_observed",
                        "win_rate",
                        "ground_frags_per_battle",
                        "ground_frags_per_death",
                    ]
                    if c in scatter_df.columns
                ],
            )
            scatter_fig.update_layout(
                xaxis_title="30-day sample battles",
                yaxis_title="Combat Effectiveness Score",
                height=520,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(scatter_fig, width="stretch")

        st.divider()

        # --- folded-in vehicle detail (was the Vehicle Explorer tab) ---
        st.subheader("Vehicle detail")

        vehicle_options = (
            filtered_vehicle_df
            .sort_values(["country", "realistic_br", "vehicle_name"])
            ["vehicle_name"]
            .dropna()
            .unique()
        )

        selected_vehicle = st.selectbox(
            "Select a vehicle",
            options=vehicle_options,
            key="rankings_vehicle_select",
        )

        vehicle_row = filtered_vehicle_df[
            filtered_vehicle_df["vehicle_name"] == selected_vehicle
        ].head(1)

        if not vehicle_row.empty:
            row = vehicle_row.iloc[0]

            if not bool(row.get("has_realistic_br", True)):
                st.warning(
                    "No Realistic BR is available from ThunderSkill for this "
                    "vehicle. It is kept in the dataset because sample performance "
                    "data exists, but it is omitted from BR-normalized views such "
                    "as CE Score, BR heatmaps, and BR grouping."
                )

            card_left, card_right = st.columns([1, 2], gap="large")

            with card_left:
                render_vehicle_image(
                    image_url=row.get("pic"),
                    caption=row.get("vehicle_name", ""),
                    width=260,
                )

            with card_right:
                st.markdown(f"### {row.get('vehicle_name', 'Unknown vehicle')}")

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

                if "vehicle_url" in row and pd.notna(row["vehicle_url"]):
                    st.link_button("Open ThunderSkill page", row["vehicle_url"])

            vehicle_slug = row.get("vehicle_slug")
            vehicle_trend_df = filtered_recent_df[
                filtered_recent_df["vehicle_slug"] == vehicle_slug
            ].sort_values("date")

            if vehicle_trend_df.empty:
                st.info("No trend data available for this vehicle.")
            else:
                detail_metric = st.selectbox(
                    "Trend metric",
                    options=[
                        c
                        for c in [
                            "win_rate",
                            "ground_frags_per_battle",
                            "ground_frags_per_death",
                            "battles",
                        ]
                        if c in vehicle_trend_df.columns
                    ],
                    index=0,
                    key="rankings_detail_metric",
                )

                detail_fig = px.line(
                    vehicle_trend_df,
                    x="date",
                    y=detail_metric,
                    markers=True,
                    title=f"{selected_vehicle}: {detail_metric.replace('_', ' ').title()}",
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
    "Data source: ThunderSkill public vehicle pages. "
    "This app is an independent analytics project and is not affiliated with Gaijin or ThunderSkill."
)

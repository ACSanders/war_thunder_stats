from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
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
    initial_sidebar_state="expanded",
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
    """vehicle_agg_df: one row per vehicle with scores, BR-relative features,
    and quality flags."""
    vehicle_df = features.build_vehicle_agg(recent_daily_df)
    vehicle_df = features.add_combat_effectiveness(vehicle_df)
    vehicle_df = features.add_br_normalized(vehicle_df)
    vehicle_df = features.add_quality_flags(vehicle_df)
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


# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.title("Filters")

st.sidebar.caption(
    "Defaults remove sparse/low-battle vehicles and focus the app on stable 30-day performance."
)

if vehicle_30d_df.empty:
    st.error("No vehicle data available after recent-date filtering.")
    st.stop()

max_battles = int(max(100, vehicle_30d_df["battles"].max()))

min_battles = st.sidebar.slider(
    "Minimum 30-day battles",
    min_value=0,
    max_value=max_battles,
    value=50,
    step=10,
)

min_days_observed = st.sidebar.slider(
    "Minimum days observed",
    min_value=1,
    max_value=30,
    value=10,
    step=1,
)

br_values = sorted(vehicle_30d_df["realistic_br"].dropna().unique())

selected_brs = st.sidebar.multiselect(
    "Realistic BR",
    options=br_values,
    default=[],
    help="Leave empty to include all BRs.",
)

countries = sorted(vehicle_30d_df["country"].dropna().unique())

selected_countries = st.sidebar.multiselect(
    "Nation",
    options=countries,
    default=[],
    help="Leave empty to include all nations.",
)

vehicle_types = sorted(vehicle_30d_df["vehicle_type"].dropna().unique())

selected_types = st.sidebar.multiselect(
    "Vehicle type / role",
    options=vehicle_types,
    default=[],
    help="Leave empty to include all vehicle types.",
)

premium_filter = st.sidebar.radio(
    "Premium filter",
    options=["All", "Non-premium only", "Premium only"],
    index=1,
)

show_squadron = st.sidebar.checkbox("Include squadron vehicles", value=True)
show_pack = st.sidebar.checkbox("Include pack vehicles", value=True)
show_marketplace = st.sidebar.checkbox("Include marketplace vehicles", value=True)


def apply_vehicle_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "battles" in out.columns:
        out = out[out["battles"].fillna(0) >= min_battles]

    if "days_observed" in out.columns:
        out = out[out["days_observed"].fillna(0) >= min_days_observed]

    if selected_brs:
        out = out[out["realistic_br"].isin(selected_brs)]

    if selected_countries:
        out = out[out["country"].isin(selected_countries)]

    if selected_types:
        out = out[out["vehicle_type"].isin(selected_types)]

    if premium_filter == "Non-premium only" and "is_premium" in out.columns:
        out = out[out["is_premium"] == False]

    if premium_filter == "Premium only" and "is_premium" in out.columns:
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

nation_30d_df = features.build_nation_aggregate(filtered_vehicle_df)
br_30d_df = features.build_br_aggregate(filtered_vehicle_df)


# ============================================================
# Header
# ============================================================

st.title("War Thunder Stats")

st.caption(
    "Realistic Ground vehicle performance using automated ThunderSkill data. "
    "Rankings describe observed player performance, not intrinsic vehicle strength."
)

if "date" in raw_df.columns and not recent_df.empty:
    raw_min_date = raw_df["date"].min()
    raw_max_date = raw_df["date"].max()
    recent_min_date = recent_df["date"].min()
    recent_max_date = recent_df["date"].max()

    st.caption(
        f"App window: **{recent_min_date.date()} to {recent_max_date.date()}**. "
        f"Raw file date range: {raw_min_date.date()} to {raw_max_date.date()}."
    )

st.caption(
    "Vehicle rankings aggregate recent observations into one 30-day vehicle row. "
    "Battles are summed; performance metrics are battle-weighted averages."
)

metric_cols = st.columns(4)

metric_cols[0].metric("Vehicles", f"{filtered_vehicle_df['vehicle_slug'].nunique():,}")
metric_cols[1].metric("30-day battles", f"{int(filtered_vehicle_df['battles'].sum()):,}")
metric_cols[2].metric("Nations", f"{filtered_vehicle_df['country'].nunique():,}")

if not filtered_vehicle_df.empty:
    metric_cols[3].metric(
        "Avg CE v2",
        f"{filtered_vehicle_df['combat_effectiveness'].mean():.1f}",
    )
else:
    metric_cols[3].metric("Avg CE v2", "N/A")


# ============================================================
# Tabs
# ============================================================

tab_overview, tab_explorer, tab_nation_br, tab_trends, tab_data = st.tabs(
    [
        "Overview",
        "Vehicle Explorer",
        "Nation & BR Meta",
        "Trends",
        "Data Notes",
    ]
)


# ============================================================
# Overview tab
# ============================================================

with tab_overview:
    st.subheader("Top 30-day performers")

    st.write(
        "Combat Effectiveness v2 combines percentile-ranked ground frags per death, "
        "ground frags per battle, win rate, and efficiency."
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
            use_container_width=True,
            hide_index=True,
            height=520,
            column_config={
                "vehicle_name": "Vehicle",
                "country": "Nation",
                "vehicle_type": "Type",
                "realistic_br": st.column_config.NumberColumn("BR", format="%.1f"),
                "battles": st.column_config.NumberColumn("Battles", format="%d"),
                "days_observed": st.column_config.NumberColumn("Days", format="%d"),
                "win_rate": st.column_config.NumberColumn("Win rate", format="%.2f%%"),
                "ground_frags_per_battle": st.column_config.NumberColumn("Frags / battle", format="%.2f"),
                "ground_frags_per_death": st.column_config.NumberColumn("Frags / death", format="%.2f"),
                "efficiency": st.column_config.NumberColumn("Efficiency", format="%.1f"),
                "combat_effectiveness": st.column_config.NumberColumn("CE v2", format="%.1f"),
            },
        )

        st.divider()

        st.subheader("Top vehicles by Combat Effectiveness v2")

        top_chart_df = rankings.sort_values("combat_effectiveness", ascending=True)

        fig = px.bar(
            top_chart_df,
            x="combat_effectiveness",
            y="vehicle_name",
            orientation="h",
            hover_data=[
                c for c in [
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

        fig.update_layout(
            xaxis_title="Combat Effectiveness v2",
            yaxis_title=None,
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Battles vs Combat Effectiveness")

        scatter_df = filtered_vehicle_df.dropna(
            subset=["battles", "combat_effectiveness", "realistic_br"]
        ).copy()

        fig = px.scatter(
            scatter_df,
            x="battles",
            y="combat_effectiveness",
            color="country",
            size="realistic_br",
            hover_name="vehicle_name",
            hover_data=[
                c for c in [
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

        fig.update_layout(
            xaxis_title="30-day battles",
            yaxis_title="Combat Effectiveness v2",
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Vehicle Explorer tab
# ============================================================

with tab_explorer:
    st.subheader("Vehicle Explorer")

    if filtered_vehicle_df.empty:
        st.warning("No vehicles match the current filters.")
    else:
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
        )

        vehicle_row = filtered_vehicle_df[
            filtered_vehicle_df["vehicle_name"] == selected_vehicle
        ].head(1)

        if not vehicle_row.empty:
            row = vehicle_row.iloc[0]

            card_left, card_right = st.columns([1, 2])

            with card_left:
                render_vehicle_image(
                    image_url=row.get("pic"),
                    caption=row.get("vehicle_name", ""),
                    width=260,
                )

            with card_right:
                st.markdown(f"### {row.get('vehicle_name', 'Unknown vehicle')}")

                c1, c2 = st.columns(2)
                c1.metric("Nation", row.get("country", "N/A"))
                c2.metric("Type", row.get("vehicle_type", "N/A"))

                c3, c4 = st.columns(2)
                c3.metric("BR", f"{row.get('realistic_br', np.nan):.1f}")
                c4.metric("Rank", f"{row.get('rank', 'N/A')}")

                c5, c6 = st.columns(2)
                c5.metric("30-day battles", f"{int(row.get('battles', 0)):,}")
                c6.metric("Days observed", f"{int(row.get('days_observed', 0))}")

                c7, c8 = st.columns(2)
                c7.metric("Win rate", f"{row.get('win_rate', np.nan):.2f}%")
                c8.metric("CE v2", f"{row.get('combat_effectiveness', np.nan):.1f}")

                c9, c10 = st.columns(2)
                c9.metric("Frags / battle", f"{row.get('ground_frags_per_battle', np.nan):.2f}")
                c10.metric("Frags / death", f"{row.get('ground_frags_per_death', np.nan):.2f}")

                if "vehicle_url" in row and pd.notna(row["vehicle_url"]):
                    st.link_button("Open ThunderSkill page", row["vehicle_url"])

            st.divider()

            st.subheader("30-day vehicle trend")

            vehicle_slug = row.get("vehicle_slug")
            vehicle_trend_df = filtered_recent_df[
                filtered_recent_df["vehicle_slug"] == vehicle_slug
            ].sort_values("date")

            if vehicle_trend_df.empty:
                st.info("No trend data available for this vehicle.")
            else:
                trend_metric = st.selectbox(
                    "Trend metric",
                    options=[
                        c for c in [
                            "win_rate",
                            "ground_frags_per_battle",
                            "ground_frags_per_death",
                            "efficiency",
                            "battles",
                        ]
                        if c in vehicle_trend_df.columns
                    ],
                    index=0,
                )

                fig = px.line(
                    vehicle_trend_df,
                    x="date",
                    y=trend_metric,
                    markers=True,
                    title=f"{selected_vehicle}: {trend_metric.replace('_', ' ').title()}",
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=trend_metric.replace("_", " ").title(),
                    height=480,
                    margin=dict(l=10, r=10, t=50, b=10),
                )

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Show raw 30-day vehicle observations"):
                    st.dataframe(
                        vehicle_trend_df[
                            [
                                c for c in [
                                    "date",
                                    "battles",
                                    "win_rate",
                                    "ground_frags_per_battle",
                                    "ground_frags_per_death",
                                    "efficiency",
                                    "pic",
                                ]
                                if c in vehicle_trend_df.columns
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# Nation & BR Meta tab
# ============================================================

with tab_nation_br:
    st.subheader("Nation comparison")

    if filtered_vehicle_df.empty:
        st.warning("No vehicles match the current filters.")
    else:
        st.dataframe(
            nation_30d_df,
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "country": "Nation",
                "vehicles": st.column_config.NumberColumn("Vehicles", format="%d"),
                "battles": st.column_config.NumberColumn("Battles", format="%d"),
                "avg_win_rate": st.column_config.NumberColumn("Avg win rate", format="%.2f%%"),
                "avg_frags_per_battle": st.column_config.NumberColumn("Avg frags / battle", format="%.2f"),
                "avg_frags_per_death": st.column_config.NumberColumn("Avg frags / death", format="%.2f"),
                "avg_efficiency": st.column_config.NumberColumn("Avg efficiency", format="%.1f"),
                "avg_combat_effectiveness": st.column_config.NumberColumn("Avg CE v2", format="%.1f"),
            },
        )

        fig = px.bar(
            nation_30d_df.sort_values("avg_combat_effectiveness", ascending=True),
            x="avg_combat_effectiveness",
            y="country",
            orientation="h",
            hover_data=["vehicles", "battles", "avg_win_rate"],
        )

        fig.update_layout(
            xaxis_title="Average Combat Effectiveness v2",
            yaxis_title=None,
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("BR meta curve")

        fig = px.line(
            br_30d_df,
            x="realistic_br",
            y="avg_combat_effectiveness",
            markers=True,
            hover_data=["vehicles", "battles", "avg_win_rate"],
        )

        fig.update_layout(
            xaxis_title="Realistic BR",
            yaxis_title="Average Combat Effectiveness v2",
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Nation × BR heatmap")

        heatmap_metric = st.selectbox(
            "Heatmap metric",
            options=[
                "combat_effectiveness",
                "win_rate",
                "ground_frags_per_battle",
                "ground_frags_per_death",
                "efficiency",
            ],
            format_func=lambda x: x.replace("_", " ").title().replace(
                "Combat Effectiveness", "Combat Effectiveness v2"
            ),
        )

        min_heatmap_vehicles = st.slider(
            "Minimum vehicles per Nation × BR cell",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
        )

        heatmap_df = (
            filtered_vehicle_df
            .groupby(["country", "realistic_br"], as_index=False)
            .agg(
                value=(heatmap_metric, "mean"),
                vehicles=("vehicle_slug", "nunique"),
                battles=("battles", "sum"),
            )
        )

        heatmap_df = heatmap_df[heatmap_df["vehicles"] >= min_heatmap_vehicles]

        if heatmap_df.empty:
            st.warning("No heatmap cells match the current filters.")
        else:
            fig = px.density_heatmap(
                heatmap_df,
                x="realistic_br",
                y="country",
                z="value",
                histfunc="avg",
                hover_data=["vehicles", "battles"],
                color_continuous_scale="RdYlGn",
            )

            fig.update_layout(
                xaxis_title="Realistic BR",
                yaxis_title="Nation",
                height=520,
                margin=dict(l=10, r=10, t=30, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Trends tab
# ============================================================

with tab_trends:
    st.subheader("Daily trends")

    st.write(
        "Trend charts use the recent row-level observations after the same vehicle filters are applied."
    )

    if filtered_recent_df.empty:
        st.warning("No trend rows match the current filters.")
    else:
        trend_metric = st.selectbox(
            "Trend metric",
            options=[
                c for c in [
                    "win_rate",
                    "ground_frags_per_battle",
                    "ground_frags_per_death",
                    "efficiency",
                    "battles",
                ]
                if c in filtered_recent_df.columns
            ],
            index=0,
            key="global_trend_metric",
        )

        trend_group = st.radio(
            "Group trend by",
            options=["Nation", "BR"],
            horizontal=True,
        )

        if trend_group == "Nation":
            grouped = (
                filtered_recent_df
                .groupby(["date", "country"], as_index=False)
                .agg(value=(trend_metric, "mean"))
                .sort_values("date")
            )

            fig = px.line(
                grouped,
                x="date",
                y="value",
                color="country",
                markers=True,
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=trend_metric.replace("_", " ").title(),
                height=520,
                margin=dict(l=10, r=10, t=30, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            grouped = (
                filtered_recent_df
                .groupby(["date", "realistic_br"], as_index=False)
                .agg(value=(trend_metric, "mean"))
                .sort_values("date")
            )

            fig = px.line(
                grouped,
                x="date",
                y="value",
                color="realistic_br",
                markers=True,
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=trend_metric.replace("_", " ").title(),
                height=520,
                margin=dict(l=10, r=10, t=30, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Data Notes tab
# ============================================================

with tab_data:
    st.subheader("Dataset notes")

    st.write(
        "The app loads the latest automated ThunderSkill CSV from GitHub. "
        "The scraper collects up to 30 chart observations per Realistic Ground vehicle."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Raw rows", f"{len(raw_df):,}")
    c2.metric("Raw vehicles", f"{raw_df['vehicle_slug'].nunique():,}")
    c3.metric("Recent rows", f"{len(recent_df):,}")

    c4, c5, c6 = st.columns(3)

    c4.metric("30-day vehicle rows", f"{len(vehicle_30d_df):,}")
    c5.metric("Filtered vehicles", f"{len(filtered_vehicle_df):,}")
    c6.metric("Filtered battles", f"{int(filtered_vehicle_df['battles'].sum()):,}")

    if "date" in raw_df.columns:
        st.write("Raw date range:", raw_df["date"].min(), "to", raw_df["date"].max())
        st.write("App recent date range:", recent_df["date"].min(), "to", recent_df["date"].max())

    st.subheader("Combat Effectiveness v2")

    st.write(
        "Combat Effectiveness v2 is a percentile-weighted descriptive score:"
    )

    st.code(
        """
CE v2 =
0.35 × percentile(ground frags per death)
+ 0.35 × percentile(ground frags per battle)
+ 0.20 × percentile(win rate)
+ 0.10 × percentile(efficiency)
        """.strip()
    )

    st.write(
        "This avoids the main weakness of a raw weighted sum: metrics with larger scales "
        "can dominate the score. Percentile ranking makes the components comparable."
    )

    st.subheader("Current filtered vehicle dataframe")

    preview_cols = [
        "vehicle_name",
        "country",
        "vehicle_type",
        "realistic_br",
        "battles",
        "days_observed",
        "win_rate",
        "ground_frags_per_battle",
        "ground_frags_per_death",
        "efficiency",
        "combat_effectiveness",
        "pic",
    ]

    preview_cols = [c for c in preview_cols if c in filtered_vehicle_df.columns]

    st.dataframe(
        filtered_vehicle_df[preview_cols]
        .sort_values("combat_effectiveness", ascending=False)
        .head(100),
        use_container_width=True,
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
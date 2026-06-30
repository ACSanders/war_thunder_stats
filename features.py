"""Pure pandas feature-engineering helpers for the War Thunder Stats app.

This module intentionally has **no Streamlit imports**. It is the single source
of truth for:

  * cleaning / typing the raw ThunderSkill CSV          -> clean_daily()
  * safe metadata fallbacks (country / type / rank)     -> apply_metadata_fallbacks()
  * the recent 30-day window                            -> recent_window()
  * the one-row-per-vehicle aggregate                   -> build_vehicle_agg()
  * descriptive scoring + BR-relative scoring           -> add_combat_effectiveness() / add_br_normalized()
  * data-quality flags                                  -> add_quality_flags()

Keeping these as plain functions means the offline clustering / precompute
scripts (future phases) can reuse the exact same logic. Streamlit-specific
caching wrappers live in streamlit_app.py.

Two canonical dataframes are produced from the raw CSV:

  cleaned_daily_df : one row per vehicle per day (typed, backfilled).
  vehicle_agg_df   : one row per vehicle (aggregated 30-day window + features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Config / thresholds  (named so they are easy to tune later)
# ============================================================

RECENT_WINDOW_DAYS = 30

# Quality-flag thresholds (see add_quality_flags).
MIN_SUFFICIENT_DAYS = 5
MIN_SUFFICIENT_BATTLES = 20

# Within-BR percentiles are noisy when a bracket has too few vehicles.
MIN_BRACKET_VEHICLES = 5

# Sentinel strings that should be treated as missing in text columns.
_NULL_TOKENS = {"", "nan", "none", "null"}

# index_country is UPPERCASE (USSR / USA / SWEDEN / BRITAIN ...).
# country (from the page scrape) is title-case (Sweden / Britain ...).
# Map both to one canonical spelling before backfilling.
COUNTRY_CANONICAL = {
    "USA": "USA",
    "USSR": "USSR",
    "BRITAIN": "Britain",
    "GERMANY": "Germany",
    "JAPAN": "Japan",
    "ITALY": "Italy",
    "FRANCE": "France",
    "SWEDEN": "Sweden",
    "CHINA": "China",
    "ISRAEL": "Israel",
}

NUMERIC_COLS = [
    "realistic_br",
    "arcade_br",
    "simulator_br",
    "rank",
    "battles",
    "win_rate",
    "efficiency",
    "air_frags_per_battle",
    "air_frags_per_death",
    "ground_frags_per_battle",
    "ground_frags_per_death",
    "index_rank",
    "index_battles",
    "index_win_rate",
    "index_efficiency",
]

BOOL_COLS = [
    "is_premium",
    "is_squadron",
    "is_pack",
    "on_marketplace",
]

TEXT_COLS = [
    "vehicle_slug",
    "vehicle_name",
    "vehicle_url",
    "pic",
    "country",
    "vehicle_type",
    "mode",
    "index_role",
    "index_type",
    "index_country",
]

# Core per-battle performance metrics (battle-weighted on aggregation).
CORE_PERF_COLS = [
    "win_rate",
    "efficiency",
    "ground_frags_per_battle",
    "ground_frags_per_death",
    "air_frags_per_battle",
    "air_frags_per_death",
]

# Stable per-vehicle metadata carried into the aggregate (first non-null value).
META_COLS = [
    "vehicle_name",
    "vehicle_url",
    "pic",
    "country",
    "vehicle_type",
    "rank",
    "realistic_br",
    "is_premium",
    "is_squadron",
    "is_pack",
    "on_marketplace",
    "release_date_raw",
    "vehicle_id",
    "mode",
]

# Combat Effectiveness weights (shared by global CE and BR-relative CE).
CE_WEIGHTS = {
    "ground_frags_per_death": 0.35,
    "ground_frags_per_battle": 0.35,
    "win_rate": 0.20,
    "efficiency": 0.10,
}


# ============================================================
# Small helpers
# ============================================================

def _clean_text_series(s: pd.Series) -> pd.Series:
    """Strip whitespace and convert sentinel strings ('nan', '', ...) to NaN.

    The previous loader used ``astype(str).str.strip()`` which turned real NaN
    into the literal string ``"nan"`` -- that leaked a phantom 'nan' nation into
    filters. This nullifies those sentinels instead.
    """
    s = s.astype(str).str.strip()
    return s.mask(s.str.lower().isin(_NULL_TOKENS), np.nan)


def _canon_country(val) -> object:
    """Return a canonical, title-cased country name (or NaN)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in _NULL_TOKENS:
        return np.nan
    return COUNTRY_CANONICAL.get(s.upper(), s.title())


def _first_valid(s: pd.Series):
    """First non-null value in a group (NaN if the group is all-null)."""
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Battle-weighted mean.

    Unlike the original helper, this returns NaN when there is no valid weight
    rather than silently falling back to an unweighted mean (which fabricated a
    value for near-empty vehicles).
    """
    valid = values.notna() & weights.notna() & weights.gt(0)
    if valid.any():
        return float(np.average(values.loc[valid], weights=weights.loc[valid]))
    return float("nan")


# ============================================================
# Cleaning + metadata fallbacks  -> cleaned_daily_df
# ============================================================

def apply_metadata_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill stable metadata from the index columns when the page scrape
    left them missing. Existing (non-null) values are never overwritten.

      country      <- normalize(index_country)   [casing-normalized]
      vehicle_type <- index_role                 [NOT index_type, which is "Ground forces"]
      rank         <- index_rank                 [rank stays source of truth]

    realistic_br is intentionally NOT backfilled: the index carries no BR.
    """
    out = df.copy()

    if "country" in out.columns:
        out["country"] = out["country"].map(_canon_country)
        if "index_country" in out.columns:
            idx_country = out["index_country"].map(_canon_country)
            out["country"] = out["country"].fillna(idx_country)

    if "vehicle_type" in out.columns and "index_role" in out.columns:
        out["vehicle_type"] = out["vehicle_type"].fillna(out["index_role"])

    if "rank" in out.columns and "index_rank" in out.columns:
        out["rank"] = out["rank"].fillna(out["index_rank"])

    return out


def clean_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build cleaned_daily_df: one typed, backfilled row per vehicle per day.

    Preserves the daily-row structure of the CSV (no rows dropped here).
    """
    df = raw_df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BOOL_COLS:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .map({"true": True, "false": False})
                    .fillna(False)
                )
            else:
                df[col] = df[col].fillna(False).astype(bool)

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = _clean_text_series(df[col])

    # Clean vehicle image URLs. Do NOT invent image URLs from vehicle_slug;
    # ThunderSkill filenames are not guaranteed to match the slug.
    if "pic" in df.columns:
        relative_pic_mask = df["pic"].notna() & df["pic"].str.startswith("/")
        df.loc[relative_pic_mask, "pic"] = (
            "https://thunderskill.com" + df.loc[relative_pic_mask, "pic"]
        )

    df = apply_metadata_fallbacks(df)

    return df


def recent_window(df: pd.DataFrame, days: int = RECENT_WINDOW_DAYS) -> pd.DataFrame:
    """Keep observations within the most recent global date window.

    The pipeline pulls up to 30 chart observations per vehicle. Most vehicles
    have recent daily data, but sparse event vehicles may carry very old
    observations; this drops those from the app's default views.
    """
    out = df.copy()

    if "date" not in out.columns or out["date"].isna().all():
        return out

    max_date = out["date"].max()
    min_date = max_date - pd.Timedelta(days=days - 1)

    return out[(out["date"] >= min_date) & (out["date"] <= max_date)].copy()


# ============================================================
# Vehicle aggregation  -> vehicle_agg_df
# ============================================================

def _weighted_means(
    df: pd.DataFrame,
    metrics: list[str],
    weight_col: str = "battles_weight",
    group_col: str = "vehicle_slug",
) -> pd.DataFrame:
    """Battle-weighted mean per vehicle for each metric.

    Vectorized: sum(metric * weight) / sum(weight) over rows where the metric is
    present and weight > 0. Returns NaN when a vehicle has no valid weighted
    rows (e.g. zero total battles), so we never emit a misleading 0.
    """
    w = df[weight_col].fillna(0).clip(lower=0)
    grp = df[group_col]

    result = {}
    for m in metrics:
        valid = df[m].notna() & (w > 0)
        num = (df[m] * w).where(valid, 0.0).groupby(grp).sum()
        den = w.where(valid, 0.0).groupby(grp).sum()
        result[m] = num / den.replace(0, np.nan)

    return pd.DataFrame(result)


def build_vehicle_agg(recent_df: pd.DataFrame) -> pd.DataFrame:
    """Build vehicle_agg_df: one row per vehicle from the recent window.

    Grouped by vehicle_slug only (not volatile columns like pic /
    release_date_raw). Battles are summed; performance metrics are
    battle-weighted averages; metadata is the first non-null value.
    """
    if recent_df.empty:
        return recent_df.copy()

    df = recent_df.copy()
    df["battles_weight"] = df["battles"].fillna(0).clip(lower=0)

    meta_cols = [c for c in META_COLS if c in df.columns]
    perf_cols = [c for c in CORE_PERF_COLS if c in df.columns]

    grp = df.groupby("vehicle_slug", dropna=False)

    meta = grp[meta_cols].agg(_first_valid)

    agg = pd.DataFrame(
        {
            "battles": grp["battles"].sum(),       # NaN treated as 0
            "observations": grp.size(),
        }
    )

    if "date" in df.columns:
        agg["first_observed"] = grp["date"].min()
        agg["last_observed"] = grp["date"].max()
        agg["days_observed"] = grp["date"].nunique()

    wmeans = _weighted_means(df, perf_cols)

    out = meta.join(agg).join(wmeans).reset_index()

    return out


# ============================================================
# Scoring
# ============================================================

def add_combat_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Combat Effectiveness v2 (global). Formula is unchanged from the original
    app: percentile-rank each metric across the whole frame, then weight.

    Note: global CE values may shift slightly (observed <= 0.2 pts; top-25
    membership unchanged) versus the pre-refactor app. This is intentional and
    not a formula change -- the upstream battle-weighted average now returns NaN
    when a vehicle has no valid battle-weighted rows instead of fabricating an
    unweighted mean (see weighted_average / _weighted_means). Excluding those
    fabricated values changes the percentile denominators feeding CE.
    """
    out = df.copy()

    available = [c for c in CE_WEIGHTS if c in out.columns]

    if not available:
        out["combat_effectiveness"] = np.nan
        out["meta_score"] = np.nan
        return out

    for col in available:
        out[f"{col}_pct"] = out[col].rank(pct=True)

    weight_sum = sum(CE_WEIGHTS[c] for c in available)

    out["combat_effectiveness"] = sum(
        out[f"{col}_pct"] * CE_WEIGHTS[col] for col in available
    ) / weight_sum

    out["combat_effectiveness"] = (out["combat_effectiveness"] * 100).round(1)

    # Alias kept for compatibility with older chart logic.
    out["meta_score"] = out["combat_effectiveness"]

    return out


def assign_br_bracket(br: pd.Series) -> pd.Series:
    """Map BR -> a 1.0-wide bracket key (floor). NaN BR stays NaN.

    Isolated here so a later phase can swap in true overlapping lineup windows
    (e.g. 5.0-5.7) without touching call sites.
    """
    return np.floor(br)


def _pct_within_bracket(
    df: pd.DataFrame,
    metric: str,
    bracket_col: str,
    min_n: int = MIN_BRACKET_VEHICLES,
) -> pd.Series:
    """Percentile rank of ``metric`` within each BR bracket.

    Brackets with fewer than ``min_n`` non-null values yield NaN (too noisy).
    Rows with a NaN bracket (missing BR) also yield NaN.
    """
    s = pd.Series(np.nan, index=df.index, dtype="float64")

    for _, idx in df.groupby(bracket_col).groups.items():
        sub = df.loc[idx, metric]
        if sub.notna().sum() >= min_n:
            s.loc[idx] = sub.rank(pct=True)

    return s


def add_br_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Add BR-relative features (additive; nothing existing is changed).

    Adds:
      br_bracket                  : 1.0-wide BR band key.
      <metric>_pct_br             : percentile of metric within its BR bracket.
      combat_effectiveness_br     : CE computed from the within-BR percentiles,
                                    i.e. "good for its BR" rather than "high BR".
    """
    out = df.copy()

    if "realistic_br" in out.columns:
        out["br_bracket"] = assign_br_bracket(out["realistic_br"])
    else:
        out["br_bracket"] = np.nan

    metrics = [c for c in CE_WEIGHTS if c in out.columns]
    for m in metrics:
        out[f"{m}_pct_br"] = _pct_within_bracket(out, m, "br_bracket")

    available = [c for c in CE_WEIGHTS if f"{c}_pct_br" in out.columns]

    if available:
        weight_sum = sum(CE_WEIGHTS[c] for c in available)
        ce_br = sum(out[f"{c}_pct_br"] * CE_WEIGHTS[c] for c in available) / weight_sum
        out["combat_effectiveness_br"] = (ce_br * 100).round(1)
    else:
        out["combat_effectiveness_br"] = np.nan

    return out


# ============================================================
# Quality flags
# ============================================================

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add data-quality flags. These are informational for now; no global
    filtering is applied by this module."""
    out = df.copy()

    out["observed_days"] = out["days_observed"] if "days_observed" in out.columns else np.nan
    out["total_battles_30d"] = out["battles"] if "battles" in out.columns else np.nan

    if "realistic_br" in out.columns:
        out["has_realistic_br"] = out["realistic_br"].notna()
    else:
        out["has_realistic_br"] = False

    out["has_sufficient_dates"] = out["observed_days"].fillna(0) >= MIN_SUFFICIENT_DAYS
    out["has_sufficient_battles"] = out["total_battles_30d"].fillna(0) >= MIN_SUFFICIENT_BATTLES

    perf_present = [c for c in CORE_PERF_COLS if c in out.columns]
    if perf_present:
        out["has_performance_data"] = out[perf_present].notna().any(axis=1)
    else:
        out["has_performance_data"] = False

    out["is_analysis_ready"] = (
        out["has_realistic_br"]
        & out["has_sufficient_dates"]
        & out["has_sufficient_battles"]
    )

    return out


# ============================================================
# Group aggregates (nation / BR) -- unchanged behavior
# ============================================================

def build_nation_aggregate(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    if vehicle_df.empty:
        return vehicle_df.copy()

    return (
        vehicle_df
        .groupby("country", as_index=False)
        .agg(
            vehicles=("vehicle_slug", "nunique"),
            battles=("battles", "sum"),
            avg_win_rate=("win_rate", "mean"),
            avg_frags_per_battle=("ground_frags_per_battle", "mean"),
            avg_frags_per_death=("ground_frags_per_death", "mean"),
            avg_efficiency=("efficiency", "mean"),
            avg_combat_effectiveness=("combat_effectiveness", "mean"),
        )
        .sort_values("avg_combat_effectiveness", ascending=False)
    )


def build_br_aggregate(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    if vehicle_df.empty:
        return vehicle_df.copy()

    return (
        vehicle_df
        .groupby("realistic_br", as_index=False)
        .agg(
            vehicles=("vehicle_slug", "nunique"),
            battles=("battles", "sum"),
            avg_win_rate=("win_rate", "mean"),
            avg_frags_per_battle=("ground_frags_per_battle", "mean"),
            avg_frags_per_death=("ground_frags_per_death", "mean"),
            avg_efficiency=("efficiency", "mean"),
            avg_combat_effectiveness=("combat_effectiveness", "mean"),
        )
        .sort_values("realistic_br")
    )

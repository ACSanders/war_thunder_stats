"""Pure pandas feature-engineering helpers for the War Thunder Stats app.

This module intentionally has **no Streamlit imports**. It is the single source
of truth for:

  * cleaning / typing the raw ThunderSkill CSV          -> clean_daily()
  * safe metadata fallbacks (country / type / rank)     -> apply_metadata_fallbacks()
  * manual War Thunder Wiki BR overrides                -> apply_wiki_br_overrides()
  * the recent 30-day window                            -> recent_window()
  * the one-row-per-vehicle aggregate                   -> build_vehicle_agg()
  * BR-relative Combat Effectiveness Score              -> add_combat_effectiveness()
  * broad lineup-style BR range fields                  -> add_br_ranges()
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

# ------------------------------------------------------------
# Combat Effectiveness Score (BR-relative, empirical-Bayes)
# ------------------------------------------------------------
# The official score is computed within each EXACT realistic_br peer group.
# Metric weights (must sum with CE_CONFIDENCE_WEIGHT to 1.0). efficiency is
# deliberately excluded -- it is already a ThunderSkill-style composite.
CE_METRIC_WEIGHTS = {
    "ground_frags_per_death": 0.40,   # K/D
    "ground_frags_per_battle": 0.40,
    "win_rate": 0.15,
}
CE_CONFIDENCE_WEIGHT = 0.05           # within-BR standardized log1p(battles)

# Metrics that get a log1p transform before smoothing/standardizing (skewed).
CE_LOG1P_METRICS = {"ground_frags_per_death", "ground_frags_per_battle"}

# Empirical-Bayes shrinkage: reliability = battles / (battles + PRIOR_BATTLES).
PRIOR_BATTLES = 100

# 0-100 mapping: score = SCORE_CENTER + SCORE_SLOPE * z_total (clipped 0-100).
SCORE_CENTER = 50.0
SCORE_SLOPE = 15.0

# Robustness guards for the within-BR standardization.
MIN_SCORE_PEERS = 8         # BRs with fewer scoreable vehicles -> NaN score
ROBUST_SCALE_FLOOR = 1e-6   # avoids divide-by-near-zero when a BR is flat
CONFIDENCE_Z_CLIP = 3.0     # bound the confidence term so big samples can't run away

# Legacy global percentile formula (kept only for combat_effectiveness_legacy).
CE_LEGACY_WEIGHTS = {
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


# ============================================================
# Manual War Thunder Wiki BR overrides
# ============================================================
# ThunderSkill occasionally exposes stale realistic_br / arcade_br /
# simulator_br metadata after War Thunder changes a vehicle's BR (e.g.
# ussr_kv_1s: ThunderSkill reports 4.0, the official Wiki reports 4.3). This
# lets a small, manually-maintained lookup CSV correct those columns before
# CE / clustering / any other BR-relative feature is computed.

WIKI_BR_LOOKUP_COLS = ["wiki_arcade_br", "wiki_realistic_br", "wiki_simulator_br"]

# ThunderSkill BR column -> the wiki lookup column that can override it.
_BR_OVERRIDE_MAP = {
    "arcade_br": "wiki_arcade_br",
    "realistic_br": "wiki_realistic_br",
    "simulator_br": "wiki_simulator_br",
}


def apply_wiki_br_overrides(df: pd.DataFrame, lookup_df: pd.DataFrame | None) -> pd.DataFrame:
    """Override stale ThunderSkill BR columns with trusted War Thunder Wiki values.

    Merges ``lookup_df`` onto ``df`` by vehicle_slug and prefers each wiki_*_br
    column wherever it is non-null; ThunderSkill values are kept for rows the
    lookup doesn't cover. Original ThunderSkill BRs are preserved as
    thunderskill_arcade_br / thunderskill_realistic_br / thunderskill_simulator_br
    so both are inspectable. Adds br_overridden (bool) and br_source
    ("war_thunder_wiki" or "thunderskill").

    Safe no-op (returns ``df`` unchanged) if ``df`` has no vehicle_slug column,
    or ``lookup_df`` is None/empty, has no vehicle_slug column, or has none of
    the wiki_*_br columns. Supports any number of lookup rows without code
    changes.
    """
    if "vehicle_slug" not in df.columns:
        return df

    if lookup_df is None or lookup_df.empty or "vehicle_slug" not in lookup_df.columns:
        return df

    wiki_cols = [c for c in WIKI_BR_LOOKUP_COLS if c in lookup_df.columns]
    if not wiki_cols:
        return df

    lookup = lookup_df[["vehicle_slug", *wiki_cols]].copy()
    lookup["vehicle_slug"] = _clean_text_series(lookup["vehicle_slug"])
    lookup = lookup.dropna(subset=["vehicle_slug"]).drop_duplicates(
        subset="vehicle_slug", keep="last"
    )

    for col in wiki_cols:
        lookup[col] = pd.to_numeric(lookup[col], errors="coerce")

    out = df.copy()

    for br_col, saved_col in [
        ("arcade_br", "thunderskill_arcade_br"),
        ("realistic_br", "thunderskill_realistic_br"),
        ("simulator_br", "thunderskill_simulator_br"),
    ]:
        if br_col in out.columns:
            out[saved_col] = out[br_col]

    out = out.merge(lookup, on="vehicle_slug", how="left")

    overridden = pd.Series(False, index=out.index)
    for br_col, wiki_col in _BR_OVERRIDE_MAP.items():
        if br_col not in out.columns or wiki_col not in out.columns:
            continue
        has_override = out[wiki_col].notna()
        out[br_col] = out[wiki_col].where(has_override, out[br_col])
        overridden = overridden | has_override

    out["br_overridden"] = overridden
    out["br_source"] = np.where(overridden, "war_thunder_wiki", "thunderskill")

    out = out.drop(columns=wiki_cols)

    return out


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

# Short debug-column names for the per-metric within-BR z-scores.
_CE_METRIC_SHORT = {
    "ground_frags_per_death": "kd",
    "ground_frags_per_battle": "fpb",
    "win_rate": "wr",
}


def _within_br_robust_z(
    values: pd.Series,
    br: pd.Series,
    clip: float | None = None,
) -> pd.Series:
    """Robust z-score of ``values`` within each exact-BR peer group.

    Uses median + MAD (scaled by 1.4826 to be normal-consistent). Guards:
      * BRs with fewer than MIN_SCORE_PEERS non-null values -> NaN.
      * A flat BR (scale below ROBUST_SCALE_FLOOR) -> z = 0 (everyone average).
      * Rows with a NaN value or NaN BR -> NaN.
    """
    med = values.groupby(br).transform("median")
    mad = (values - med).abs().groupby(br).transform("median")
    scale = 1.4826 * mad
    count = values.groupby(br).transform("count")

    has_spread = scale >= ROBUST_SCALE_FLOOR
    z = (values - med) / scale.where(has_spread)
    z = z.where(has_spread, 0.0)                 # flat BR -> everyone == average
    z = z.where(values.notna(), np.nan)          # missing value -> NaN
    z = z.where(count >= MIN_SCORE_PEERS, np.nan)  # too few peers -> NaN

    if clip is not None:
        z = z.clip(-clip, clip)

    return z


def add_combat_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Combat Effectiveness Score: BR-relative overperformance, 0-100.

    Computed within each EXACT realistic_br peer group:

      1. transform skewed metrics with log1p (K/D, frags per battle);
         win rate is used as-is.
      2. empirical-Bayes shrinkage toward the exact-BR center, with
         reliability = battles / (battles + PRIOR_BATTLES). Low-battle vehicles
         are pulled strongly toward the BR average; high-battle vehicles keep
         more of their observed value.
      3. robust within-BR standardization (median + MAD) of the smoothed metric.
      4. weighted combine (40% K/D, 40% frags/battle, 15% win rate, 5% a
         within-BR confidence term = standardized log1p(battles), clipped).
      5. score = 50 + 15 * z_total, clipped to 0-100.

    50 = roughly BR-average; ~65 = a strong step above; ~95+ = exceptional.
    Because it is robust-z (not percentile / min-max), the top vehicle in a BR
    is NOT automatically 100. efficiency is intentionally excluded.

    Missing-BR or signal-less (no battles) vehicles get NaN (not scoreable).
    Adds debug columns: reliability, confidence_z, {kd,fpb,wr}_z_br.
    """
    out = df.copy()

    if "realistic_br" not in out.columns:
        out["combat_effectiveness"] = np.nan
        out["meta_score"] = np.nan
        out["reliability"] = np.nan
        return out

    br = out["realistic_br"]
    battles = out["total_battles_30d"] if "total_battles_30d" in out.columns else out.get("battles")
    battles = pd.to_numeric(battles, errors="coerce").fillna(0).clip(lower=0)

    reliability = battles / (battles + PRIOR_BATTLES)
    out["reliability"] = reliability.round(4)

    # --- per-metric smoothed, standardized z within exact BR ---
    metric_terms = {}  # metric -> (weight, z Series)
    for metric, weight in CE_METRIC_WEIGHTS.items():
        if metric not in out.columns:
            continue
        raw = pd.to_numeric(out[metric], errors="coerce")
        t = np.log1p(raw) if metric in CE_LOG1P_METRICS else raw
        center = t.groupby(br).transform("median")          # exact-BR prior center
        t_smooth = reliability * t + (1.0 - reliability) * center
        z = _within_br_robust_z(t_smooth, br)
        out[f"{_CE_METRIC_SHORT[metric]}_z_br"] = z
        metric_terms[metric] = (weight, z)

    # --- confidence / stability term (within-BR standardized log1p battles) ---
    conf_z = _within_br_robust_z(np.log1p(battles), br, clip=CONFIDENCE_Z_CLIP)
    out["confidence_z"] = conf_z

    # --- weighted combine with per-row renormalization over available terms ---
    num = pd.Series(0.0, index=out.index)
    den = pd.Series(0.0, index=out.index)
    n_metric = pd.Series(0, index=out.index)

    for _, (weight, z) in metric_terms.items():
        avail = z.notna()
        num = num + (weight * z).where(avail, 0.0)
        den = den + pd.Series(weight, index=out.index).where(avail, 0.0)
        n_metric = n_metric + avail.astype(int)

    # Scoreable requires a real BR and at least one usable metric (so a 0-battle
    # vehicle is never scored "average" off the confidence term alone).
    scoreable = br.notna() & (n_metric >= 1)

    conf_avail = conf_z.notna() & scoreable
    num = num + (CE_CONFIDENCE_WEIGHT * conf_z).where(conf_avail, 0.0)
    den = den + pd.Series(CE_CONFIDENCE_WEIGHT, index=out.index).where(conf_avail, 0.0)

    z_total = (num / den.where(den > 0)).where(scoreable, np.nan)
    score = (SCORE_CENTER + SCORE_SLOPE * z_total).clip(0, 100).round(1)

    out["combat_effectiveness"] = score.where(scoreable, np.nan)
    out["meta_score"] = out["combat_effectiveness"]  # vestigial alias

    return out


def add_combat_effectiveness_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy global-percentile score, kept only as combat_effectiveness_legacy
    for validation/debug. This is the pre-change formula (incl. efficiency)."""
    out = df.copy()

    available = [c for c in CE_LEGACY_WEIGHTS if c in out.columns]
    if not available:
        out["combat_effectiveness_legacy"] = np.nan
        return out

    weight_sum = sum(CE_LEGACY_WEIGHTS[c] for c in available)
    legacy = sum(
        out[c].rank(pct=True) * CE_LEGACY_WEIGHTS[c] for c in available
    ) / weight_sum
    out["combat_effectiveness_legacy"] = (legacy * 100).round(1)

    return out


def assign_br_bracket(br: pd.Series) -> pd.Series:
    """Map BR -> a 1.0-wide bracket key (floor). NaN BR stays NaN."""
    return np.floor(br)


def add_br_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Add broad lineup-style BR range fields (future-ready: HDBSCAN, sleepers,
    lineup ranker, meta views). These are NOT the official scoring peer group --
    the score uses exact realistic_br. Each range spans X.0 to X.7.

    Adds: br_bracket (floor), br_range_min, br_range_max, br_range_label.
    """
    out = df.copy()

    if "realistic_br" in out.columns:
        floor = assign_br_bracket(out["realistic_br"])
    else:
        floor = pd.Series(np.nan, index=out.index)

    out["br_bracket"] = floor
    out["br_range_min"] = floor
    out["br_range_max"] = floor + 0.7
    out["br_range_label"] = floor.apply(
        lambda x: f"{x:.1f}–{x + 0.7:.1f}" if pd.notna(x) else np.nan
    )

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


# ============================================================
# Nation Meta aggregates (for the redesigned Nation Meta tab)
# ============================================================

def build_nation_br_heatmap(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (country, br_range_label) for the Nation x BR Range heatmap.

    win_rate_bw is the battle-weighted win rate over the 30-day window; the
    avg_* fields and counts are for hover. Requires br_range_label / br_range_min
    (from add_br_ranges) on the vehicle-level frame.
    """
    if vehicle_df.empty or "br_range_label" not in vehicle_df.columns:
        return pd.DataFrame()

    df = vehicle_df.dropna(subset=["country", "br_range_label"]).copy()
    if df.empty:
        return pd.DataFrame()

    w = df["battles"].fillna(0).clip(lower=0)
    df["_wr_num"] = (df["win_rate"] * w).where(df["win_rate"].notna(), 0.0)
    df["_wr_den"] = w.where(df["win_rate"].notna(), 0.0)

    out = df.groupby(["country", "br_range_label"], as_index=False).agg(
        br_range_min=("br_range_min", "min"),
        battles=("battles", "sum"),
        vehicles=("vehicle_slug", "nunique"),
        avg_kd=("ground_frags_per_death", "mean"),
        avg_fpb=("ground_frags_per_battle", "mean"),
        avg_ce=("combat_effectiveness", "mean"),
        _wr_num=("_wr_num", "sum"),
        _wr_den=("_wr_den", "sum"),
    )

    out["win_rate_bw"] = out["_wr_num"] / out["_wr_den"].where(out["_wr_den"] > 0)
    out = out.drop(columns=["_wr_num", "_wr_den"])

    return out.sort_values(["country", "br_range_min"]).reset_index(drop=True)


def build_nation_daily_trend(recent_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, country) with battle-weighted daily metrics.

    Metrics are battle-weighted across that nation's vehicles on each day:
    win_rate, ground_frags_per_death, ground_frags_per_battle; battles is summed.
    CE Score is a window-level vehicle score and is deliberately not produced
    here. Rolling smoothing is applied in the app layer.
    """
    if recent_df.empty:
        return pd.DataFrame()

    df = recent_df.dropna(subset=["date", "country"]).copy()
    if df.empty:
        return pd.DataFrame()

    w = df["battles"].fillna(0).clip(lower=0)
    metrics = ["win_rate", "ground_frags_per_death", "ground_frags_per_battle"]

    agg_kwargs = {"battles": ("battles", "sum")}
    for m in metrics:
        df[f"_{m}_num"] = (df[m] * w).where(df[m].notna(), 0.0)
        df[f"_{m}_den"] = w.where(df[m].notna(), 0.0)
        agg_kwargs[f"_{m}_num"] = (f"_{m}_num", "sum")
        agg_kwargs[f"_{m}_den"] = (f"_{m}_den", "sum")

    out = df.groupby(["date", "country"], as_index=False).agg(**agg_kwargs)

    for m in metrics:
        out[m] = out[f"_{m}_num"] / out[f"_{m}_den"].where(out[f"_{m}_den"] > 0)

    keep = ["date", "country", "battles"] + metrics
    return out[keep].sort_values(["country", "date"]).reset_index(drop=True)


def build_nation_summary(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    """Nation-level summary cards/table: CE central tendency, totals, top BR
    range (by battles), and the nation's best vehicle by CE Score."""
    if vehicle_df.empty:
        return pd.DataFrame()

    df = vehicle_df.dropna(subset=["country"]).copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for country, g in df.groupby("country"):
        ce = g["combat_effectiveness"]

        scored = g.dropna(subset=["combat_effectiveness"])
        if not scored.empty:
            best = scored.loc[scored["combat_effectiveness"].idxmax()]
            best_vehicle = best.get("vehicle_name", np.nan)
            best_vehicle_ce = best.get("combat_effectiveness", np.nan)
        else:
            best_vehicle = np.nan
            best_vehicle_ce = np.nan

        top_br_range = np.nan
        if "br_range_label" in g.columns and g["br_range_label"].notna().any():
            by_range = (
                g.dropna(subset=["br_range_label"])
                .groupby("br_range_label")["battles"].sum()
            )
            if not by_range.empty:
                top_br_range = by_range.idxmax()

        rows.append(
            {
                "country": country,
                "vehicles": g["vehicle_slug"].nunique(),
                "battles": g["battles"].sum(),
                "avg_ce": ce.mean(),
                "median_ce": ce.median(),
                "top_br_range": top_br_range,
                "best_vehicle": best_vehicle,
                "best_vehicle_ce": best_vehicle_ce,
            }
        )

    return pd.DataFrame(rows).sort_values("avg_ce", ascending=False).reset_index(drop=True)


# ============================================================
# Performance clustering (v1)
# ============================================================
# Clustering intentionally uses only three standardized features:
#   CE Score, K/D (ground frags per death), and log1p(sample battles).
# Win rate / frags-per-battle / nation / vehicle type are context only.
CLUSTER_MIN_SAMPLE_BATTLES = 100   # default evidence floor (user-adjustable in UI)
CLUSTER_MIN_VEHICLES = 15          # below this we do not attempt clustering
CLUSTER_Z_HIGH = 0.4               # z-median above this counts as "high"
CLUSTER_Z_LOW = -0.4               # z-median below this counts as "low"


def build_vehicle_clusters(
    df: pd.DataFrame,
    min_sample_battles: int = CLUSTER_MIN_SAMPLE_BATTLES,
    min_cluster_size=None,
    min_samples=None,
):
    """Cluster vehicles with HDBSCAN on standardized
    [combat_effectiveness, ground_frags_per_death, log1p(total_battles_30d)].

    Missing-BR vehicles and rows lacking CE / K/D are excluded, as are vehicles
    below ``min_sample_battles``. Returns ``(clustered_df, meta)`` where
    clustered_df has: log1p_sample_battles, z_ce, z_kd, z_log_battles, cluster_id
    (-1 = HDBSCAN noise). ``meta`` describes counts / quality. sklearn is imported
    lazily so this module still imports if scikit-learn is absent.
    """
    meta = {
        "available": True,
        "n_vehicles": 0,
        "n_clusters": 0,
        "noise_pct": float("nan"),
        "silhouette": None,
        "quality_label": "Weak",
        "min_cluster_size": None,
        "reason": None,
    }

    if df.empty:
        return df.iloc[0:0].copy(), {**meta, "reason": "empty"}

    work = df.copy()
    has_br = (
        work["has_realistic_br"].fillna(False)
        if "has_realistic_br" in work.columns
        else work["realistic_br"].notna()
    )
    mask = (
        has_br
        & work["combat_effectiveness"].notna()
        & work["ground_frags_per_death"].notna()
        & (work["total_battles_30d"].fillna(0) >= min_sample_battles)
    )
    cand = work[mask].copy()
    n = len(cand)
    meta["n_vehicles"] = n

    if n < CLUSTER_MIN_VEHICLES:
        return cand, {**meta, "reason": "too_few"}

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import HDBSCAN
        from sklearn.metrics import silhouette_score
    except ImportError:
        return cand, {**meta, "available": False, "reason": "sklearn_missing"}

    cand["log1p_sample_battles"] = np.log1p(cand["total_battles_30d"].fillna(0))
    feats = cand[
        ["combat_effectiveness", "ground_frags_per_death", "log1p_sample_battles"]
    ].to_numpy(dtype=float)
    z = StandardScaler().fit_transform(feats)
    cand["z_ce"] = z[:, 0]
    cand["z_kd"] = z[:, 1]
    cand["z_log_battles"] = z[:, 2]

    # WT BR slices are small and standout groups may be only ~3 vehicles, so use
    # a low min_cluster_size / min_samples. (Larger values marked almost
    # everything as noise on these spread-out performance clouds.)
    if min_cluster_size is None:
        min_cluster_size = 3
    if min_samples is None:
        min_samples = 2
    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        copy=True,
    )
    labels = model.fit_predict(z)
    cand["cluster_id"] = labels

    non_noise = labels != -1
    n_clusters = int(len({c for c in labels if c != -1}))
    noise_pct = float((~non_noise).mean() * 100.0)

    silhouette = None
    if n_clusters >= 2 and int(non_noise.sum()) > n_clusters:
        try:
            silhouette = float(silhouette_score(z[non_noise], labels[non_noise]))
        except Exception:
            silhouette = None

    if n_clusters >= 2 and silhouette is not None and silhouette >= 0.5 and noise_pct < 30:
        quality = "Strong"
    elif n_clusters >= 2 and ((silhouette is not None and silhouette >= 0.25) or noise_pct < 50):
        quality = "Moderate"
    else:
        quality = "Weak"

    meta.update(
        {
            "n_clusters": n_clusters,
            "noise_pct": noise_pct,
            "silhouette": silhouette,
            "quality_label": quality,
            "min_cluster_size": int(min_cluster_size),
            "reason": None if n_clusters >= 1 else "no_clusters",
        }
    )
    return cand, meta


def _classify_cluster(z_ce: float, z_kd: float, z_sb: float) -> str:
    """Friendly archetype label from a cluster's standardized medians.

    Priority order (first match wins):
      1. Popular Strugglers  -> high sample battles + below-average CE
      2. Core Meta           -> high CE + high K/D + high sample battles
      3. Underplayed Meta    -> high CE + high K/D (not widely played)
      4. Niche Signals       -> low sample battles (uncertain signal)
      5. Off-Meta            -> low CE + low K/D
      6. Solid Picks         -> everything else (mid)
    """
    hi, lo = CLUSTER_Z_HIGH, CLUSTER_Z_LOW
    if z_sb >= hi and z_ce < 0:
        return "Popular Strugglers"
    if z_ce >= hi and z_kd >= hi and z_sb >= hi:
        return "Core Meta"
    if z_ce >= hi and z_kd >= hi:
        return "Underplayed Meta"
    if z_sb <= lo:
        return "Niche Signals"
    if z_ce <= lo and z_kd <= lo:
        return "Off-Meta"
    return "Solid Picks"


def label_clusters(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Add a friendly archetype label to each row.

    Keeps raw ``cluster_id`` as the internal key. When two or more clusters map
    to the same friendly label, a suffix (A, B, ...) is appended, ordered by
    cluster mean CE descending, so labels stay unambiguous. HDBSCAN noise
    (cluster_id == -1) is always labelled "Outliers".
    """
    out = clustered_df.copy()
    if out.empty or "cluster_id" not in out.columns:
        out["friendly_label_base"] = pd.Series(dtype="object")
        out["friendly_label"] = pd.Series(dtype="object")
        return out

    non_noise = out[out["cluster_id"] != -1]
    prof = non_noise.groupby("cluster_id")[["z_ce", "z_kd", "z_log_battles"]].median()
    mean_ce = non_noise.groupby("cluster_id")["combat_effectiveness"].mean()

    base_label = {-1: "Outliers"}
    for cid, r in prof.iterrows():
        base_label[cid] = _classify_cluster(r["z_ce"], r["z_kd"], r["z_log_battles"])

    # Disambiguate duplicate friendly labels (excluding Outliers).
    from collections import defaultdict

    label_to_cids = defaultdict(list)
    for cid, lab in base_label.items():
        if cid != -1:
            label_to_cids[lab].append(cid)

    final = {-1: "Outliers"}
    for lab, cids in label_to_cids.items():
        if len(cids) == 1:
            final[cids[0]] = lab
        else:
            ordered = sorted(cids, key=lambda c: mean_ce.get(c, float("-inf")), reverse=True)
            for i, c in enumerate(ordered):
                final[c] = f"{lab} {chr(65 + i)}"

    out["friendly_label_base"] = out["cluster_id"].map(base_label)
    out["friendly_label"] = out["cluster_id"].map(final)
    return out


# ============================================================
# Lineup Builder (v1)
# ============================================================
# Builds a same-nation lineup within a BR range, ranked mostly by average CE
# Score with an optional small role-diversity bonus. Intentionally simple: no
# hard role/BR-coverage constraints, no uptier modelling.
LINEUP_TOP_N = 25   # only combine the top-N eligible vehicles by CE (keeps search fast)
LINEUP_TOP_K = 20   # number of scored lineups returned
LINEUP_MIN_SAMPLE_BATTLES = 100


def lineup_diversity_bonus(n_types: int) -> float:
    """Small bonus for role diversity: 0 / +2.5 / +5.0 / +7.5 (capped at 4+)."""
    if n_types <= 1:
        return 0.0
    if n_types == 2:
        return 2.5
    if n_types == 3:
        return 5.0
    return 7.5


def build_lineup_candidates(
    vehicle_df: pd.DataFrame,
    nation: str,
    br_min: float,
    br_max: float,
    allowed_types=None,
    include_premium: bool = True,
    min_sample_battles: int = LINEUP_MIN_SAMPLE_BATTLES,
) -> pd.DataFrame:
    """Eligible vehicles for a lineup: one nation, within the BR range, with a
    Realistic BR and a CE Score, above the sample-battle floor, in the allowed
    types, honoring the premium setting. Sorted by CE Score descending."""
    if vehicle_df.empty:
        return vehicle_df.copy()

    out = vehicle_df.copy()
    has_br = (
        out["has_realistic_br"].fillna(False)
        if "has_realistic_br" in out.columns
        else out["realistic_br"].notna()
    )
    mask = (
        (out["country"] == nation)
        & has_br
        & out["combat_effectiveness"].notna()
        & (out["realistic_br"] >= br_min)
        & (out["realistic_br"] <= br_max)
        & (out["total_battles_30d"].fillna(0) >= min_sample_battles)
    )
    if allowed_types:
        mask &= out["vehicle_type"].isin(allowed_types)
    if not include_premium and "is_premium" in out.columns:
        mask &= out["is_premium"] == False

    cand = out[mask].copy()
    return cand.sort_values("combat_effectiveness", ascending=False).reset_index(drop=True)


def build_lineups(
    candidate_df: pd.DataFrame,
    size: int,
    prefer_diversity: bool = True,
    top_n: int = LINEUP_TOP_N,
    top_k: int = LINEUP_TOP_K,
) -> pd.DataFrame:
    """Score every size-combination from the top-N candidates and return the
    top-K lineups.

    lineup_score = avg CE Score + (diversity bonus if prefer_diversity).

    Returns one row per lineup with: member_slugs (tuple), member_names (tuple),
    lineup_score, avg_ce, avg_kd, avg_frags_per_battle, avg_win_rate,
    median_sample_battles, n_types, premium_count, br_spread, br_lo, br_hi.
    """
    import itertools

    cols = [
        "vehicle_slug",
        "vehicle_name",
        "vehicle_type",
        "realistic_br",
        "combat_effectiveness",
        "ground_frags_per_death",
        "ground_frags_per_battle",
        "win_rate",
        "total_battles_30d",
        "is_premium",
    ]
    if candidate_df.empty or len(candidate_df) < size:
        return pd.DataFrame()

    pool = candidate_df.head(top_n).reset_index(drop=True)
    pool = pool[[c for c in cols if c in pool.columns]].copy()

    slug = pool["vehicle_slug"].to_numpy()
    name = pool["vehicle_name"].to_numpy()
    vtype = pool["vehicle_type"].astype("object").to_numpy()
    br = pool["realistic_br"].to_numpy(dtype=float)
    ce = pool["combat_effectiveness"].to_numpy(dtype=float)
    kd = pool["ground_frags_per_death"].to_numpy(dtype=float)
    fpb = pool["ground_frags_per_battle"].to_numpy(dtype=float)
    wr = pool["win_rate"].to_numpy(dtype=float)
    sb = pool["total_battles_30d"].to_numpy(dtype=float)
    prem = (
        pool["is_premium"].fillna(False).to_numpy(dtype=bool)
        if "is_premium" in pool.columns
        else np.zeros(len(pool), dtype=bool)
    )

    rows = []
    for combo in itertools.combinations(range(len(pool)), size):
        idx = list(combo)
        n_types = len(set(vtype[i] for i in idx))
        avg_ce = float(np.nanmean(ce[idx]))
        bonus = lineup_diversity_bonus(n_types) if prefer_diversity else 0.0
        rows.append(
            {
                "member_slugs": tuple(slug[i] for i in idx),
                "member_names": tuple(name[i] for i in idx),
                "lineup_score": avg_ce + bonus,
                "avg_ce": avg_ce,
                "avg_kd": float(np.nanmean(kd[idx])),
                "avg_frags_per_battle": float(np.nanmean(fpb[idx])),
                "avg_win_rate": float(np.nanmean(wr[idx])),
                "median_sample_battles": float(np.nanmedian(sb[idx])),
                "n_types": n_types,
                "premium_count": int(np.sum(prem[idx])),
                "br_spread": float(np.nanmax(br[idx]) - np.nanmin(br[idx])),
                "br_lo": float(np.nanmin(br[idx])),
                "br_hi": float(np.nanmax(br[idx])),
            }
        )

    result = pd.DataFrame(rows)
    return (
        result.sort_values(["lineup_score", "avg_ce"], ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


# ============================================================
# Meta Signals (v1): Rising Performers + Underplayed Meta
# ============================================================
# Daily Performance Score is a *daily BR-relative analogue* of the Combat
# Effectiveness Score, used only for trend detection. It mirrors CE's structure
# (log1p K/D & frags per battle, within-BR robust z, same weights, 50 + 15*z)
# but is computed per day WITHOUT empirical-Bayes shrinkage (daily battle counts
# are too small for per-day shrinkage). The official 30-day CE is unchanged.
META_RELIABILITY_PRIOR = 50          # reliability = battles / (battles + 50)
MOMENTUM_MIN_OBSERVED = 20           # min observed scored days for a momentum signal
MOMENTUM_MIN_SAMPLE_BATTLES = 50     # min 30-day sample battles for a momentum signal
MOMENTUM_WINDOW = 10                 # days averaged at each end
META_VALUE_MIN_SAMPLE_BATTLES = 25   # default floor for Underplayed Meta

# Daily Performance Score uses the same core metrics/weights as CE (minus the
# sample-confidence term, which is meaningless per day).
_DAILY_SCORE_WEIGHTS = {
    "ground_frags_per_death": 0.40,
    "ground_frags_per_battle": 0.40,
    "win_rate": 0.15,
}


def build_daily_br_score(recent_df: pd.DataFrame) -> pd.DataFrame:
    """Daily BR-relative Performance Score (CE-like) per vehicle per day.

    For each date, within each exact BR, log1p-transform K/D & frags per battle,
    robust-z each metric, combine with the CE weights (renormalized over the
    metrics available that day), and map to 50 + 15*z clipped to 0-100.
    Returns columns: vehicle_slug, date, daily_score. Computed on the FULL daily
    cross-section so the within-BR distribution is stable.
    """
    if recent_df.empty:
        return pd.DataFrame(columns=["vehicle_slug", "date", "daily_score"])

    df = recent_df.dropna(subset=["realistic_br"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["vehicle_slug", "date", "daily_score"])

    metrics = [m for m in _DAILY_SCORE_WEIGHTS if m in df.columns]
    parts = []
    for _, day in df.groupby("date"):
        br = day["realistic_br"]
        num = pd.Series(0.0, index=day.index)
        den = pd.Series(0.0, index=day.index)
        for m in metrics:
            w = _DAILY_SCORE_WEIGHTS[m]
            vals = np.log1p(day[m]) if m in CE_LOG1P_METRICS else day[m]
            z = _within_br_robust_z(vals, br)
            avail = z.notna()
            num = num + (w * z).where(avail, 0.0)
            den = den + pd.Series(w, index=day.index).where(avail, 0.0)
        z_total = num / den.where(den > 0)
        out_day = day[["vehicle_slug", "date"]].copy()
        out_day["daily_score"] = (SCORE_CENTER + SCORE_SLOPE * z_total).clip(0, 100)
        parts.append(out_day)

    return pd.concat(parts, ignore_index=True)


def build_momentum(
    daily_score_df: pd.DataFrame,
    vehicle_agg_df: pd.DataFrame,
    min_observed: int = MOMENTUM_MIN_OBSERVED,
    min_sample_battles: int = MOMENTUM_MIN_SAMPLE_BATTLES,
    window: int = MOMENTUM_WINDOW,
) -> pd.DataFrame:
    """Per-vehicle Momentum Score from the daily Performance Score series.

    momentum = ce_gain * coverage_weight * reliability, where
      ce_gain     = mean(last `window` scored days) - mean(first `window`)
      coverage    = min(observed_scored_days / 30, 1)
      reliability = sample_battles / (sample_battles + META_RELIABILITY_PRIOR)

    Requires >= min_observed scored days and >= min_sample_battles. Returned for
    ALL qualifying vehicles (the app filters to the top-deck slice by slug).
    """
    if daily_score_df.empty or vehicle_agg_df.empty:
        return pd.DataFrame()

    scored = daily_score_df.dropna(subset=["daily_score"]).sort_values(["vehicle_slug", "date"])
    rows = []
    for slug, g in scored.groupby("vehicle_slug"):
        n = len(g)
        if n < min_observed:
            continue
        early = float(g["daily_score"].head(window).mean())
        late = float(g["daily_score"].tail(window).mean())
        rows.append(
            {
                "vehicle_slug": slug,
                "observed_scored_days": n,
                "early_ce": round(early, 1),
                "late_ce": round(late, 1),
                "ce_gain": round(late - early, 1),
            }
        )
    if not rows:
        return pd.DataFrame()

    mom = pd.DataFrame(rows)
    ctx_cols = [
        c for c in [
            "vehicle_slug", "vehicle_name", "country", "vehicle_type",
            "realistic_br", "combat_effectiveness", "ground_frags_per_death",
            "ground_frags_per_battle", "win_rate", "total_battles_30d",
            "has_realistic_br", "is_premium",
        ]
        if c in vehicle_agg_df.columns
    ]
    mom = mom.merge(vehicle_agg_df[ctx_cols], on="vehicle_slug", how="left")

    mom = mom[
        mom.get("has_realistic_br", True).fillna(False)
        & (mom["total_battles_30d"].fillna(0) >= min_sample_battles)
    ].copy()
    if mom.empty:
        return mom

    coverage = (mom["observed_scored_days"] / 30).clip(upper=1.0)
    reliability = mom["total_battles_30d"] / (mom["total_battles_30d"] + META_RELIABILITY_PRIOR)
    mom["momentum_score"] = (mom["ce_gain"] * coverage * reliability).round(2)
    return mom.sort_values("momentum_score", ascending=False).reset_index(drop=True)


def build_meta_value(
    vehicle_df: pd.DataFrame,
    min_sample_battles: int = META_VALUE_MIN_SAMPLE_BATTLES,
) -> pd.DataFrame:
    """Underplayed Meta: Meta Value Score on the (already filtered) 30-day
    aggregate, using percentiles within this slice.

    performance_strength = 0.50*CE_pct + 0.30*KD_pct + 0.20*fpb_pct
    underplay_strength   = 1 - sample_battles_pct
    reliability          = sample_battles / (sample_battles + 50)
    meta_value = 100 * performance_strength * (0.50 + 0.50*underplay_strength) * reliability
    """
    if vehicle_df.empty:
        return pd.DataFrame()

    has_br = (
        vehicle_df["has_realistic_br"].fillna(False)
        if "has_realistic_br" in vehicle_df.columns
        else vehicle_df["realistic_br"].notna()
    )
    d = vehicle_df[
        has_br
        & vehicle_df["combat_effectiveness"].notna()
        & vehicle_df["ground_frags_per_death"].notna()
        & vehicle_df["ground_frags_per_battle"].notna()
        & (vehicle_df["total_battles_30d"].fillna(0) >= min_sample_battles)
    ].copy()
    if d.empty:
        return d

    d["ce_pct"] = d["combat_effectiveness"].rank(pct=True)
    d["kd_pct"] = d["ground_frags_per_death"].rank(pct=True)
    d["fpb_pct"] = d["ground_frags_per_battle"].rank(pct=True)
    d["sample_battles_pct"] = d["total_battles_30d"].rank(pct=True)

    perf = 0.50 * d["ce_pct"] + 0.30 * d["kd_pct"] + 0.20 * d["fpb_pct"]
    underplay = 1.0 - d["sample_battles_pct"]
    reliability = d["total_battles_30d"] / (d["total_battles_30d"] + META_RELIABILITY_PRIOR)

    d["performance_strength"] = perf
    d["underplay_strength"] = underplay
    d["meta_value_score"] = (100.0 * perf * (0.50 + 0.50 * underplay) * reliability).round(2)
    return d.sort_values("meta_value_score", ascending=False).reset_index(drop=True)

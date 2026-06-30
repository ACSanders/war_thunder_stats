"""Pure pandas feature-engineering helpers for the War Thunder Stats app.

This module intentionally has **no Streamlit imports**. It is the single source
of truth for:

  * cleaning / typing the raw ThunderSkill CSV          -> clean_daily()
  * safe metadata fallbacks (country / type / rank)     -> apply_metadata_fallbacks()
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

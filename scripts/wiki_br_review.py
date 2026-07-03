"""Compare ThunderSkill realistic_br and premium status against official War
Thunder Wiki metadata.

Reads the processed ThunderSkill CSV, fetches each vehicle's War Thunder Wiki
page once, and extracts both its Realistic (RB) battle rating and its
premium status. Ground Realistic only -- Arcade and Simulator BR are
ignored. BR and premium extraction share a single page fetch per vehicle.

Two modes:

  Default (review mode): writes a review/audit CSV
  (data/metadata/wiki_realistic_br_mismatch_review.csv) covering every row
  checked this run -- successes, mismatches, 404s, and parse failures alike,
  for both BR and premium status. This never touches the trusted app lookup
  file.

  --write-lookup (automation mode): additionally builds a candidate update
  for the trusted app lookup file (data/metadata/wiki_ground_br_lookup.csv)
  from this run's successful, sane values; merges it with any existing
  trusted rows (preserving rows -- and individual BR/premium fields -- for
  vehicles this run didn't successfully re-check on that axis); validates
  the merged result against strict thresholds; and only then atomically
  replaces the real lookup file. If validation fails, the real lookup file
  is left untouched AND the process exits with status 1 (including under
  --dry-run, so a failing dry run is observable to a caller like CI without
  grepping stdout). --dry-run runs the full pipeline (including validation)
  without ever writing the real lookup file, so it's safe to preview what a
  run would do.

Wiki premium status is read from the page's root container class
(div.game-unit), which is one of: game-unit--premium, game-unit--regular,
or game-unit--squadron. Squadron is a separate availability axis (squadron
vehicles are not the same thing as premium vehicles), so it's explicitly
mapped to "not premium" rather than treated as an unknown/ambiguous status.
Any other combination (missing, or more than one status class) is left
unknown (wiki_is_premium=None) -- never guessed.

Usage:
    python scripts/wiki_br_review.py --slugs ussr_kv_1s ussr_t_80bvm
    python scripts/wiki_br_review.py --sample 10
    python scripts/wiki_br_review.py --limit 50
    python scripts/wiki_br_review.py
    python scripts/wiki_br_review.py --write-lookup --dry-run
    python scripts/wiki_br_review.py --write-lookup
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_DIR / "data" / "processed" / "ground_realistic_30_days_latest.csv"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "data" / "metadata" / "wiki_realistic_br_mismatch_review.csv"
DEFAULT_LOOKUP_CSV = PROJECT_DIR / "data" / "metadata" / "wiki_ground_br_lookup.csv"
CHECKPOINT_PATH = PROJECT_DIR / "data" / "checkpoints" / "wiki_br_review_checkpoint.csv"

WIKI_BASE_URL = "https://wiki.warthunder.com/unit"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WarThunderStatsBot/0.1; +https://github.com/ACSanders)"
}

BR_MISMATCH_TOLERANCE = 1e-9

# Sane-range / valid-step guards for anything written to the trusted lookup.
# War Thunder ground RB BRs run 1.0-13.7 in x.0 / x.3 / x.7 steps.
BR_MIN = 1.0
BR_MAX = 13.7
BR_VALID_LAST_DIGITS = {0, 3, 7}
BR_STEP_TOLERANCE = 0.02

# Wiki root-container status classes that map to a known premium signal.
# game-unit--squadron is a separate availability axis, not premium.
PREMIUM_STATUS_CLASS_MAP = {
    "game-unit--premium": True,
    "game-unit--regular": False,
    "game-unit--squadron": False,
}

LOOKUP_NOTE = "Official War Thunder Wiki RB BR"
LOOKUP_COLUMNS = [
    "vehicle_slug",
    "vehicle_name",
    "wiki_arcade_br",
    "wiki_realistic_br",
    "wiki_simulator_br",
    "wiki_is_premium",
    "wiki_url",
    "checked_at",
    "notes",
]

# Metadata carried into the review row (first non-null value per vehicle_slug).
METADATA_COLS = ["vehicle_name", "country", "vehicle_type", "realistic_br", "vehicle_url", "is_premium"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_valid(series: pd.Series):
    """First non-null value in a group (NaN if the group is all-null).

    Deliberately local rather than importing features._first_valid -- this
    script has no dependency on features.py.
    """
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


def build_unique_vehicles(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle_slug using first-non-null metadata.

    Plain drop_duplicates() can keep a row whose metadata happens to be blank
    for that particular date; this instead pulls the first non-null value per
    column across all of that vehicle's rows.
    """
    if raw_df.empty or "vehicle_slug" not in raw_df.columns:
        return pd.DataFrame(columns=["vehicle_slug", *METADATA_COLS])

    cols = [c for c in METADATA_COLS if c in raw_df.columns]
    grouped = raw_df.groupby("vehicle_slug", dropna=False)[cols].agg(first_valid)
    return grouped.reset_index()


def fetch_wiki_page(vehicle_slug: str, timeout: float = 20.0):
    """Fetch a vehicle's Wiki page once, shared by both the BR and premium
    extractors so a vehicle only costs a single HTTP request per run.

    Returns (soup, wiki_url, early_result). ``soup`` is None and
    ``early_result`` is a dict with keys (status, error) when the page
    itself couldn't be fetched or parsed into HTML -- in that case both BR
    and premium extraction share that same failure status for this vehicle.
    On success ``soup`` is a BeautifulSoup object and ``early_result`` is
    None.
    """
    wiki_url = f"{WIKI_BASE_URL}/{vehicle_slug}"

    try:
        response = requests.get(wiki_url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        return None, wiki_url, {"status": "request_error", "error": str(exc)}

    if response.status_code == 404:
        return None, wiki_url, {"status": "http_404", "error": None}

    if response.status_code != 200:
        return None, wiki_url, {"status": "http_error", "error": f"HTTP {response.status_code}"}

    try:
        soup = BeautifulSoup(response.text, "lxml")
    except Exception as exc:
        return None, wiki_url, {"status": "parse_error", "error": str(exc)}

    return soup, wiki_url, None


def extract_rb_br(soup: BeautifulSoup, wiki_url: str) -> dict:
    """Extract Realistic (RB) battle rating from an already-parsed Wiki page.

    Returns a dict with keys: wiki_realistic_br, wiki_url, status, error.
    status is one of: ok, no_br_block, rb_mode_not_found, parse_error.
    """
    try:
        br_block = soup.select_one(".game-unit_br")

        if br_block is None:
            return {"wiki_realistic_br": None, "wiki_url": wiki_url, "status": "no_br_block", "error": None}

        rb_value_text = None
        for item in br_block.select(".game-unit_br-item"):
            mode_el = item.select_one(".mode")
            value_el = item.select_one(".value")
            if mode_el and value_el and mode_el.get_text(strip=True) == "RB":
                rb_value_text = value_el.get_text(strip=True)
                break

        if rb_value_text is None:
            return {"wiki_realistic_br": None, "wiki_url": wiki_url, "status": "rb_mode_not_found", "error": None}

        wiki_br = pd.to_numeric(rb_value_text, errors="coerce")
        if pd.isna(wiki_br):
            return {
                "wiki_realistic_br": None,
                "wiki_url": wiki_url,
                "status": "parse_error",
                "error": f"Could not parse RB value: {rb_value_text!r}",
            }

        return {"wiki_realistic_br": float(wiki_br), "wiki_url": wiki_url, "status": "ok", "error": None}

    except Exception as exc:
        return {"wiki_realistic_br": None, "wiki_url": wiki_url, "status": "parse_error", "error": str(exc)}


def extract_premium_status(soup: BeautifulSoup, wiki_url: str) -> dict:
    """Extract Wiki premium status from an already-parsed Wiki page.

    Returns a dict with keys: wiki_is_premium, premium_status, premium_error.
    premium_status is one of: ok, no_root_found, ambiguous_status_class,
    parse_error. Ambiguous/unknown always yields wiki_is_premium=None --
    never guessed.
    """
    try:
        root = soup.select_one("div.game-unit")

        if root is None:
            return {"wiki_is_premium": None, "premium_status": "no_root_found", "premium_error": None}

        classes = root.get("class") or []
        matched = [known for known in PREMIUM_STATUS_CLASS_MAP if known in classes]

        if len(matched) != 1:
            return {
                "wiki_is_premium": None,
                "premium_status": "ambiguous_status_class",
                "premium_error": f"classes={classes}",
            }

        return {
            "wiki_is_premium": PREMIUM_STATUS_CLASS_MAP[matched[0]],
            "premium_status": "ok",
            "premium_error": None,
        }

    except Exception as exc:
        return {"wiki_is_premium": None, "premium_status": "parse_error", "premium_error": str(exc)}


def build_review_row(vehicle_row: pd.Series) -> dict:
    """Fetch + compare one vehicle's BR and premium status. Never fabricates
    a Wiki value -- a failed fetch/parse leaves the corresponding wiki_*
    field null, its mismatch flag False, and the ThunderSkill value
    untouched. BR and premium extraction share a single page fetch."""
    vehicle_slug = vehicle_row["vehicle_slug"]
    checked_at = utc_now_iso()

    ts_br = pd.to_numeric(vehicle_row.get("realistic_br"), errors="coerce")
    ts_br = float(ts_br) if pd.notna(ts_br) else None

    ts_premium = vehicle_row.get("is_premium")
    ts_premium = bool(ts_premium) if pd.notna(ts_premium) else None

    soup, wiki_url, early_result = fetch_wiki_page(vehicle_slug)

    if early_result is not None:
        br_result = {"wiki_realistic_br": None, "wiki_url": wiki_url, **early_result}
        premium_result = {
            "wiki_is_premium": None,
            "premium_status": early_result["status"],
            "premium_error": early_result["error"],
        }
    else:
        br_result = extract_rb_br(soup, wiki_url)
        premium_result = extract_premium_status(soup, wiki_url)

    wiki_br = br_result["wiki_realistic_br"]
    br_delta = None
    mismatch = False
    if ts_br is not None and wiki_br is not None:
        br_delta = wiki_br - ts_br
        mismatch = abs(br_delta) > BR_MISMATCH_TOLERANCE

    wiki_premium = premium_result["wiki_is_premium"]
    premium_mismatch = (
        ts_premium is not None and wiki_premium is not None and ts_premium != wiki_premium
    )

    return {
        "vehicle_slug": vehicle_slug,
        "vehicle_name": vehicle_row.get("vehicle_name"),
        "country": vehicle_row.get("country"),
        "vehicle_type": vehicle_row.get("vehicle_type"),
        "thunderskill_realistic_br": ts_br,
        "wiki_realistic_br": wiki_br,
        "br_delta": br_delta,
        "mismatch": mismatch,
        "thunderskill_is_premium": ts_premium,
        "wiki_is_premium": wiki_premium,
        "premium_mismatch": premium_mismatch,
        "wiki_url": wiki_url,
        "status": br_result["status"],
        "error": br_result["error"],
        "premium_status": premium_result["premium_status"],
        "premium_error": premium_result["premium_error"],
        "checked_at": checked_at,
    }


def save_checkpoint(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Checkpoint saved: {path} ({len(rows)} rows)")


# ------------------------------------------------------------
# Automation mode: review rows -> validated trusted lookup update
# ------------------------------------------------------------

def is_sane_br(value) -> bool:
    if value is None or pd.isna(value):
        return False
    value = float(value)
    return BR_MIN <= value <= BR_MAX


def is_valid_br_step(value) -> bool:
    """True if value is within tolerance of an x.0 / x.3 / x.7 BR step."""
    if value is None or pd.isna(value):
        return False
    scaled = round(float(value) * 10)
    if abs(float(value) * 10 - scaled) > BR_STEP_TOLERANCE * 10:
        return False
    return (scaled % 10) in BR_VALID_LAST_DIGITS


def eligible_for_lookup(review_df: pd.DataFrame) -> pd.DataFrame:
    """Rows from this run that are safe to feed into the trusted lookup's BR
    columns: status == 'ok', numeric, in-range, on a valid BR step."""
    if review_df.empty:
        return review_df

    mask = (
        (review_df["status"] == "ok")
        & review_df["wiki_realistic_br"].notna()
        & review_df["wiki_realistic_br"].apply(is_sane_br)
        & review_df["wiki_realistic_br"].apply(is_valid_br_step)
    )
    return review_df[mask].copy()


def eligible_for_premium_lookup(review_df: pd.DataFrame) -> pd.DataFrame:
    """Rows from this run that are safe to feed into the trusted lookup's
    premium column: premium_status == 'ok', i.e. a known True/False signal."""
    if review_df.empty:
        return review_df

    mask = (review_df["premium_status"] == "ok") & review_df["wiki_is_premium"].notna()
    return review_df[mask].copy()


def load_existing_lookup(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LOOKUP_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=LOOKUP_COLUMNS)
    for col in LOOKUP_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[LOOKUP_COLUMNS]


def build_lookup_candidate(review_df: pd.DataFrame, existing_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Merge this run's eligible BR and premium values on top of the
    existing trusted lookup, upserted independently per axis.

    A vehicle that only refreshed successfully on one axis this run (BR or
    premium) keeps its previously trusted value on the other axis -- neither
    axis is ever cleared just because the other one didn't refresh this run.
    Vehicles untouched this run keep their existing trusted row entirely.
    """
    base = existing_lookup_df.copy()
    for col in LOOKUP_COLUMNS:
        if col not in base.columns:
            base[col] = pd.NA
    base = base[LOOKUP_COLUMNS].drop_duplicates(subset="vehicle_slug", keep="last")
    base = base.set_index("vehicle_slug")

    br_eligible = eligible_for_lookup(review_df)
    premium_eligible = eligible_for_premium_lookup(review_df)

    if not br_eligible.empty:
        br_updates = br_eligible.set_index("vehicle_slug")[
            ["vehicle_name", "wiki_realistic_br", "wiki_url", "checked_at"]
        ].copy()
        br_updates["notes"] = LOOKUP_NOTE
    else:
        br_updates = pd.DataFrame(columns=["vehicle_name", "wiki_realistic_br", "wiki_url", "checked_at", "notes"])

    if not premium_eligible.empty:
        premium_updates = premium_eligible.set_index("vehicle_slug")[
            ["vehicle_name", "wiki_is_premium", "wiki_url", "checked_at"]
        ].copy()
    else:
        premium_updates = pd.DataFrame(columns=["vehicle_name", "wiki_is_premium", "wiki_url", "checked_at"])

    # A brand-new vehicle_slug only enters the trusted lookup once it has a
    # valid BR -- premium-only data is never enough to create a new row by
    # itself (DataFrame.update() below silently skips premium rows whose
    # vehicle_slug isn't already present in base, rather than adding them).
    # This keeps every candidate row's wiki_realistic_br guaranteed non-null
    # without loosening that validation invariant.
    base = base.reindex(base.index.union(br_updates.index))

    if not br_updates.empty:
        base.update(br_updates)
    if not premium_updates.empty:
        base.update(premium_updates)

    # Ground Realistic only -- always blank, regardless of source.
    base["wiki_arcade_br"] = pd.NA
    base["wiki_simulator_br"] = pd.NA

    out = base.reset_index().rename(columns={"index": "vehicle_slug"})
    out = out.sort_values("vehicle_slug").reset_index(drop=True)
    return out[LOOKUP_COLUMNS]


def validate_lookup_candidate(
    candidate_df: pd.DataFrame,
    attempted_count: int,
    ok_count: int,
    valid_count: int,
    premium_ok_count: int,
    min_ok_rate: float,
    min_valid_wiki_rate: float,
    min_premium_ok_rate: float,
    min_vehicles: int,
) -> list:
    """Returns a list of failure reasons; empty list means validation passed."""
    failures = []

    if attempted_count < min_vehicles:
        failures.append(
            f"attempted_count={attempted_count} below --min-vehicles={min_vehicles}"
        )

    ok_rate = (ok_count / attempted_count) if attempted_count else 0.0
    if ok_rate < min_ok_rate:
        failures.append(f"ok_rate={ok_rate:.3f} below --min-ok-rate={min_ok_rate}")

    valid_rate = (valid_count / attempted_count) if attempted_count else 0.0
    if valid_rate < min_valid_wiki_rate:
        failures.append(
            f"valid_wiki_rate={valid_rate:.3f} below --min-valid-wiki-rate={min_valid_wiki_rate}"
        )

    premium_ok_rate = (premium_ok_count / attempted_count) if attempted_count else 0.0
    if premium_ok_rate < min_premium_ok_rate:
        failures.append(
            f"premium_ok_rate={premium_ok_rate:.3f} below --min-premium-ok-rate={min_premium_ok_rate}"
        )

    if candidate_df.empty:
        failures.append("candidate lookup is empty")
        return failures

    if candidate_df["vehicle_slug"].duplicated().any():
        dupes = candidate_df.loc[candidate_df["vehicle_slug"].duplicated(), "vehicle_slug"].tolist()
        failures.append(f"duplicate vehicle_slug rows in candidate: {dupes}")

    if candidate_df["wiki_realistic_br"].isna().any():
        failures.append("candidate contains null wiki_realistic_br rows")
    else:
        if not candidate_df["wiki_realistic_br"].apply(is_sane_br).all():
            failures.append(f"candidate contains wiki_realistic_br outside [{BR_MIN}, {BR_MAX}]")
        if not candidate_df["wiki_realistic_br"].apply(is_valid_br_step).all():
            failures.append("candidate contains wiki_realistic_br not on a valid x.0/x.3/x.7 step")

    if candidate_df["wiki_arcade_br"].notna().any() or candidate_df["wiki_simulator_br"].notna().any():
        failures.append("candidate has non-blank wiki_arcade_br/wiki_simulator_br (Ground Realistic only)")

    if "wiki_is_premium" in candidate_df.columns:
        premium_values = candidate_df["wiki_is_premium"].dropna()
        if not premium_values.empty and not premium_values.isin([True, False]).all():
            failures.append("candidate contains wiki_is_premium values that aren't boolean True/False")

    return failures


def run_write_lookup(
    review_df: pd.DataFrame,
    attempted_count: int,
    lookup_output: Path,
    min_ok_rate: float,
    min_valid_wiki_rate: float,
    min_premium_ok_rate: float,
    min_vehicles: int,
    dry_run: bool,
) -> None:
    print()
    print("=" * 60)
    print("Automation mode: evaluating trusted lookup update")
    print("=" * 60)

    ok_count = int((review_df["status"] == "ok").sum())
    eligible_df = eligible_for_lookup(review_df)
    valid_count = len(eligible_df)

    premium_ok_count = int((review_df["premium_status"] == "ok").sum())
    premium_eligible_df = eligible_for_premium_lookup(review_df)
    premium_valid_count = len(premium_eligible_df)

    print(f"Attempted this run: {attempted_count}")
    print(f"BR status == ok: {ok_count}")
    print(f"BR eligible for lookup (ok + sane + valid step): {valid_count}")
    print(f"Premium status == ok: {premium_ok_count}")
    print(f"Premium eligible for lookup: {premium_valid_count}")

    existing_lookup_df = load_existing_lookup(lookup_output)
    candidate_df = build_lookup_candidate(review_df, existing_lookup_df)

    failures = validate_lookup_candidate(
        candidate_df=candidate_df,
        attempted_count=attempted_count,
        ok_count=ok_count,
        valid_count=valid_count,
        premium_ok_count=premium_ok_count,
        min_ok_rate=min_ok_rate,
        min_valid_wiki_rate=min_valid_wiki_rate,
        min_premium_ok_rate=min_premium_ok_rate,
        min_vehicles=min_vehicles,
    )

    if failures:
        print("VALIDATION FAILED -- real lookup file NOT modified:")
        for reason in failures:
            print(f"  - {reason}")
        sys.exit(1)

    print("Validation passed.")
    print(f"Candidate lookup rows (existing + refreshed): {len(candidate_df)}")

    temp_path = lookup_output.parent / f"{lookup_output.stem}.tmp{lookup_output.suffix}"
    lookup_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(temp_path, index=False)
    print(f"Temporary candidate lookup written: {temp_path}")

    if dry_run:
        print("--dry-run set: real lookup file NOT modified.")
        os.remove(temp_path)
        return

    os.replace(temp_path, lookup_output)
    print(f"Trusted lookup updated: {lookup_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review ThunderSkill realistic_br and premium status against War Thunder Wiki metadata."
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N vehicles instead of the full list.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N vehicles (in existing order).",
    )
    parser.add_argument(
        "--slugs", nargs="+", default=None,
        help="Only check these specific vehicle_slug values.",
    )
    parser.add_argument("--sleep-min", type=float, default=1.0, help="Minimum seconds between requests.")
    parser.add_argument("--sleep-max", type=float, default=2.0, help="Maximum seconds between requests.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output review CSV path.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save a checkpoint every N vehicles.")

    parser.add_argument(
        "--write-lookup", action="store_true",
        help="After the review run, attempt a validated update of the trusted app lookup file.",
    )
    parser.add_argument(
        "--lookup-output", type=Path, default=DEFAULT_LOOKUP_CSV,
        help="Trusted app lookup CSV to update in --write-lookup mode.",
    )
    parser.add_argument(
        "--min-ok-rate", type=float, default=0.98,
        help="Minimum fraction of attempted vehicles with BR status == ok, required to write the lookup.",
    )
    parser.add_argument(
        "--min-valid-wiki-rate", type=float, default=0.98,
        help="Minimum fraction of attempted vehicles with a sane, valid-step Wiki BR, required to write the lookup.",
    )
    parser.add_argument(
        "--min-premium-ok-rate", type=float, default=0.98,
        help="Minimum fraction of attempted vehicles with a known Wiki premium status, required to write the lookup.",
    )
    parser.add_argument(
        "--min-vehicles", type=int, default=100,
        help="Minimum number of attempted vehicles this run, required to write the lookup.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the full --write-lookup pipeline (including validation) without touching the real lookup file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = pd.read_csv(INPUT_CSV)
    vehicles_df = build_unique_vehicles(raw_df)

    if args.slugs:
        vehicles_df = vehicles_df[vehicles_df["vehicle_slug"].isin(args.slugs)].copy()
    elif args.sample:
        vehicles_df = vehicles_df.sample(n=min(args.sample, len(vehicles_df)))
    elif args.limit:
        vehicles_df = vehicles_df.head(args.limit)

    vehicles_df = vehicles_df.reset_index(drop=True)

    print(f"Vehicles to check: {len(vehicles_df)}")

    rows = []

    for i, (_, vehicle_row) in enumerate(vehicles_df.iterrows()):
        slug = vehicle_row["vehicle_slug"]
        print(f"[{i + 1}/{len(vehicles_df)}] {slug}")

        row = build_review_row(vehicle_row)
        rows.append(row)

        if (i + 1) % args.checkpoint_every == 0:
            save_checkpoint(rows, CHECKPOINT_PATH)

        if i < len(vehicles_df) - 1:
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    review_df = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(args.output, index=False)

    print(f"Saved review CSV: {args.output}")
    print(f"Rows: {len(review_df)}")
    if not review_df.empty:
        print("BR status counts:")
        print(review_df["status"].value_counts().to_string())
        print(f"BR mismatches: {int(review_df['mismatch'].sum())}")
        print()
        print("Premium status counts:")
        print(review_df["premium_status"].value_counts().to_string())
        print(f"Premium mismatches: {int(review_df['premium_mismatch'].sum())}")

    if args.write_lookup:
        run_write_lookup(
            review_df=review_df,
            attempted_count=len(vehicles_df),
            lookup_output=args.lookup_output,
            min_ok_rate=args.min_ok_rate,
            min_valid_wiki_rate=args.min_valid_wiki_rate,
            min_premium_ok_rate=args.min_premium_ok_rate,
            min_vehicles=args.min_vehicles,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

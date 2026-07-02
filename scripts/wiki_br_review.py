"""Compare ThunderSkill realistic_br against official War Thunder Wiki RB BR.

Reads the processed ThunderSkill CSV, fetches each vehicle's War Thunder Wiki
page, and extracts its Realistic (RB) battle rating. Ground Realistic only --
Arcade and Simulator BR are ignored.

Two modes:

  Default (review mode): writes a review/audit CSV
  (data/metadata/wiki_realistic_br_mismatch_review.csv) covering every row
  checked this run -- successes, mismatches, 404s, and parse failures alike.
  This never touches the trusted app lookup file.

  --write-lookup (automation mode): additionally builds a candidate update
  for the trusted app lookup file (data/metadata/wiki_ground_br_lookup.csv)
  from this run's successful, sane, in-range RB values; merges it with any
  existing trusted rows (preserving rows for vehicles this run didn't
  successfully re-check); validates the merged result against strict
  thresholds; and only then atomically replaces the real lookup file. If
  validation fails, the real lookup file is left untouched AND the process
  exits with status 1 (including under --dry-run, so a failing dry run is
  observable to a caller like CI without grepping stdout). --dry-run runs
  the full pipeline (including validation) without ever writing the real
  lookup file, so it's safe to preview what a run would do.

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

LOOKUP_NOTE = "Official War Thunder Wiki RB BR"
LOOKUP_COLUMNS = [
    "vehicle_slug",
    "vehicle_name",
    "wiki_arcade_br",
    "wiki_realistic_br",
    "wiki_simulator_br",
    "wiki_url",
    "checked_at",
    "notes",
]

# Metadata carried into the review row (first non-null value per vehicle_slug).
METADATA_COLS = ["vehicle_name", "country", "vehicle_type", "realistic_br", "vehicle_url"]


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


def fetch_wiki_rb_br(vehicle_slug: str, timeout: float = 20.0) -> dict:
    """Fetch a vehicle's Wiki page and extract its Realistic (RB) battle rating.

    Returns a dict with keys: wiki_realistic_br, wiki_url, status, error.
    status is one of: ok, http_404, http_error, request_error, no_br_block,
    rb_mode_not_found, parse_error.
    """
    wiki_url = f"{WIKI_BASE_URL}/{vehicle_slug}"

    try:
        response = requests.get(wiki_url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        return {"wiki_realistic_br": None, "wiki_url": wiki_url, "status": "request_error", "error": str(exc)}

    if response.status_code == 404:
        return {"wiki_realistic_br": None, "wiki_url": wiki_url, "status": "http_404", "error": None}

    if response.status_code != 200:
        return {
            "wiki_realistic_br": None,
            "wiki_url": wiki_url,
            "status": "http_error",
            "error": f"HTTP {response.status_code}",
        }

    try:
        soup = BeautifulSoup(response.text, "lxml")
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


def build_review_row(vehicle_row: pd.Series) -> dict:
    """Fetch + compare one vehicle. Never fabricates a Wiki value -- a failed
    fetch/parse leaves wiki_realistic_br null, mismatch False, and the
    ThunderSkill value untouched."""
    vehicle_slug = vehicle_row["vehicle_slug"]

    ts_br = pd.to_numeric(vehicle_row.get("realistic_br"), errors="coerce")
    ts_br = float(ts_br) if pd.notna(ts_br) else None

    result = fetch_wiki_rb_br(vehicle_slug)
    wiki_br = result["wiki_realistic_br"]

    br_delta = None
    mismatch = False
    if ts_br is not None and wiki_br is not None:
        br_delta = wiki_br - ts_br
        mismatch = abs(br_delta) > BR_MISMATCH_TOLERANCE

    return {
        "vehicle_slug": vehicle_slug,
        "vehicle_name": vehicle_row.get("vehicle_name"),
        "country": vehicle_row.get("country"),
        "vehicle_type": vehicle_row.get("vehicle_type"),
        "thunderskill_realistic_br": ts_br,
        "wiki_realistic_br": wiki_br,
        "br_delta": br_delta,
        "mismatch": mismatch,
        "wiki_url": result["wiki_url"],
        "status": result["status"],
        "error": result["error"],
        "checked_at": utc_now_iso(),
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
    """Rows from this run that are safe to feed into the trusted lookup:
    status == 'ok', numeric, in-range, on a valid BR step."""
    if review_df.empty:
        return review_df

    mask = (
        (review_df["status"] == "ok")
        & review_df["wiki_realistic_br"].notna()
        & review_df["wiki_realistic_br"].apply(is_sane_br)
        & review_df["wiki_realistic_br"].apply(is_valid_br_step)
    )
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


def build_lookup_candidate(eligible_df: pd.DataFrame, existing_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Merge this run's eligible rows on top of the existing trusted lookup.

    Vehicles this run didn't successfully re-check keep their existing
    trusted row untouched (preserved, not deleted). Vehicles this run did
    successfully re-check get refreshed with the new Wiki value.
    """
    new_rows = pd.DataFrame({
        "vehicle_slug": eligible_df["vehicle_slug"],
        "vehicle_name": eligible_df["vehicle_name"],
        "wiki_arcade_br": pd.NA,
        "wiki_realistic_br": eligible_df["wiki_realistic_br"],
        "wiki_simulator_br": pd.NA,
        "wiki_url": eligible_df["wiki_url"],
        "checked_at": eligible_df["checked_at"],
        "notes": LOOKUP_NOTE,
    })[LOOKUP_COLUMNS] if not eligible_df.empty else pd.DataFrame(columns=LOOKUP_COLUMNS)

    combined = pd.concat([existing_lookup_df, new_rows], ignore_index=True)

    # New rows are appended last, so keep="last" refreshes any vehicle this
    # run re-checked successfully while preserving untouched vehicles.
    combined = combined.drop_duplicates(subset="vehicle_slug", keep="last")
    combined = combined.sort_values("vehicle_slug").reset_index(drop=True)

    return combined


def validate_lookup_candidate(
    candidate_df: pd.DataFrame,
    attempted_count: int,
    ok_count: int,
    valid_count: int,
    min_ok_rate: float,
    min_valid_wiki_rate: float,
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

    return failures


def run_write_lookup(
    review_df: pd.DataFrame,
    attempted_count: int,
    lookup_output: Path,
    min_ok_rate: float,
    min_valid_wiki_rate: float,
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

    print(f"Attempted this run: {attempted_count}")
    print(f"status == ok: {ok_count}")
    print(f"Eligible for lookup (ok + sane + valid step): {valid_count}")

    existing_lookup_df = load_existing_lookup(lookup_output)
    candidate_df = build_lookup_candidate(eligible_df, existing_lookup_df)

    failures = validate_lookup_candidate(
        candidate_df=candidate_df,
        attempted_count=attempted_count,
        ok_count=ok_count,
        valid_count=valid_count,
        min_ok_rate=min_ok_rate,
        min_valid_wiki_rate=min_valid_wiki_rate,
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
        description="Review ThunderSkill realistic_br against War Thunder Wiki RB BR."
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
        help="Minimum fraction of attempted vehicles with status == ok, required to write the lookup.",
    )
    parser.add_argument(
        "--min-valid-wiki-rate", type=float, default=0.98,
        help="Minimum fraction of attempted vehicles with a sane, valid-step Wiki BR, required to write the lookup.",
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
        print("Status counts:")
        print(review_df["status"].value_counts().to_string())
        print(f"Mismatches: {int(review_df['mismatch'].sum())}")

    if args.write_lookup:
        run_write_lookup(
            review_df=review_df,
            attempted_count=len(vehicles_df),
            lookup_output=args.lookup_output,
            min_ok_rate=args.min_ok_rate,
            min_valid_wiki_rate=args.min_valid_wiki_rate,
            min_vehicles=args.min_vehicles,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

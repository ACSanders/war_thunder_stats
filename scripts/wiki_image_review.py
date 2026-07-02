"""Collect War Thunder Wiki vehicle image URLs for ground vehicles.

Reads the processed ThunderSkill CSV, fetches each vehicle's War Thunder Wiki
page, and extracts the main vehicle render image URL. Stores URLs only --
never downloads or saves image files.

This is a review/audit script only: it writes
data/metadata/wiki_vehicle_images.csv and does not touch any app source
files, the Wiki BR lookup/review files, or the Wiki BR GitHub Action.

Usage:
    python scripts/wiki_image_review.py --slugs ussr_kv_1s ussr_t_80bvm
    python scripts/wiki_image_review.py --sample 10
    python scripts/wiki_image_review.py --limit 50
    python scripts/wiki_image_review.py
"""

import argparse
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_DIR / "data" / "processed" / "ground_realistic_30_days_latest.csv"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "data" / "metadata" / "wiki_vehicle_images.csv"
CHECKPOINT_PATH = PROJECT_DIR / "data" / "checkpoints" / "wiki_image_review_checkpoint.csv"

WIKI_BASE_URL = "https://wiki.warthunder.com/unit"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WarThunderStatsBot/0.1; +https://github.com/ACSanders)"
}

# Metadata carried into the review row (first non-null value per vehicle_slug).
METADATA_COLS = ["vehicle_name"]


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
    """One row per vehicle_slug using first-non-null vehicle_name.

    Plain drop_duplicates() can keep a row whose vehicle_name happens to be
    blank for that particular date; this instead pulls the first non-null
    value per column across all of that vehicle's rows.
    """
    if raw_df.empty or "vehicle_slug" not in raw_df.columns:
        return pd.DataFrame(columns=["vehicle_slug", *METADATA_COLS])

    cols = [c for c in METADATA_COLS if c in raw_df.columns]
    grouped = raw_df.groupby("vehicle_slug", dropna=False)[cols].agg(first_valid)
    return grouped.reset_index()


def fetch_wiki_image(vehicle_slug: str, timeout: float = 20.0) -> dict:
    """Fetch a vehicle's Wiki page and extract its main vehicle image URL.

    Returns a dict with keys: wiki_image_url, wiki_url, status, error.
    status is one of: ok, http_404, http_error, request_error,
    no_image_found, invalid_image_url, parse_error.
    """
    wiki_url = f"{WIKI_BASE_URL}/{vehicle_slug}"

    try:
        response = requests.get(wiki_url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        return {"wiki_image_url": None, "wiki_url": wiki_url, "status": "request_error", "error": str(exc)}

    if response.status_code == 404:
        return {"wiki_image_url": None, "wiki_url": wiki_url, "status": "http_404", "error": None}

    if response.status_code != 200:
        return {
            "wiki_image_url": None,
            "wiki_url": wiki_url,
            "status": "http_error",
            "error": f"HTTP {response.status_code}",
        }

    try:
        soup = BeautifulSoup(response.text, "lxml")
        img = soup.select_one("img.game-unit_template-image")

        if img is None:
            return {"wiki_image_url": None, "wiki_url": wiki_url, "status": "no_image_found", "error": None}

        src = img.get("src")

        if not src or not str(src).startswith("http"):
            return {
                "wiki_image_url": None,
                "wiki_url": wiki_url,
                "status": "invalid_image_url",
                "error": f"Unexpected src value: {src!r}",
            }

        return {"wiki_image_url": str(src), "wiki_url": wiki_url, "status": "ok", "error": None}

    except Exception as exc:
        return {"wiki_image_url": None, "wiki_url": wiki_url, "status": "parse_error", "error": str(exc)}


def build_review_row(vehicle_row: pd.Series) -> dict:
    """Fetch + record one vehicle's Wiki image URL."""
    vehicle_slug = vehicle_row["vehicle_slug"]

    result = fetch_wiki_image(vehicle_slug)

    return {
        "vehicle_slug": vehicle_slug,
        "vehicle_name": vehicle_row.get("vehicle_name"),
        "wiki_image_url": result["wiki_image_url"],
        "wiki_url": result["wiki_url"],
        "checked_at": utc_now_iso(),
        "status": result["status"],
        "error": result["error"],
    }


def save_checkpoint(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Checkpoint saved: {path} ({len(rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect War Thunder Wiki vehicle image URLs for ground vehicles."
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save a checkpoint every N vehicles.")
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

    print(f"Saved image URL CSV: {args.output}")
    print(f"Rows: {len(review_df)}")
    if not review_df.empty:
        print("Status counts:")
        print(review_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()

import json
import html as html_lib
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://thunderskill.com"
LOAD_MORE_URL = f"{BASE_URL}/en/vehicles/load-more"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WarThunderStatsBot/0.1; +https://github.com/ACSanders)"
}

PROJECT_DIR = Path(__file__).resolve().parent

RAW_INDEX_DIR = PROJECT_DIR / "data" / "raw" / "vehicle_index"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SNAPSHOT_DIR = PROCESSED_DIR / "snapshots"
CHECKPOINT_DIR = PROJECT_DIR / "data" / "checkpoints"
LOGS_DIR = PROJECT_DIR / "logs"

for directory in [RAW_INDEX_DIR, PROCESSED_DIR, SNAPSHOT_DIR, CHECKPOINT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def extract_between(text: str, start_label: str, end_label: str):
    pattern = rf"{re.escape(start_label)}\s*\|\s*(.*?)\s*\|\s*{re.escape(end_label)}"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def parse_vehicle_entries_from_json(payload: dict, snapshot_date: str, pulled_at: str) -> pd.DataFrame:
    entries = payload.get("entries", [])
    rows = []

    for entry in entries:
        vehicle_url = entry.get("vehicleUrl")
        if vehicle_url:
            vehicle_url = vehicle_url if vehicle_url.startswith("http") else BASE_URL + vehicle_url

        vehicle_slug = entry.get("objectCode")
        if not vehicle_slug and vehicle_url:
            vehicle_slug = vehicle_url.rstrip("/").split("/")[-1]

        rows.append({
            "snapshot_date": snapshot_date,
            "vehicle_id": entry.get("vehicleId"),
            "vehicle_slug": vehicle_slug,
            "vehicle_name": entry.get("vehicleName"),
            "vehicle_url": vehicle_url,
            "index_type": entry.get("typeLabel"),
            "index_type_code": entry.get("objectType"),
            "index_role": entry.get("roleName"),
            "index_role_code": entry.get("roleCode"),
            "index_country": entry.get("country"),
            "index_mode": entry.get("mode"),
            "index_rank": entry.get("rankValue"),
            "index_battles": entry.get("battleCount"),
            "index_win_rate": entry.get("winrate"),
            "index_efficiency": entry.get("efficiency"),
            "search": entry.get("search"),
            "pic": entry.get("pic"),
            "pulled_at": pulled_at,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["index_country"] = (
        df["index_country"]
        .astype(str)
        .str.replace("country_", "", regex=False)
        .str.upper()
        .replace("NAN", pd.NA)
    )

    numeric_cols = [
        "vehicle_id",
        "index_type_code",
        "index_rank",
        "index_battles",
        "index_win_rate",
        "index_efficiency",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def pull_ground_realistic_vehicle_index(limit: int = 100, max_pages: int = 100) -> pd.DataFrame:
    snapshot_date = utc_today()
    all_pages = []
    offset = 0

    print("Pulling full Realistic ground vehicle index...")

    for page in range(max_pages):
        params = {
            "mode": "R",
            "offset": offset,
            "limit": limit,
            "layout": "table",
            "type": "2",
        }

        print(f"Index page {page + 1}: offset={offset}")

        response = requests.get(
            LOAD_MORE_URL,
            headers=HEADERS,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()

        page_df = parse_vehicle_entries_from_json(
            payload=payload,
            snapshot_date=snapshot_date,
            pulled_at=utc_now_iso(),
        )

        print(f"Rows returned: {len(page_df)}")

        if page_df.empty:
            break

        all_pages.append(page_df)

        if len(page_df) < limit:
            break

        offset += len(page_df)
        time.sleep(random.uniform(0.75, 1.5))

    if not all_pages:
        return pd.DataFrame()

    index_df = (
        pd.concat(all_pages, ignore_index=True)
        .drop_duplicates(subset=["vehicle_slug"])
        .reset_index(drop=True)
    )

    return index_df


def parse_vehicle_info_from_page(
    vehicle_soup: BeautifulSoup,
    vehicle_slug: str,
    vehicle_url: str,
    pulled_at: str,
) -> dict:
    page_text = vehicle_soup.get_text(" | ", strip=True)

    country = extract_between(page_text, "Country", "Vehicle type")
    vehicle_type = extract_between(page_text, "Vehicle type", "Rank")
    rank = extract_between(page_text, "Rank", "Battle rating")

    arcade_br = extract_between(page_text, "Arcade mode", "Realistic mode")
    realistic_br = extract_between(page_text, "Realistic mode", "Simulator mode")
    simulator_br = extract_between(page_text, "Simulator mode", "Premium Vehicle:")

    premium_vehicle = extract_between(page_text, "Premium Vehicle:", "Squadron Vehicle:")
    squadron_vehicle = extract_between(page_text, "Squadron Vehicle:", "Pack Vehicle:")
    pack_vehicle = extract_between(page_text, "Pack Vehicle:", "On Marketplace:")
    on_marketplace = extract_between(page_text, "On Marketplace:", "Release Date:")
    release_date = extract_between(page_text, "Release Date:", "Statistics for")

    return {
        "vehicle_slug": vehicle_slug,
        "vehicle_url": vehicle_url,
        "country": country,
        "vehicle_type": vehicle_type,
        "rank": pd.to_numeric(rank, errors="coerce"),
        "arcade_br": pd.to_numeric(arcade_br, errors="coerce"),
        "realistic_br": pd.to_numeric(realistic_br, errors="coerce"),
        "simulator_br": pd.to_numeric(simulator_br, errors="coerce"),
        "is_premium": premium_vehicle == "Yes",
        "is_squadron": squadron_vehicle == "Yes",
        "is_pack": pack_vehicle == "Yes",
        "on_marketplace": on_marketplace == "Yes",
        "release_date_raw": release_date,
        "pulled_at": pulled_at,
    }


def parse_vehicle_stat_charts(
    vehicle_soup: BeautifulSoup,
    vehicle_slug: str,
    vehicle_url: str,
    pulled_at: str,
) -> pd.DataFrame:
    rows = []

    chart_canvases = vehicle_soup.select("canvas[data-symfony--ux-chartjs--chart-view-value]")

    for canvas in chart_canvases:
        raw_value = canvas.get("data-symfony--ux-chartjs--chart-view-value")
        if not raw_value:
            continue

        chart_json = json.loads(html_lib.unescape(raw_value))

        labels = chart_json.get("data", {}).get("labels", [])
        datasets = chart_json.get("data", {}).get("datasets", [])

        parent_tab = canvas.find_parent(class_="tab-pane")
        parent_id = parent_tab.get("id") if parent_tab else ""

        match = re.search(r"vehicle-mode-metric-([a-z]+)-(.+)", parent_id)

        if match:
            mode_code = match.group(1)
            metric_code = match.group(2)
        else:
            chart_id = canvas.get("id", "")
            id_match = re.search(r"_([ars])_30$", chart_id)
            mode_code = id_match.group(1) if id_match else None
            metric_code = None

        mode_map = {
            "a": "arcade",
            "r": "realistic",
            "s": "simulator",
        }

        mode = mode_map.get(mode_code, mode_code)

        for dataset in datasets:
            metric_label = dataset.get("label")
            values = dataset.get("data", [])

            for date_value, metric_value in zip(labels, values):
                rows.append({
                    "vehicle_slug": vehicle_slug,
                    "vehicle_url": vehicle_url,
                    "mode": mode,
                    "mode_code": mode_code,
                    "metric": metric_label,
                    "metric_code": metric_code,
                    "date": date_value,
                    "value": metric_value,
                    "pulled_at": pulled_at,
                })

    return pd.DataFrame(rows)


def keep_latest_n_dates_per_vehicle(df: pd.DataFrame, n_dates: int = 30) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    kept_parts = []

    for vehicle_slug, group in df.groupby("vehicle_slug", sort=False):
        latest_dates = (
            group["date"]
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(n_dates)
        )

        kept = group[group["date"].isin(latest_dates)].copy()
        kept = kept.sort_values("date")
        kept_parts.append(kept)

    return pd.concat(kept_parts, ignore_index=True)


def pull_one_vehicle_realistic_30_days(vehicle_slug: str, vehicle_url: str) -> pd.DataFrame:
    pulled_at = utc_now_iso()

    response = requests.get(vehicle_url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    vehicle_soup = BeautifulSoup(response.text, "lxml")

    vehicle_info = parse_vehicle_info_from_page(
        vehicle_soup=vehicle_soup,
        vehicle_slug=vehicle_slug,
        vehicle_url=vehicle_url,
        pulled_at=pulled_at,
    )

    stats_long_df = parse_vehicle_stat_charts(
        vehicle_soup=vehicle_soup,
        vehicle_slug=vehicle_slug,
        vehicle_url=vehicle_url,
        pulled_at=pulled_at,
    )

    if stats_long_df.empty or "mode" not in stats_long_df.columns:
        raise ValueError("No chart data found")

    realistic_long_df = stats_long_df[stats_long_df["mode"] == "realistic"].copy()

    if realistic_long_df.empty:
        raise ValueError("No realistic chart data found")

    realistic_wide_df = (
        realistic_long_df
        .pivot_table(
            index=["vehicle_slug", "vehicle_url", "mode", "date", "pulled_at"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    realistic_wide_df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_per_")
        .replace("-", "_")
        for column in realistic_wide_df.columns
    ]

    realistic_wide_df["date"] = pd.to_datetime(realistic_wide_df["date"])
    realistic_wide_df = realistic_wide_df.sort_values("date").reset_index(drop=True)

    vehicle_info_df = pd.DataFrame([vehicle_info])

    final_df = realistic_wide_df.merge(
        vehicle_info_df,
        on=["vehicle_slug", "vehicle_url", "pulled_at"],
        how="left",
    )

    front_cols = [
        "date",
        "vehicle_slug",
        "country",
        "vehicle_type",
        "rank",
        "realistic_br",
        "is_premium",
        "is_squadron",
        "is_pack",
        "on_marketplace",
        "release_date_raw",
        "mode",
    ]

    remaining_cols = [column for column in final_df.columns if column not in front_cols]
    final_df = final_df[front_cols + remaining_cols]

    final_df = keep_latest_n_dates_per_vehicle(final_df, n_dates=30)

    return final_df


def attach_index_metadata(vehicle_df: pd.DataFrame, index_row: pd.Series) -> pd.DataFrame:
    vehicle_df = vehicle_df.copy()

    vehicle_df["vehicle_id"] = index_row.get("vehicle_id")
    vehicle_df["vehicle_name"] = index_row.get("vehicle_name")
    vehicle_df["index_type"] = index_row.get("index_type")
    vehicle_df["index_type_code"] = index_row.get("index_type_code")
    vehicle_df["index_role"] = index_row.get("index_role")
    vehicle_df["index_role_code"] = index_row.get("index_role_code")
    vehicle_df["index_country"] = index_row.get("index_country")
    vehicle_df["index_rank"] = index_row.get("index_rank")
    vehicle_df["index_battles"] = index_row.get("index_battles")
    vehicle_df["index_win_rate"] = index_row.get("index_win_rate")
    vehicle_df["index_efficiency"] = index_row.get("index_efficiency")
    vehicle_df["index_mode"] = index_row.get("index_mode")
    vehicle_df["pic"] = index_row.get("pic")

    return vehicle_df


def save_checkpoint(
    all_vehicle_dfs: list[pd.DataFrame],
    failed_vehicles: list[dict],
    run_date: str,
    count: int,
) -> None:
    checkpoint_df = pd.concat(all_vehicle_dfs, ignore_index=True) if all_vehicle_dfs else pd.DataFrame()
    checkpoint_df = keep_latest_n_dates_per_vehicle(checkpoint_df, n_dates=30)

    checkpoint_path = CHECKPOINT_DIR / f"ground_realistic_30_days_checkpoint_{run_date}_{count}.csv"
    checkpoint_df.to_csv(checkpoint_path, index=False)

    failed_checkpoint_df = pd.DataFrame(failed_vehicles)
    failed_checkpoint_path = LOGS_DIR / f"ground_realistic_30_days_failed_checkpoint_{run_date}_{count}.csv"
    failed_checkpoint_df.to_csv(failed_checkpoint_path, index=False)

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Rows so far: {len(checkpoint_df)}")
    print(f"Vehicles so far: {checkpoint_df['vehicle_slug'].nunique() if not checkpoint_df.empty else 0}")
    print(f"Failures so far: {len(failed_vehicles)}")


def run_pipeline() -> None:
    run_date = utc_today()
    run_started_at = utc_now_iso()

    print("=" * 80)
    print("War Thunder Stats ThunderSkill pipeline")
    print(f"Run date: {run_date}")
    print(f"Started at: {run_started_at}")
    print("=" * 80)

    index_df = pull_ground_realistic_vehicle_index(limit=100)

    if index_df.empty:
        raise RuntimeError("Vehicle index pull returned no rows.")

    index_dated_path = RAW_INDEX_DIR / f"ground_realistic_vehicle_index_{run_date}.csv"
    index_latest_path = RAW_INDEX_DIR / "ground_realistic_vehicle_index_latest.csv"

    index_df.to_csv(index_dated_path, index=False)
    index_df.to_csv(index_latest_path, index=False)

    print(f"Saved dated index: {index_dated_path}")
    print(f"Saved latest index: {index_latest_path}")
    print(f"Index rows: {len(index_df)}")

    all_vehicle_dfs = []
    failed_vehicles = []

    vehicles_to_pull = index_df.copy().reset_index(drop=True)

    print(f"Vehicles to scrape: {len(vehicles_to_pull)}")

    for i, row in vehicles_to_pull.iterrows():
        vehicle_slug = row["vehicle_slug"]
        vehicle_url = row["vehicle_url"]

        print(f"Pulling {i + 1}/{len(vehicles_to_pull)}: {vehicle_slug}")

        try:
            vehicle_df = pull_one_vehicle_realistic_30_days(
                vehicle_slug=vehicle_slug,
                vehicle_url=vehicle_url,
            )

            vehicle_df = attach_index_metadata(vehicle_df, row)
            vehicle_df = keep_latest_n_dates_per_vehicle(vehicle_df, n_dates=30)

            all_vehicle_dfs.append(vehicle_df)

        except Exception as exc:
            failed_vehicles.append({
                "vehicle_slug": vehicle_slug,
                "vehicle_url": vehicle_url,
                "vehicle_name": row.get("vehicle_name"),
                "index_type": row.get("index_type"),
                "index_role": row.get("index_role"),
                "index_country": row.get("index_country"),
                "index_battles": row.get("index_battles"),
                "error": str(exc),
                "failed_at": utc_now_iso(),
            })
            print(f"FAILED: {vehicle_slug} | {exc}")

        if (i + 1) % 50 == 0:
            save_checkpoint(
                all_vehicle_dfs=all_vehicle_dfs,
                failed_vehicles=failed_vehicles,
                run_date=run_date,
                count=i + 1,
            )

        time.sleep(random.uniform(1.0, 2.0))

    full_df = pd.concat(all_vehicle_dfs, ignore_index=True) if all_vehicle_dfs else pd.DataFrame()
    full_df = keep_latest_n_dates_per_vehicle(full_df, n_dates=30)

    failed_df = pd.DataFrame(failed_vehicles)

    final_dated_path = SNAPSHOT_DIR / f"ground_realistic_30_days_{run_date}.csv"
    final_latest_path = PROCESSED_DIR / "ground_realistic_30_days_latest.csv"

    failed_dated_path = LOGS_DIR / f"ground_realistic_30_days_failed_{run_date}.csv"
    failed_latest_path = LOGS_DIR / "ground_realistic_30_days_failed_latest.csv"

    full_df.to_csv(final_dated_path, index=False)
    full_df.to_csv(final_latest_path, index=False)

    failed_df.to_csv(failed_dated_path, index=False)
    failed_df.to_csv(failed_latest_path, index=False)

    print("=" * 80)
    print("Pipeline complete")
    print(f"Saved dated dataset: {final_dated_path}")
    print(f"Saved latest dataset: {final_latest_path}")
    print(f"Saved dated failures: {failed_dated_path}")
    print(f"Saved latest failures: {failed_latest_path}")
    print(f"Rows: {len(full_df)}")
    print(f"Vehicles: {full_df['vehicle_slug'].nunique() if not full_df.empty else 0}")
    print(f"Failures: {len(failed_df)}")

    if not full_df.empty:
        print(f"Date range: {full_df['date'].min()} to {full_df['date'].max()}")
        print(f"BR range: {full_df['realistic_br'].min()} to {full_df['realistic_br'].max()}")

    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()

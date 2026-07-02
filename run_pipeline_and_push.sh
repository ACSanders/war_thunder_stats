#!/usr/bin/env bash
set -euo pipefail

cd /root/war-thunder-data-pipeline

source .venv/bin/activate

echo "============================================================"
echo "Starting ThunderSkill pipeline at $(date -u)"
echo "============================================================"

echo "Syncing with origin/main before pipeline run..."
git fetch origin main
git rebase origin/main

python pull_thunderskill.py

echo "============================================================"
echo "Pipeline finished at $(date -u)"
echo "Checking for data changes..."
echo "============================================================"

git add data/processed/ground_realistic_30_days_latest.csv
git add data/raw/vehicle_index/ground_realistic_vehicle_index_latest.csv

if git diff --cached --quiet; then
    echo "No data changes to commit."
else
    git commit -m "Update ThunderSkill data $(date -u +%Y-%m-%d)"

    echo "Syncing with origin/main before push..."
    git fetch origin main
    git rebase origin/main

    git push origin main
fi

echo "Done at $(date -u)"

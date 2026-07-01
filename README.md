# War Thunder Stats

**A data science dashboard for War Thunder Realistic Ground vehicle performance, meta signals, lineup building, and vehicle analysis.**

[Live app](https://warthunderstats.app) · [Alternate domain](https://war-thunder-stats.com) · [Render deployment](https://warthunderstats.onrender.com)

---

## Overview

**War Thunder Stats** is an applied data science project that turns ThunderSkill vehicle performance data into an interactive Streamlit analytics product. The app is designed to help players explore the ground-vehicle meta using empirical performance signals rather than anecdotal tier lists.

The project combines a custom data pipeline, feature engineering, composite scoring, similarity modeling, density-based clustering, trend detection, underplayed-vehicle discovery, and lineup optimization into a polished public web app.

The app currently includes five core views:

1. **Nation Meta** — compare national performance across BR ranges.
2. **Vehicle Rankings** — rank and inspect vehicles by Combat Effectiveness.
3. **Performance Clusters** — discover statistical vehicle archetypes with HDBSCAN.
4. **Meta Signals** — identify rising performers and underplayed high-value vehicles.
5. **Lineup Builder** — generate same-nation BR-bounded lineups using CE Score and role variety.

---

## Live App

The app is deployed on Render with a custom domain:

**https://warthunderstats.app**

The app is also reachable through:

**https://war-thunder-stats.com**

Render's default subdomain:

**https://warthunderstats.onrender.com**

---

## Project Motivation

War Thunder vehicle discussions are often driven by subjective experience, community perception, or isolated stat comparisons. This project asks a more data-driven question:

> Given recent tracked-player performance data, which vehicles and nations appear to be overperforming within their Battle Rating context?

The goal is not to claim that the data perfectly measures intrinsic vehicle strength. Instead, the app provides a structured way to explore observed player performance outcomes using transparent, reproducible analytics.

---

## Data Source

The app uses ThunderSkill vehicle data, processed into a rolling 30-day snapshot for Realistic Ground vehicles.

Primary processed dataset:

```text
data/processed/ground_realistic_30_days_latest.csv
```

Important caveat:

> Battle counts in the app are labeled as **sample battles** because they represent ThunderSkill tracked-user samples, not global War Thunder totals.

Vehicles with missing Realistic BR metadata are preserved in the raw/processed data, but they are excluded from BR-normalized views because Combat Effectiveness and other BR-relative views require valid BR context.

---

## Data Pipeline

This project includes an automated data pipeline that collects, processes, and publishes updated ThunderSkill data.

Pipeline highlights:

- Scrapes ThunderSkill vehicle leaderboard and per-vehicle pages.
- Extracts public vehicle metadata and recent performance statistics.
- Processes rolling 30-day Realistic Ground vehicle data.
- Produces cleaned CSV outputs for the Streamlit app.
- Runs on a DigitalOcean Ubuntu droplet.
- Uses a scheduled cron job for recurring updates.
- Pushes updated processed data back to GitHub for the deployed app.

Architecture:

```text
ThunderSkill
    ↓
Python scraping / parsing pipeline
    ↓
Raw + processed CSV files
    ↓
GitHub repository
    ↓
Render-hosted Streamlit app
```

This separation keeps the user-facing app lightweight while allowing the data pipeline to run independently.

---

## Core Metric: Combat Effectiveness Score

The central metric in the app is **Combat Effectiveness Score**, or **CE Score**.

CE Score is a BR-relative composite metric designed to measure how much a vehicle overperforms the average vehicle at its exact Realistic BR.

The score combines:

- Ground frags per death / K-D
- Ground frags per battle
- Win rate
- Data confidence from sample battles

Each metric is smoothed toward the BR average using an empirical-Bayes-style reliability weight:

```text
reliability = battles / (battles + 100)
```

This prevents low-sample vehicles from dominating rankings purely because of noisy outlier performance.

The combined score is:

```text
z_total =
  0.40 × z(K/D, ground frags per death)
+ 0.40 × z(ground frags per battle)
+ 0.15 × z(win rate)
+ 0.05 × z(data confidence = log1p(sample battles))

Combat Effectiveness Score = clip(50 + 15 × z_total, 0, 100)
```

Interpretation:

- **50** ≈ BR-average
- **~65** = strong performance
- **~95+** = exceptional performance

ThunderSkill's own efficiency score is intentionally excluded because it is already a composite. The CE Score is designed to be transparent, BR-relative, and aligned with the app's analytical goals.

---

## App Features

### 1. Nation Meta

The Nation Meta tab summarizes national performance across BR ranges.

Includes:

- Nation × BR heatmap
- Nation daily trend view
- Nation distribution plots
- Nation strength dumbbell chart
- Nation strength summary table
- BR meta curve

This tab helps answer questions like:

- Which nations are strongest at specific BR ranges?
- Where does a nation overperform or underperform?
- How does national performance vary across the BR ladder?

---

### 2. Vehicle Rankings

The Vehicle Rankings tab ranks individual vehicles by Combat Effectiveness and provides deeper vehicle-level exploration.

Includes:

- Top CE vehicle bar chart
- Top performers table
- Daily K/D stability plot
- Vehicle detail selector
- Vehicle stat cards
- Radar chart
- Similar vehicles by K-nearest neighbors
- Similarity ladder
- Vehicle trend chart

The vehicle detail section uses unique labels such as:

```text
M4A3 (105) — France, BR 3.0
```

This avoids duplicate-name bugs across nations and variants.

#### K-Nearest Neighbors Similarity

The app uses a KNN-style similarity helper to identify vehicles with similar performance profiles. Similarity is based on filtered vehicles and favors:

- Same or similar vehicle type
- Nearby BR
- Similar performance metrics
- Comparable observed sample context

This makes the app useful not just for ranking vehicles, but for answering questions like:

> If I like this vehicle, what performs similarly in the current data?

---

### 3. Performance Clusters

The Performance Clusters tab uses **HDBSCAN** to identify vehicle archetypes within the currently filtered slice.

Clustering features are intentionally focused:

```text
CE Score
K/D
log1p(sample battles)
```

Win rate, frags per battle, nation, BR, and vehicle type are shown as context, but they are not directly used as clustering inputs.

#### Why HDBSCAN?

HDBSCAN is a density-based clustering method. Unlike KMeans, it does not force every vehicle into a fixed number of clusters. It can find natural dense pockets in the data and mark uncertain points as outliers.

This is a good fit for game-performance data because the vehicle meta is not cleanly separated into perfect groups. Some vehicles form obvious performance pockets, while others are unusual, sparse, or hard to classify.

The app converts raw HDBSCAN cluster IDs into user-friendly archetype labels such as:

- Core Meta
- Underplayed Meta
- Solid Picks
- Popular Strugglers
- Niche Signals
- Off-Meta
- Outliers

Visualizations include:

- 3D feature-space cluster plot
- 2D archetype map of CE Score vs K/D
- Cluster quality cards
- Cluster profile summary
- Cluster drilldown
- Full clustered vehicle table

The tab also includes a busy-slice warning when a broad filter creates too many small archetypes, nudging users toward cleaner BR-focused analysis.

---

### 4. Meta Signals

The Meta Signals tab looks for movement and overlooked value inside the current filter slice. It is designed to answer two questions:

1. Which vehicles are improving over the rolling 30-day window?
2. Which strong vehicles appear underplayed relative to their performance?

The tab has two sections:

#### Rising Performers

Rising Performers uses a **Daily Performance Score** to identify vehicles with meaningful improvement during the rolling 30-day window.

The Daily Performance Score is a daily BR-relative analogue of CE Score used only for trend detection. It is not the official 30-day CE Score.

Each day, within each exact BR:

- K/D and frags per battle are log-transformed to reduce skew.
- K/D, frags per battle, and win rate are robust z-scored against same-BR vehicles.
- The weighted score is mapped to a 0-100 scale, where 50 is roughly BR-average.
- No per-day sample smoothing is applied because daily battle counts are often too small.

Momentum is calculated as:

```text
early_score = mean of the first 10 observed daily scores
late_score  = mean of the last 10 observed daily scores
gain        = late_score - early_score
coverage    = min(observed_days / 30, 1)
reliability = sample_battles / (sample_battles + 50)

Momentum Score = gain × coverage × reliability
```

This rewards vehicles that show meaningful improvement, have enough observed days, and have enough sample battles to make the trend more credible.

Outputs include:

- Top rising vehicles by Momentum Score
- Daily Performance Score trend chart
- CE=50 reference line
- Compact Rising Performers table
- Formula and interpretation expander

#### Underplayed Meta

Underplayed Meta identifies vehicles that look strong and lethal but are not among the most-played vehicles in the current filtered slice.

The metric is **Meta Value Score**.

Percentiles are computed within the currently filtered slice and scaled from 0 to 1 internally. Battle counts are ThunderSkill tracked-user sample battles, not global War Thunder totals.

```text
performance_strength =
  0.50 × CE percentile
+ 0.30 × K/D percentile
+ 0.20 × frags-per-battle percentile

underplay_strength = 1 - sample-battles percentile
reliability        = sample_battles / (sample_battles + 50)

Meta Value Score = 100 × performance_strength
                       × (0.50 + 0.50 × underplay_strength)
                       × reliability
```

Interpretation:

- **performance_strength** rewards high CE, K/D, and frags per battle.
- **underplay_strength** rewards lower battle volume relative to the filtered slice.
- **reliability** prevents very tiny samples from dominating.
- The final score surfaces vehicles that are both strong and relatively underplayed.

Outputs include:

- Opportunity Map: sample-battle percentile vs CE Score
- Point size by Meta Value Score
- Nation-colored scatter points
- Top Meta Value Score bar chart
- Underplayed Meta results table
- Formula and interpretation expander

This tab makes the app less static by highlighting both emerging performers and overlooked high-value vehicles.

---

### 5. Lineup Builder

The Lineup Builder replaces a basic trends view with a more practical decision-support tool.

It builds same-nation lineups within a selected BR range.

Controls include:

- Nation
- BR minimum and maximum
- Lineup size: 3, 4, or 5 vehicles
- Include or exclude premiums
- Allowed vehicle types
- Minimum sample battles
- Optional role-variety preference

The scoring model is intentionally simple and explainable:

```text
Lineup Score = average CE Score + optional role-variety bonus
```

The role-variety bonus rewards lineups that include multiple vehicle types:

```text
1 unique type  = +0.0
2 unique types = +2.5
3 unique types = +5.0
4+ types       = +7.5
```

The app searches candidate combinations from the top eligible vehicles by CE Score and returns the best-scoring lineups.

Outputs include:

- Recommended lineup cards
- Lineup Score
- Average CE Score
- Average K/D
- Median sample battles
- Types covered
- Recommended lineup CE bar chart
- CE vs K/D performance map
- Alternative lineup table
- Eligible candidate pool expander

This turns the app from a dashboard into a practical lineup recommendation tool.

---

## Technical Stack

Core app:

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn

Modeling / analytics:

- Empirical-Bayes-style smoothing
- Robust BR-relative z-scoring
- Composite metric design
- Daily trend scoring
- Momentum scoring
- Underplayed-vehicle value scoring
- K-nearest-neighbor similarity search
- HDBSCAN density clustering
- Combination search for lineup optimization

Infrastructure:

- GitHub for version control and app/data delivery
- DigitalOcean droplet for scheduled data pipeline execution
- Cron for automated pipeline runs
- Render for production Streamlit hosting
- Cloudflare Registrar / DNS for custom domain routing

---

## Repository Structure

Representative structure:

```text
war_thunder_stats/
├── streamlit_app.py
├── features.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
│       └── ground_realistic_30_days_latest.csv
├── logs/
└── README.md
```

Key files:

- `streamlit_app.py` — Streamlit UI and tab layout
- `features.py` — feature engineering, scoring, clustering, similarity, meta-signal, and lineup helpers
- `requirements.txt` — Python dependencies
- `data/processed/ground_realistic_30_days_latest.csv` — latest app-ready data snapshot

---

## Applied Science Highlights

This project is designed as an applied science portfolio project, not just a dashboard.

It demonstrates:

### Data product thinking

- Converts messy third-party game-performance data into a usable analytical product.
- Designs user-facing metrics that are interpretable and aligned with player decisions.
- Balances statistical rigor with practical usability.
- Turns static rankings into actionable discovery and recommendation workflows.

### Metric design

- Builds a transparent composite metric.
- Uses BR-relative comparisons instead of global rankings.
- Applies reliability weighting to reduce low-sample overreaction.
- Designs separate metrics for ranking, trend detection, underplayed discovery, clustering, and lineup optimization.
- Avoids opaque third-party composite metrics when they duplicate the role of the custom score.

### Machine learning and unsupervised learning

- Uses KNN-style similarity to surface comparable vehicles.
- Uses HDBSCAN to discover density-based vehicle archetypes.
- Handles outliers and weak cluster structure transparently.
- Separates modeling features from contextual display features to avoid noisy cluster bias.

### Trend detection and opportunity scoring

- Computes a daily BR-relative performance score for trend detection.
- Uses early-window vs late-window comparison to detect meaningful performance movement.
- Applies coverage and reliability weighting to reduce noise from sparse daily observations.
- Uses percentile-based opportunity scoring to surface strong but underplayed vehicles.

### Optimization and recommendation

- Implements a lineup search system over constrained same-nation BR ranges.
- Scores combinations using CE Score and role-variety preferences.
- Produces alternative recommendations instead of a single opaque answer.

### Engineering and deployment

- Builds an automated data pipeline on a cloud droplet.
- Uses cron for scheduled updates.
- Maintains a GitHub-backed data/app workflow.
- Deploys a production Streamlit app on Render.
- Configures custom domains through Cloudflare.

---

## Limitations

This project uses observed ThunderSkill tracked-user data, not official global Gaijin data.

Important limitations:

- Sample battles are not total War Thunder battles.
- ThunderSkill tracked users may not represent the full player population.
- High or low performance can reflect player selection effects.
- Win rate is influenced by team, matchmaking, nation popularity, and lineup context.
- CE Score measures observed overperformance within BR, not intrinsic vehicle power.
- Daily Performance Score is a trend-detection analogue, not the official CE Score.
- Momentum Score can be influenced by sparse or unstable daily samples.
- Meta Value Score is slice-relative and should be interpreted as an exploratory signal.
- Clusters are slice-relative and should be interpreted as exploratory archetypes.
- Lineup Builder recommendations are data-backed suggestions, not guarantees of in-game performance.

The app is best used as a structured analytical tool for exploring the meta, not as an absolute truth source.

---

## Future Work

Potential next improvements:

- Add historical snapshot comparisons across multiple pipeline refreshes.
- Add stronger lineup constraints such as requiring at least one light tank, SPAA, or tank destroyer.
- Add lineup comparison mode.
- Add BR-specific annotations for notable vehicles and meta shifts.
- Add performance caching or precomputed artifacts for faster Render loads.
- Precompute cluster labels or meta-signal outputs in the pipeline if app-side computation becomes expensive.
- Expand beyond Realistic Ground to other modes if the data quality supports it.

---

## Attribution

Independent data science project by **Adam Sanders / War Thunder Stats**.

Data source: **ThunderSkill**.

This project is not affiliated with Gaijin Entertainment or ThunderSkill.

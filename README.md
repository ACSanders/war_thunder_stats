# War Thunder Stats

**A data-driven dashboard for War Thunder Ground Realistic Battles.**

[Live app](https://warthunderstats.app) · [Alternate domain](https://war-thunder-stats.com) · [Render deployment](https://warthunderstats.onrender.com)

---

## What it is

War Thunder Stats turns recent ThunderSkill vehicle data into a set of practical tools for exploring the Ground RB meta.

The app is built around a simple idea: raw stats are much more useful when vehicles are compared against others at the same Battle Rating. Instead of relying on global averages or subjective tier lists, the dashboard focuses on BR-relative performance, sample strength, recent movement, and lineup context.

It is designed for players who want to answer questions like:

- Which nations are performing best at a given BR?
- Which vehicles stand out against their direct peers?
- Is a vehicle improving or falling off?
- Which strong vehicles are seeing relatively little tracked play?
- What is the best lineup I can build within a BR range?
- Between two tanks, which one has the stronger recent statistical profile?

---

## App sections

### Nation Meta

Explore how nations perform across the BR ladder.

Includes:

- Nation-by-BR heatmaps
- Daily nation trends
- Performance distributions
- BR strength curves
- Summary tables and comparison views

This is the quickest way to see where a nation is strong, weak, or unusually competitive.

### Vehicle Rankings

Rank vehicles using **Combat Effectiveness Score**, then drill into individual vehicles.

Includes:

- BR-relative vehicle rankings
- CE Score breakdown
- Evidence badges based on tracked sample battles
- Vehicle images and stat cards
- Radar and trend views
- K/D stability
- Similar-vehicle recommendations

The CE breakdown shows exactly where a vehicle gains or loses points rather than presenting the score as a black box.

### Performance Clusters

Use HDBSCAN to discover groups of vehicles with similar statistical profiles.

The clustering view uses a focused set of features:

- CE Score
- K/D
- Sample-battle volume

The app turns raw cluster IDs into readable archetypes such as Core Meta, Solid Picks, Popular Strugglers, Niche Signals, and Outliers. The goal is exploratory: find patterns and unusual vehicles that may not stand out in a normal ranking table.

### Meta Signals

Look beyond static rankings.

**Rising Performers** highlights vehicles whose BR-relative daily performance has improved across the recent window. Results include evidence level, early and late scores, current CE, and a plain-language explanation of why the vehicle appears.

**Strong & Less-Played** finds vehicles that combine top-tier CE performance with below-median tracked usage. It uses a simple qualification rule rather than another opaque composite score:

- Top 20% in CE within the current comparison set
- Bottom 50% in tracked sample battles
- At least 50 sample battles

The section includes a quadrant-style map, featured results, full qualifiers, and clear caveats about tracked-user usage versus global popularity.

### Lineup Builder

Build same-nation lineups inside a selected BR range.

Controls include:

- Nation
- Minimum and maximum BR
- Lineup size
- Premium inclusion
- Vehicle types
- Minimum sample battles
- Optional role-variety preference

Lineups are scored using average CE plus a small, transparent diversity bonus. The app returns a recommended lineup along with alternatives and supporting charts.

### Tank vs Tank

Compare two vehicles head to head using recent BR-relative performance.

The tab includes:

- Searchable vehicle selectors
- Same-BR comparison by default
- Local filters plus an option to ignore the app-wide filters
- Side-by-side vehicle cards
- CE-based verdicts: Clear Edge, Slight Edge, Too Close to Call, or Descriptive Only
- Exact-BR percentile comparisons for CE, K/D, frags per battle, and win rate
- Core-profile labels such as Core Metric Sweep, Broad Performance Advantage, 2–1 Performance Edge, Tradeoff Matchup, Narrow Advantage, and Dead Heat
- Signature advantages for each vehicle
- Side-by-side CE contribution waterfalls
- Momentum, stability, sample evidence, and recent trend views
- Cross-BR and SPAA caveats

CE Score determines the overall statistical edge. The category profile explains where each vehicle leads. It is a performance comparison, not a duel simulator or a prediction that one vehicle would beat another in a one-on-one fight.

---

## Combat Effectiveness Score

**Combat Effectiveness Score (CE)** is the central ranking metric in the app.

It combines:

- Ground frags per death
- Ground frags per battle
- Win rate
- Sample evidence

The performance metrics are compared against vehicles at the same exact Realistic BR. Low-sample results are smoothed toward the BR median so that a small number of unusually good battles does not dominate the rankings.

The score is centered around 50:

- Around **50**: roughly neutral within the vehicle's BR peer group
- Around **65**: strong
- Around **80**: very strong
- Around **95+**: exceptional

CE is intentionally transparent. The app shows the contribution from each component, the effect of sample evidence, and any final clipping at the 0–100 bounds.

ThunderSkill's own efficiency score is not used in CE because it is already a composite metric.

---

## Evidence badges

Sample size matters, so the app labels vehicles by tracked battle volume:

- **Limited:** fewer than 50 sample battles
- **Developing:** 50–149
- **Solid:** 150–499
- **Strong:** 500 or more

These badges do not guarantee that a result is correct, but they make it easier to distinguish a strong signal from a thin sample.

Battle counts throughout the app refer to **ThunderSkill tracked-user sample battles**, not all War Thunder battles.

---

## Data pipeline

The app is backed by an automated data pipeline running on a DigitalOcean Ubuntu server.

The pipeline:

1. Pulls public ThunderSkill vehicle and performance data
2. Extracts recent Ground RB statistics
3. Cleans and validates the results
4. Applies vehicle metadata and BR corrections
5. Builds the processed dataset used by the app
6. Pushes updated data to GitHub for the Render deployment

```text
ThunderSkill
    ↓
Python data pipeline
    ↓
Processed Ground RB dataset
    ↓
GitHub
    ↓
Render-hosted Streamlit app
```

The pipeline runs separately from the web app, which keeps the dashboard lightweight and makes recurring data updates easier to manage.

---

## Technical stack

### App and analysis

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- HDBSCAN

### Methods used

- Robust BR-relative standardization
- Empirical-Bayes-style smoothing
- Composite metric design
- K-nearest-neighbor similarity
- Density-based clustering
- Trend and momentum scoring
- Percentile-based discovery rules
- Constrained lineup search
- Deterministic matchup classification

### Infrastructure

- GitHub
- DigitalOcean
- Cron
- Render
- Cloudflare DNS

---

## Repository structure

```text
war_thunder_stats/
├── streamlit_app.py
├── features.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
│       └── ground_realistic_30_days_latest.csv
└── README.md
```

Key files:

- `streamlit_app.py` — Streamlit interface and tab layout
- `features.py` — scoring, clustering, similarity, meta signals, lineup, and matchup helpers
- `data/processed/ground_realistic_30_days_latest.csv` — current app-ready dataset

---

## Why I built it

War Thunder vehicle discussions are often based on personal experience, community reputation, or isolated stat comparisons. Those perspectives are useful, but they can be hard to separate from confirmation bias and small samples.

This project does not try to replace player judgment. It adds a consistent analytical layer: compare vehicles within their BR context, show the strength of the evidence, explain how scores are built, and make it easy to move from broad meta questions to specific vehicle decisions.

It is also an applied data science project built around a real product rather than a standalone notebook. It combines data collection, feature engineering, model-driven exploration, recommendation logic, visualization, deployment, and ongoing maintenance.

---

## Limitations

The app uses ThunderSkill tracked-user data, not official global Gaijin data.

That means:

- The sample may not represent the full player population
- Tracked battle counts are not global usage totals
- Strong results may partly reflect who chooses to play a vehicle
- Win rate depends on teams, matchmaking, nation popularity, and lineup context
- CE measures observed BR-relative performance, not pure vehicle power
- Momentum and cluster labels are exploratory signals
- Lineup recommendations are data-backed suggestions, not guarantees
- Tank vs Tank compares statistical profiles and does not simulate combat outcomes
- Cross-BR Tank vs Tank results are descriptive because the vehicles are evaluated against different peer groups

The app is best used as a research and decision-support tool, not as an absolute source of truth.

---

## Planned work

Likely next steps include:

- Creator Mode for turning app results into structured video research notes
- Historical comparisons across saved pipeline snapshots
- More lineup constraints and lineup-vs-lineup comparisons
- Additional caching and precomputed artifacts for faster load times
- Expansion to other modes when the data quality supports it

---

## Attribution

Independent project by **Adam Sanders / War Thunder Stats**.

Data source: **ThunderSkill**.

War Thunder Stats is not affiliated with Gaijin Entertainment or ThunderSkill.

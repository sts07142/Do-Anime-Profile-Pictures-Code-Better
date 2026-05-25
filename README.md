# Do Anime PFP Developers Actually Code Better?

**A data-driven investigation into the correlation between developer profile picture types and coding skills.**

We analyzed **45,715 GitHub users** and **48,611 Codeforces users**, classified each profile image into 4 categories (**Anime / Human / Other / Default**) with YOLOv8 face detection + CLIP zero-shot verification, and asked: *do Anime-PFP developers actually score higher on activity / skill metrics?*

> **Taxonomy (updated 2026-05-24)**: "**Anime**" now means **any drawn / illustrated / rendered character** — Japanese anime, Studio Ghibli, Disney/Pixar, 2D/3D illustration, digital painting, and photorealistic AI/illustrated portraits. "**Human**" means a **photograph of a real person**, including filtered/edited selfies. (Earlier versions counted only Japanese-otaku anime as Anime.) Manual-validation accuracy below was measured under the *old* taxonomy and must be re-labeled for the new one.

> **Disclaimer**: This study analyzes **correlations only** — it does not claim causation. Region, sampling group, account age, and primary language are known confounders.

**Short answer**: No, once you compare apples to apples. The "Anime PFP ⇒ higher activity" effect that shows up in raw Anime-vs-NonAnime comparisons is almost entirely the *Default-vs-everyone-else* gap. Once you exclude Default-PFP users, **Anime users are statistically indistinguishable from Human / Other users on GitHub `total_contributions`** (δ ≈ −0.03, negligible). On **Codeforces `rating`**, however, once we CLIP-verify the CF Anime labels too (removing YOLO false-positive photos), Anime shows a **small but robust edge over Human** (δ = +0.18) — so *activity* shows no difference but *skill rating* shows a small one. Both conclusions hold even after 20% adversarial classification noise (see [Sensitivity](#classification-error-sensitivity-does-the-headline-survive-20-mislabeling)).

## Research Questions

| RQ | Platform | Question | This study's answer |
|----|----------|----------|---------------------|
| RQ1 | GitHub | Do anime-PFP users have different activity (`total_contributions`, followers, stars, repos)? | Higher than **Default**, indistinguishable from **Human / Other** (δ ≈ −0.03, negligible). |
| RQ2 | Codeforces | Do anime-PFP users have different `rating` / rank distributions? | Higher than Default/Other, and a **small edge over Human** (δ = +0.18) after CLIP-verifying CF Anime. Anime share rises with rank through grandmaster (14% → 38%). |

## Key Findings

### TL;DR

Once you exclude Default-PFP users (who barely contribute by construction),
**Anime users are statistically indistinguishable from Human / Other users**
on GitHub `total_contributions`. The widely-cited "Anime PFP = higher
activity" effect is mostly the Default-vs-everyone-else gap in disguise.
**Codeforces `rating` tells a slightly different story**: after CLIP-verifying
the CF Anime labels (Anime / Human / Other medians 1,204 / 1,064 / 1,162;
Default 962), Anime sits a small but robust step above Human (δ = +0.18, small).
So the meme has no leg to stand on for GitHub *activity*, but a small one for
Codeforces *skill rating*.

### Headline — GitHub `total_contributions` by 4-cat (n = 45,715)

`total_contributions` = commits + PRs + issues + reviews in the last year (GraphQL).

| Category | n | median | mean | p75 |
|---|---:|---:|---:|---:|
| Anime   |  6,552 |   154 |   871.3 |   743 |
| Human   | 15,797 | **190** | 902.2 |   928 |
| Other   |  7,932 |    98 |   782.4 |   589 |
| Default | 15,434 |     1 |    99.1 |    19 |

**Kruskal–Wallis 4-group**: H = 9523.0, p < 1e-300 ***

### Pairwise Cliff's δ (GitHub `total_contributions`)

| Pair | δ | size | p |
|---|---:|---|---:|
| Anime vs Human   | **−0.029** | negligible | 6.0e-04 |
| Anime vs Other   | +0.085     | negligible | 8.7e-19 |
| Anime vs Default | **+0.579** | large      | ~0      |
| Human vs Other   | +0.110     | negligible | 2.0e-43 |
| Human vs Default | +0.571     | large      | ~0      |
| Other vs Default | +0.486     | large      | ~0      |
| **Anime vs NonAnime (all)** | **+0.234** | small | 1.2e-203 |
| **Anime vs (Human + Other) — excl Default** | **+0.009** | negligible | 0.26 |

Read this carefully:

- "Anime vs NonAnime" looks like a real effect (small, +0.234).
- But when you split NonAnime into Default vs (Human + Other), the *only*
  thing Anime beats decisively is Default. Anime vs Human is actually a
  slight loss (median 154 < 190, δ = −0.029).
- The earlier framing of this project — "anime PFP correlates with higher
  activity" — survives only in the trivial sense that "having any PFP at
  all" correlates with higher activity than "having the default Gravatar."

### PFP distribution

| Platform | n | Anime | Human | Other | Default |
|----------|------:|------:|------:|------:|--------:|
| GitHub     | 45,715 | 14.3% | 34.6% | 17.4% | 33.8% |
| Codeforces | 48,611 | 17.8% |  8.7% | 15.4% | 58.0% |

CF is dominated by Default (Gravatar/no-avatar) and Other (logos / emblems /
abstract icons typical for competitive-programming profiles). Real-face
Human PFPs are rare on CF (8.7%) compared to GitHub (34.6%).

### Region breakdown — median `total_contributions` (GitHub)

| Region | n | Anime | Human | Other | Default |
|---|---:|---:|---:|---:|---:|
| East Asia      | 2,481 | 172 | 173 | 175 |  11 |
| Southeast Asia |   478 | 258 | 286 | 152 |  14 |
| South Asia     | 1,427 | 187 | 145 | 195 |  18 |
| Middle East    |   372 | 186 | 252 | 110 |  72 |
| Europe         | 5,476 | 329 | 410 | 236 |  42 |
| North America  | 4,119 | 358 | 311 | 214 |  23 |
| Latin America  |   674 | 220 | 160 | 113 |  24 |
| Oceania        |   384 | 366 | 383 | 211 |  26 |
| Africa         |   251 | 269 | 278 | 189 |  24 |
| Unknown        |30,053 | 105 | 107 |  63 |   1 |

Under the broadened taxonomy Anime is competitive in more regions (top in
North America, Latin America), but Europe / Southeast Asia still match or beat
it with Human, and East Asia is a near tie. The "Unknown"
region dominates the sample (location parsing recall is the bottleneck — see
`src/analysis/geo.py`).

See `data/processed/analysis_total_contrib.md` for full per-region Cliff's δ
tables and CF cross-platform context (regenerate with
`python scripts/analyze_total_contrib.py`).

### RQ2 — Codeforces (Anime share by rank, unchanged)

| Rank | n | Anime % |
|------|---:|-------:|
| newbie               | 31,651 | 13.6% |
| pupil                | 7,665 | 23.0% |
| specialist           | 4,869 | 26.6% |
| expert               | 2,872 | 29.2% |
| candidate master     | 781 | 29.2% |
| master               | 427 | 31.6% |
| international master | 113 | 29.2% |
| grandmaster          | 138 | **37.7%** |
| international grandmaster | 74 | 21.6% |
| legendary grandmaster | 21 | 23.8% |

Anime share rises with rank through grandmaster, then **tapers off** at the
top three tiers (where n is small). Anime vs Non-Anime on `rating`:
**δ = +0.272**, small effect. CF lacks a contributions-equivalent metric, so
the headline `total_contributions` comparison is GitHub-only; CF is shown
here purely as a directional cross-check.

### CF rating × 4-cat — a small Anime edge (where GitHub had none)

| Category | n | median rating | mean | p75 |
|---|---:|---:|---:|---:|
| Anime   |  8,670 | **1,204** | 1,218 | 1,447 |
| Human   |  4,225 | 1,064 | 1,094 | 1,347 |
| Other   |  7,503 | 1,162 | 1,187 | 1,420 |
| Default | 28,213 |   962 |   976 | 1,220 |

Kruskal–Wallis: H = 3300.4, p ~ 0 (***). After CLIP-verifying CF Anime (which
moved 3,363 YOLO false-positive photos into Human), **Anime's median (1,204)
sits above Human (1,064) and Other (1,162)** — Anime vs Human δ = +0.184
(small), stable across all sensitivity scenarios. This is the one place the
pattern diverges from GitHub. Caveat: part of the gap is a composition effect
(low-rated mislabels joining Human dragged its median down from 1,131 to 1,064).

### Classification-error sensitivity (does the headline survive ~20% mislabeling?)

`scripts/sensitivity_analysis.py` re-runs the headline δ under five
robustness scenarios. Short version: **yes, the conclusion is stable** —
Anime vs Human stays in the *negligible* range and Anime vs NonAnime stays
*small* across every scenario on both platforms.

GitHub `total_contributions` (Anime n=6,552 / Human n=15,797):

| Scenario | Anime vs Human δ | Anime vs NonAnime δ |
|---|---:|---:|
| Baseline                            | −0.029              | +0.234            |
| Bootstrap 95% CI                    | [−0.046, −0.013]    | [+0.220, +0.249]  |
| Drop ambiguous (n=373, margin<0.2)  | −0.030              | +0.235            |
| Random 10% flip MC                  | [−0.032, −0.011]    | [+0.236, +0.252]  |
| Random 20% flip MC                  | [−0.029, −0.003]    | [+0.243, +0.261]  |
| Worst-case Anime→Human 20% (n=1,310)| −0.018              | +0.235            |

Codeforces `rating` (Anime n=8,670 / Human n=4,225):

| Scenario | Anime vs Human δ | Anime vs NonAnime δ |
|---|---:|---:|
| Baseline                            | +0.184 (small)     | +0.272             |
| Bootstrap 95% CI                    | [+0.163, +0.203]   | [+0.260, +0.284]   |
| Drop ambiguous (n=249, margin<0.2)  | +0.187             | +0.273             |
| Random 10% flip MC                  | [+0.128, +0.154]   | [+0.253, +0.264]   |
| Random 20% flip MC                  | [+0.084, +0.117]   | [+0.235, +0.250]   |
| Worst-case Anime→Human 20% (n=1,734)| +0.151 (small)     | +0.272             |

**Takeaway**: on **GitHub** the headline is robust — even after deliberately
corrupting 20% of Anime labels (random ±20% Anime↔Human swaps, or relabeling
the 20% least-confident Anime calls as Human), Anime stays statistically
indistinguishable from Human. On **Codeforces** the small Anime-over-Human edge
(δ = +0.184) is now **stable across every scenario** — including the worst-case
flip (+0.151) — because CF Anime is finally CLIP-verified, so `clip_margin` is
populated and the earlier +0.779 artifact (from ranking Anime by `anime_conf`)
is gone. GitHub says "no difference"; CF says "small, robust difference".

The full sensitivity report (with `maxRating` and per-scenario p-values) is
in `data/processed/analysis_sensitivity.md`.

#### Confidence signals now persisted in the CSV

Both `data/processed/classified_4cat.csv` and `codeforces_classified.csv`
now carry four classifier-validation columns (added by
`scripts/add_validation_cols.py`):

| Column | Meaning |
|---|---|
| `clip_margin` | \|clip_anime_score − clip_human_score\|, range 0–1. NaN where CLIP wasn't run. |
| `is_ambiguous` | `clip_margin < 0.20` — CLIP was within 20 pts of a coin flip. |
| `low_yolo_conf` | `0 < anime_conf < 0.05` — YOLO found a face but barely. |
| `clip_evaluated` | True iff CLIP was actually run for this row (GH: all face-detected; CF: only the 9.6K Photo-split rows). |

CF Anime labels still come from YOLO@0.3 alone — a known asymmetry, called
out in §4 of `analysis_sensitivity.md` and listed under Future Work.

## How It Works

### Classification Pipeline

Profile images are classified into four categories using **YOLOv8 anime-face
detection** followed by **CLIP zero-shot verification** (anime vs real human):

| Category | Method | Description |
|----------|--------|-------------|
| **Default** | URL + pixel analysis | Gravatar URL or identicon pattern (5×5 symmetric grid, ≤4 colors) |
| **Anime** | YOLOv8x6 + CLIP | Face detected by YOLO **AND** CLIP picks an illustration prompt (Japanese anime, Ghibli, Disney/Pixar, 2D/3D illustration, digital painting, AI/illustrated portrait) |
| **Human** | YOLOv8x6 + CLIP | Face detected by YOLO **AND** CLIP picks a real-photo prompt (a photograph of a real person, including filtered/edited selfies) |
| **Other** | Exclusion | YOLO did not detect a face — logos, scenery, abstract art, anything without a recognizable face |

### Sampling Strategy

GitHub users are collected via 8 stratified groups (~1,700 per group, configurable) to ensure diversity:

- **ML / AI**: Contributors to core ML / LLM / generative-AI repos (transformers, pytorch, langchain, vllm, ComfyUI, …). Sub-tiers: `core_ml`, `llm`, `generative`.
- **Trending**: Contributors to repos curated by [trendshift.io](https://trendshift.io) — daily front page, all-time history page, and ~10 non-AI topic pages. Snapshots of each fetched list are saved under `data/raw/trendshift_snapshots/` for reproducibility. Depends on trendshift HTML structure (no official API), so this group is best-effort and may degrade if upstream changes.
- **Popular OSS**: Contributors to repos with stars > 1,000
- **Language Community**: Top repo contributors per language (Python, JS, Go, Rust, Java, TS, C++, Kotlin)
- **General Active**: Users split by follower ranges
- **New Users**: Created after 2025
- **Long-term Users**: Created before 2015
- **Org Members**: Google, Microsoft, Vercel, Rust-lang, etc.

> **Note on incremental runs**: when re-running `collect` against an existing
> `data/raw/sampled_users.json`, users already labeled with another group are
> NOT re-classified into the new groups — dedup happens by user id. To get a
> clean partition under the new priority order, delete `sampled_users.json`
> and re-run.

## Architecture

```
├── app.py                          # Streamlit dashboard (Overview / GitHub / CF / Cross-Platform / 🌐 Region Compare / 🔍 Inspector)
├── src/
│   ├── device.py                   # Auto-detect CUDA/MPS/CPU
│   ├── pipeline/                   # Orchestration (shared by CLI + notebook)
│   │   ├── runner.py               # run_collect, per-stage functions
│   │   └── cli.py                  # `collect` CLI entry point
│   ├── classification/             # YOLOv8 + CLIP classifier
│   │   └── classifier.py           # classify_avatars (resumable, 4-cat)
│   ├── collectors/                 # GitHub data collection
│   │   ├── github_client.py        # REST API client (sync, for sampling)
│   │   ├── async_github_client.py  # REST API client (async, for enrichment)
│   │   ├── enricher.py             # User detail enrichment (async)
│   │   ├── contributions.py        # GraphQL contributions collector
│   │   ├── sampler.py              # Stratified sampling (8 groups)
│   │   └── rate_limit.py           # Shared rate-limit handler
│   ├── images/                     # Image processing
│   │   ├── downloader.py           # Avatar download (async)
│   │   └── prefilter.py            # Default avatar pre-filter
│   └── analysis/
│       └── geo.py                  # Free-text location → ISO-2 → region
├── scripts/                        # One-shot analysis / data-shaping scripts
│   ├── cf_split_photo.py           # Re-runs YOLO@0.01+CLIP on CF Photo bucket → Human/Other (resumable)
│   ├── add_validation_cols.py      # Derives clip_margin / is_ambiguous / low_yolo_conf / clip_evaluated columns
│   ├── analyze_4cat.py             # 9-metric 4-cat analysis (legacy headline)
│   ├── analyze_total_contrib.py    # total_contributions-centred 4-cat + region report
│   ├── sensitivity_analysis.py     # Headline δ under 5 mislabeling scenarios
│   └── anime_follower_ratio.py     # Anime-PFP follower homophily on GitHub
├── notebooks/                      # Interactive analysis
│   ├── 01_anime_classification         # Runs pipeline + visualizes 4-cat results
│   ├── 02_classification_validation    # Manual-label precision/recall
│   ├── 03_github_eda
│   ├── 04_github_statistical_test
│   ├── 05_codeforces_collection        # CF collection + 4-way classification
│   └── 06_codeforces_analysis          # CF 4-cat statistical analysis
├── docs/superpowers/specs/         # Design specs (this work: 2026-05-12-cf-4cat-classification-design.md)
└── data/                           # (gitignored)
    └── processed/
        ├── classified_4cat.csv         # GitHub 4-cat + clip_margin + clip_evaluated …
        ├── codeforces_classified.csv   # CF 4-cat + face_detected_lowconf + clip scores …
        ├── analysis_4cat.md            # Legacy 9-metric report
        ├── analysis_total_contrib.md   # Headline total_contributions report
        └── analysis_sensitivity.md     # Robustness check (5 scenarios)
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager)

### Installation

```bash
# 1. Clone
git clone https://github.com/sts07142/Do-Anime-Profile-Pictures-Code-Better.git
cd Do-Anime-Profile-Pictures-Code-Better

# 2. Create virtual environment & install dependencies
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
uv sync

# 3. Set up GitHub token
cp .env.example .env
# Edit .env and add your GITHUB_TOKEN (required for data collection)
```

## Usage

### Step 1: Run the full pipeline (CLI or notebook)

The `collect` command runs the entire pipeline — sample → enrich → prefilter →
download → classify (YOLOv8 + CLIP) — and resumes from the last checkpoint on
every re-run.

```bash
# Default: target 10,200 users total, full pipeline
collect

# Smaller run for testing
collect --total 600

# Show progress snapshot across all stages
collect --status

# Common options
collect --total 5000 \
        --enrich-concurrency 20 \
        --download-concurrency 40 \
        --clip-model ViT-B-32 \
        --skip-classify       # skip the YOLO+CLIP step
```

The same pipeline is callable from Python / notebooks:

```python
from pipeline import PipelineConfig, run_collect
await run_collect(PipelineConfig(total=600))
```

### Step 2: Analysis (Notebooks)

```bash
# for mac
python -m ipykernel install --user
```

Run notebooks in order:

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `anime_classification` | Runs `pipeline.run_collect` and visualizes 4-way classification (Default/Anime/Human/Other) |
| 02 | `classification_validation` | Classification accuracy validation |
| 03 | `github_eda` | GitHub exploratory data analysis |
| 04 | `github_statistical_test` | Statistical hypothesis testing (KW, MW, Cliff's δ) |
| 05 | `codeforces_collection` | Codeforces data collection + 4-way avatar classification (Default/Anime/Human/Other) |
| 06 | `codeforces_analysis` | Codeforces statistical analysis |

> **Note**: Notebook 05 handles Codeforces data collection internally (no separate CLI step needed).

### Step 3: Headline analysis (one-shot scripts)

After the collection + classification CSVs exist, regenerate every headline number with:

```bash
# 1. If CF was collected before the 4-cat split landed, re-classify the Photo bucket.
.venv/bin/python scripts/cf_split_photo.py            # resumable; ~1.5h on MPS

# 2. Add confidence / ambiguity columns to both CSVs (no inference, fast).
.venv/bin/python scripts/add_validation_cols.py

# 3. Headline total_contributions × 4-cat × region report → analysis_total_contrib.md
.venv/bin/python scripts/analyze_total_contrib.py

# 4. Robustness check (bootstrap + MC flip scenarios) → analysis_sensitivity.md
.venv/bin/python scripts/sensitivity_analysis.py
```

All four scripts are idempotent and write to `data/processed/`. The robustness
results live in `data/processed/analysis_sensitivity.md`.

### Step 4: Dashboard

```bash
streamlit run app.py
```

## Rate Limit Handling

All GitHub API rate limits are handled automatically with retry + resume:

| API | Limit | Strategy |
|-----|-------|----------|
| REST API | 5,000 req/hr | Auto-wait until `X-RateLimit-Reset` |
| Search API | 30 req/min | Secondary rate limit detection + `Retry-After` |
| GraphQL API | 5,000 pts/hr | `RATE_LIMIT` error detection + reset query |
| 429 responses | varies | `Retry-After` header-based backoff |

All collection tasks support **checkpointing** — interrupted runs automatically resume from where they left off.

## Statistical Methods

- **Kruskal-Wallis H test**: Non-parametric comparison across 3 groups
- **Mann-Whitney U test**: Pairwise comparison (anime vs non-anime)
- **Cliff's delta (δ)**: Effect size measurement (|δ| > 0.474 = large)
- **Bonferroni correction**: Multiple comparison adjustment
- **Chi-square independence test**: Categorical variable association
- **Cramer's V**: Chi-square effect size

## Privacy

- All personal identifiers (username, avatar URL, bio, location) are **SHA-256 hashed** (8-char) in the dashboard
- Raw data is gitignored and never published
- Analysis uses only **publicly available** profile information

## Tech Stack

- **Collection**: `requests`, `aiohttp`, GitHub REST/GraphQL API, Codeforces API
- **Classification**: `ultralytics` (YOLOv8 anime face detection)
- **Analysis**: `pandas`, `scipy`, `scikit-learn`, `umap-learn`
- **Visualization**: `plotly`, `streamlit`, `matplotlib`, `seaborn`

## Future Work

- **Re-validate accuracy under the new taxonomy.** The 1,900 manual labels were drawn under the old (Japanese-anime-only) definition. Re-label a fresh sample under the broadened "any illustration = Anime" rule to report real precision/recall.
- **Disentangle the CF small effect (real vs composition).** CF Anime is now CLIP-verified (3,363 false-positive photos moved to Human, stabilizing the worst-case), and Anime shows δ = +0.18 over Human. But part of that gap is a composition effect — the low-rated mislabels that joined Human dragged its median down. Separating the genuine skill signal from this artifact is open work.
- **Investigate the GitHub vs CF anime-share gap (14.3% vs 17.8%).** Possible causes: real cultural difference, YOLO+CLIP threshold sensitivity against stylized non-anime art, sampling bias toward org/OSS accounts that tend not to use anime PFPs, or simply that the CF Anime YOLO@0.3 threshold is more permissive than the GitHub YOLO@0.01+CLIP pipeline. Threshold sweeps + manual labeling at the new scale would isolate this.
- **Re-balance stratified GitHub sampling.** Audit each of the 8 strata for anime-share / activity-level skew, and re-weight (or re-collect) to reduce confounds between "what group the user was sampled from" and "what PFP they use." `analyze_4cat.py` already shows the sampling-group × PFP chi-square is non-negligible.
- **Region coverage.** ~66% of GitHub users have no detected region (free-text `location` parser limits, see `src/analysis/geo.py`). Either expand the pattern table or switch to a proper geocoder for tighter regional analysis.
- **Confounder controls.** Region/country, account age, primary language, and sampling-group effects should be regressed out before making any causal claim. The headline "Anime ≈ Human" survives the current sensitivity analysis, but a multivariate regression with these controls would close the loop.

## License

MIT

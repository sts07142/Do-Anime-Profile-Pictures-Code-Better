# Do Anime PFP Developers Actually Code Better?

**A data-driven investigation into the correlation between developer profile picture types and coding skills.**

We analyzed **~9.3K GitHub users** and **~48.6K Codeforces users** to answer one burning question: *Is there a statistically significant correlation between having an anime profile picture and developer activity/skill metrics?*

> **Disclaimer**: This study analyzes **correlations only** — it does not claim causation.

## Research Questions

| RQ | Platform | Question |
|----|----------|----------|
| RQ1 | GitHub | Do anime PFP users have significantly different follower counts, stars, repos, and contributions? |
| RQ2 | Codeforces | Do anime PFP users have significantly different ratings and rank distributions? |

## Key Findings

### PFP distribution

| Platform | n | Anime | Photo | Default |
|----------|------:|------:|------:|--------:|
| GitHub   |  9,280 |  **5.9%** | 55.4% | 38.8% |
| Codeforces | 48,611 | **22.3%** | 19.7% | 58.0% |

### RQ1 — GitHub (Anime vs Non-Anime, Cliff's δ)

| Metric | Cliff's δ | Effect | p |
|--------|----------:|--------|---:|
| followers     | +0.219 | small | < 1e-17 |
| total_stars   | +0.224 | small | < 1e-19 |
| public_repos  | +0.234 | small | < 1e-19 |
| total_forks   | +0.167 | small | < 1e-12 |

Anime PFP users do trend higher on every GitHub activity metric, but the effect size is **small** — much smaller than the "medium-to-large" effect implied by earlier drafts of this project.

### RQ2 — Codeforces (Anime share by rank)

| Rank | n | Anime % |
|------|---:|-------:|
| newbie               | 31,651 | 18.4% |
| pupil                | 7,665 | 26.5% |
| specialist           | 4,869 | 30.9% |
| expert               | 2,872 | 32.5% |
| candidate master     | 781 | 33.9% |
| master               | 427 | 33.7% |
| international master | 113 | 36.3% |
| grandmaster          | 138 | **38.4%** |
| international grandmaster | 74 | 28.4% |
| legendary grandmaster | 21 | 33.3% |

Anime share rises with rank through grandmaster, then **tapers off** at the top three tiers (where n is small). Cliff's δ for rating (Anime vs Non-Anime) = **+0.208**, small effect.

### Cross-validation

Direction is consistent across both platforms — anime PFP correlates with slightly higher activity/skill metrics. But the absolute anime share differs dramatically (GitHub 5.9% vs CF 22.3%), which itself is a finding worth interrogating (see Future Work).

## How It Works

### Classification Pipeline

Profile images are classified into three categories using **YOLOv8 anime-face
detection** followed by **CLIP zero-shot verification** (anime vs real human):

| Category | Method | Description |
|----------|--------|-------------|
| **Default** | URL + pixel analysis | Gravatar URL or identicon pattern (5×5 symmetric grid, ≤4 colors) |
| **Anime** | YOLOv8x6 + CLIP | Face detected by YOLO **AND** CLIP says "anime" over "human" on the cropped bbox |
| **Photo** | Exclusion | Everything else (real photos, logos, YOLO-detected human faces) |

### Sampling Strategy

GitHub users are collected via 6 stratified groups (~1,700 per group, configurable) to ensure diversity:

- **Popular OSS**: Contributors to repos with stars > 1,000
- **Language Community**: Top repo contributors per language (Python, JS, Go, Rust, Java, TS, C++, Kotlin)
- **General Active**: Users split by follower ranges
- **New Users**: Created after 2025
- **Long-term Users**: Created before 2015
- **Org Members**: Google, Microsoft, Vercel, Rust-lang, etc.

## Architecture

```
├── app.py                          # Streamlit dashboard
├── src/
│   ├── device.py                   # Auto-detect CUDA/MPS/CPU
│   ├── pipeline/                   # Orchestration (shared by CLI + notebook)
│   │   ├── runner.py               # run_collect, per-stage functions
│   │   └── cli.py                  # `collect` CLI entry point
│   ├── classification/             # YOLOv8 + CLIP classifier
│   │   └── classifier.py           # classify_avatars (resumable)
│   ├── collectors/                 # GitHub data collection
│   │   ├── github_client.py        # REST API client (sync, for sampling)
│   │   ├── async_github_client.py  # REST API client (async, for enrichment)
│   │   ├── enricher.py             # User detail enrichment (async)
│   │   ├── contributions.py        # GraphQL contributions collector
│   │   ├── sampler.py              # Stratified sampling (6 groups)
│   │   └── rate_limit.py           # Shared rate-limit handler
│   └── images/                     # Image processing
│       ├── downloader.py           # Avatar download (async)
│       └── prefilter.py            # Default avatar pre-filter
├── notebooks/                      # Analysis & visualization
│   ├── 01_anime_classification     # Runs pipeline + visualizes results
│   ├── 02_classification_validation
│   ├── 03_github_eda
│   ├── 04_github_statistical_test
│   ├── 05_codeforces_collection    # CF data collection + classification
│   └── 06_codeforces_analysis
└── data/                           # (gitignored)
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
| 01 | `anime_classification` | Runs `pipeline.run_collect` and visualizes YOLOv8 + CLIP 3-way classification |
| 02 | `classification_validation` | Classification accuracy validation |
| 03 | `github_eda` | GitHub exploratory data analysis |
| 04 | `github_statistical_test` | Statistical hypothesis testing (KW, MW, Cliff's δ) |
| 05 | `codeforces_collection` | Codeforces data collection + avatar classification |
| 06 | `codeforces_analysis` | Codeforces statistical analysis |

> **Note**: Notebook 05 handles Codeforces data collection internally (no separate CLI step needed).

### Step 3: Dashboard

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

- **Scale up the GitHub sample to ~50K users** (currently ~9.3K) so it matches the Codeforces sample size and is powered enough to resolve small effects. The present GitHub sample also looks unbalanced across the 6 stratified groups, which likely compresses the observed effect sizes.
- **Investigate the 5.9% GitHub anime share.** This is suspiciously low next to Codeforces' 22.3%. The gap could reflect a real cultural difference, a YOLO+CLIP confidence-threshold issue (especially against stylized non-anime art), or sampling bias toward org/OSS accounts that tend not to use anime PFPs. Manual labeling + threshold sweeps are needed before reading too much into this number.
- **Re-balance stratified sampling.** Audit each of the 6 GitHub strata for anime-share / activity-level skew, and re-weight (or re-collect) to reduce confounds between "what group the user was sampled from" and "what PFP they use."
- **Classifier validation at the new scale.** Once the GitHub sample grows, re-run notebook 02 with a larger manually labeled set and publish precision/recall per class so the 5.9% number comes with a real confidence interval.
- **Confounder controls.** Region/country, account age, primary language, and sampling-group effects should be regressed out before making any strong claim about "anime PFP → higher skill."

## License

MIT

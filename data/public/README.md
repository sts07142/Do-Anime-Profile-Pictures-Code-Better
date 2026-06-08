# `data/public/` — Anonymized public datasets

This directory holds the **two public, de-identified CSVs** for the study
*"Do Anime PFP Developers Actually Code Better?"*. 

| File | Rows | Cols | Platform | Unit row |
|---|---:|---:|---|---|
| `classified_4cat_enriched_anon.csv` | 45,715 | 27 | GitHub     | one user |
| `codeforces_classified_anon.csv`    | 48,611 | 18 | Codeforces | one user |

(Row counts exclude the header.)

## How the data was anonymized

Per `scripts/anonymize_for_commit.py`:

- **Identifiers (username / handle) → SHA-256 hash (12 hex)**
  - `uid` / `handle` are concatenated with a 32-byte random salt and
    hashed with SHA-256; the first 12 hex chars (48 bits, ≈ 281 T values)
    are kept. Collision probability at ~50K users is effectively zero.
  - The salt (`data/processed/.anon_salt`) is **gitignored and never
    published** → no back-mapping is possible from the public CSV alone.
  - Resulting columns: `uid_hash`, `handle_hash`.
- **Free-text columns dropped**
  - GitHub: `uid`, `bio`, `company`, `location`, `created_at`
  - Codeforces: `handle`, `avatar_url`, `organization`, `registrationTimeSeconds`
- **Timestamps blurred to month**
  - GitHub `created_at` → `created_year_month` (e.g. `2014-04`)
  - CF `registrationTimeSeconds` → `registration_year_month`
- **Coarse geographic columns kept**
  - GitHub: `country_code` (ISO-2), `region` (continent-level)
  - CF: `country` (dropdown-backed, not free text)
- **Numeric / classification columns preserved as-is**
  - `followers`, `total_contributions`, `rating`, `anime_conf`,
    `clip_*`, etc. are kept unchanged so statistics reproduce exactly.

> Source rows with duplicate `uid`/`handle` (1 GitHub case) are
> de-duplicated with `keep="first"` before hashing.

## `classified_4cat_enriched_anon.csv` (GitHub)

One row per GitHub user — 4-category PFP classification + activity metrics.

### Columns

| # | Column | Type | Description |
|--:|---|---|---|
| 1  | `uid_hash`            | str   | Salted SHA-256 (first 12 hex) of the GitHub uid. Stable, anonymous user key. |
| 2  | `profile_type`        | str   | 4-cat label: `Anime` / `Human` / `Other` / `Default`. |
| 3  | `is_anime`            | bool  | Convenience flag for `profile_type == "Anime"`. |
| 4  | `anime_conf`          | float | Top YOLOv8x6 anime-face confidence (0–1). NaN/0 if no face detected. |
| 5  | `anime_faces`         | int   | Number of anime-face boxes returned by YOLO. |
| 6  | `clip_is_anime`       | bool  | True when CLIP zero-shot picks an illustration prompt over a real-photo prompt. |
| 7  | `clip_anime_score`    | float | CLIP score for the anime / illustration prompt (0–1). |
| 8  | `clip_human_score`    | float | CLIP score for the real-photo prompt (0–1). Two scores roughly sum to 1. |
| 9  | `top_language`        | str   | User's most-used repo language (`Python`, `C++`, …). |
| 10 | `clip_margin`         | float | `\|clip_anime_score − clip_human_score\|`. NaN where CLIP was not run. |
| 11 | `is_ambiguous`        | bool  | `clip_margin < 0.20` — CLIP was within 20 pts of a coin flip. |
| 12 | `low_yolo_conf`       | bool  | `0 < anime_conf < 0.05` — YOLO detected a face, but barely. |
| 13 | `clip_evaluated`      | bool  | True iff CLIP was actually run on this row (GH: every face-detected row). |
| 14 | `followers`           | float | GitHub follower count. |
| 15 | `public_repos`        | float | Number of public repositories. |
| 16 | `total_stars`         | float | Sum of stars across the user's repos. |
| 17 | `total_forks`         | float | Sum of forks across the user's repos. |
| 18 | `activity_grade`      | str   | Activity tier — `high` / `mid` / `low` / `dormant`. |
| 19 | `sampling_group`      | str   | Stratum the user was sampled from: `ml_ai`, `trending`, `popular_oss`, `language`, `general`, `new_users`, `long_term`, `org_members`. |
| 20 | `country_code`        | str   | ISO-2 country code (e.g. `FR`, `KR`). Empty when parsing failed. |
| 21 | `region`              | str   | Continent-level region (`Europe`, `East Asia`, …). |
| 22 | `commits`             | float | Commits in the last year (GraphQL contributions). |
| 23 | `prs`                 | float | Pull requests opened in the last year. |
| 24 | `issues`              | float | Issues opened in the last year. |
| 25 | `reviews`             | float | Reviews authored in the last year. |
| 26 | `total_contributions` | float | `commits + prs + issues + reviews` — the headline activity metric. |
| 27 | `created_year_month`  | str   | Account creation month (`YYYY-MM`, e.g. `2012-10`). |

### Class distribution

| Category | n | % |
|---|---:|---:|
| Default | 15,434 | 33.8% |
| Human   | 15,797 | 34.6% |
| Other   |  7,932 | 17.4% |
| Anime   |  6,552 | 14.3% |

## `codeforces_classified_anon.csv` (Codeforces)

One row per Codeforces user — 4-category PFP classification + rating.

### Columns

| # | Column | Type | Description |
|--:|---|---|---|
| 1  | `handle_hash`             | str   | Salted SHA-256 (first 12 hex) of the CF handle. |
| 2  | `rating`                  | int   | Current CF rating. |
| 3  | `maxRating`               | int   | All-time peak rating. |
| 4  | `rank`                    | str   | Current rank (`newbie` … `legendary grandmaster`). |
| 5  | `maxRank`                 | str   | Peak rank. |
| 6  | `profile_type`            | str   | 4-cat label — same taxonomy as GitHub. |
| 7  | `is_anime`                | bool  | `profile_type == "Anime"`. |
| 8  | `anime_conf`              | float | Top YOLOv8 anime-face confidence. CF Anime is labeled with **YOLO@0.3 alone**. |
| 9  | `anime_faces`             | int   | Number of anime-face boxes returned by YOLO. |
| 10 | `face_detected_lowconf`   | bool  | YOLO detected a face at very low confidence (used by `cf_split_photo.py`). |
| 11 | `clip_anime_score`        | float | CLIP anime score — populated only for Photo-bucket re-classification rows. |
| 12 | `clip_human_score`        | float | CLIP real-photo score. |
| 13 | `country`                 | str   | CF profile country (dropdown value, not free text). |
| 14 | `clip_margin`             | float | Same definition as GitHub. NaN when CLIP was not run. |
| 15 | `is_ambiguous`            | bool  | `clip_margin < 0.20`. |
| 16 | `low_yolo_conf`           | bool  | `0 < anime_conf < 0.05`. |
| 17 | `clip_evaluated`          | bool  | On CF, True only for the 9.6K Photo-split rows. |
| 18 | `registration_year_month` | str   | Registration month (`YYYY-MM`). |

### Class distribution

| Category | n | % |
|---|---:|---:|
| Default | 28,213 | 58.0% |
| Anime   |  8,670 | 17.8% |
| Other   |  7,503 | 15.4% |
| Human   |  4,225 |  8.7% |

> **Asymmetry warning**: GitHub Anime is labeled via YOLO@0.01 + CLIP
> verification, while CF Anime relies on YOLO@0.3 alone. CF Anime is
> therefore more vulnerable to false positives than GitHub Anime. See
> §4 of `data/processed/analysis_sensitivity.md` for details.

## `profile_type` classification pipeline (summary)

| Category | Rule |
|---|---|
| **Default** | Gravatar URL or identicon pattern (5×5 symmetric grid, ≤4 colors). |
| **Anime**   | YOLOv8x6 detects an anime face **AND** CLIP picks an illustration prompt (Japanese anime, Ghibli, Disney/Pixar, 2D/3D illustration, digital painting, AI / illustrated portrait). |
| **Human**   | YOLOv8x6 detects a face **AND** CLIP picks a real-photo prompt (a photograph of a real person, including filtered / edited selfies). |
| **Other**   | YOLO did not detect a face — logos, scenery, abstract art, etc. |

## License & ethics

- Only **publicly visible profile fields** (those GitHub / Codeforces
  expose to anyone) were collected.
- User identifiers are anonymized as described above; the salt is held
  out of this directory, so the public CSV cannot be reverse-mapped.
- Free-text fields (bio, location, company) and the original avatar
  images are NOT included here.
- If you redistribute the CSVs, please ship this README alongside so the
  anonymization method and limitations travel with the data.
- Code license: repo-root `LICENSE` (MIT).

## Limitations

- **Correlational only.** These CSVs do not support causal claims.
  Region, sampling group, account age, and primary language are known
  confounders.
- **CF 4-cat asymmetry.** Many CF rows have no CLIP evaluation, so CF
  Anime is not the same quality of label as GitHub Anime (see the
  asymmetry warning above).
- **Large "Unknown" region share on GitHub** — ~66% of GitHub users have
  no detected region (free-text `location` parser limits). Be careful
  with regional cuts.
- **Classification error in the low single-digit %.** Under the new
  taxonomy this has not been re-validated yet; the `is_ambiguous` and
  `low_yolo_conf` flags can be used to filter suspicious rows.

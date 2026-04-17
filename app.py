"""
Profile x Skill — GitHub x Codeforces Profile Image Analysis Dashboard

v2 analysis:
  - yolov8_animeface 3-way classification (Anime / Default / Photo)
  - RQ1 (GitHub) & RQ2 (Codeforces): Unified 4-tab structure
    [Distribution -> Group Comparison -> Statistical Tests -> Cross Analysis]

Privacy: No username/handle/avatar URL/location exposed. SHA-256 8-char anonymous IDs only.
"""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

# -- Page Config -------------------------------------------------------
st.set_page_config(
    page_title="Profile x Skill Analysis",
    layout="wide",
    page_icon="🎭",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background: #0e1117; padding: 12px; border-radius: 8px; border: 1px solid #262730; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1 { border-bottom: 2px solid #4ecdc4; padding-bottom: 8px; }
    h2 { color: #4ecdc4; }
    .caption-box { background: #1a1d24; padding: 12px 16px; border-left: 3px solid #4ecdc4;
                   border-radius: 4px; margin: 8px 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# -- Constants ---------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROC_DIR = DATA_DIR / 'processed'
CSV_3CAT_PATH = PROC_DIR / 'classified_3cat.csv'
CF_CSV_PATH = PROC_DIR / 'codeforces_classified.csv'
ENRICHED_PATH = RAW_DIR / 'enriched_users.json'
CONTRIB_PATH = RAW_DIR / 'contributions.json'

# Korean -> English label mapping (CSV data uses Korean labels)
LABEL_MAP = {'애니': 'Anime', '기본': 'Default', '일반': 'Photo'}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

COLORS_3CAT = {'Anime': '#ff6b6b', 'Default': '#c0c0c0', 'Photo': '#4ecdc4'}
ORDER_3CAT = ['Anime', 'Photo', 'Default']
RANK_ORDER = [
    'newbie', 'pupil', 'specialist', 'expert', 'candidate master',
    'master', 'international master', 'grandmaster',
    'international grandmaster', 'legendary grandmaster'
]
SENSITIVE_COLS = {'username', 'handle', 'avatar_url', 'bio', 'company',
                  'location', 'organization', 'country', 'country_code'}


# -- Helpers -----------------------------------------------------------
def anonymize_id(value) -> str:
    if pd.isna(value):
        return "—"
    return hashlib.sha256(str(value).encode()).hexdigest()[:8]


def remap_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remap Korean profile_type labels to English."""
    if 'profile_type' in df.columns:
        df['profile_type'] = df['profile_type'].map(LABEL_MAP).fillna(df['profile_type'])
    return df


def cliff_delta(x, y):
    u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
    d = (2 * u) / (len(x) * len(y)) - 1
    return d, p


def effect_label(d):
    ad = abs(d)
    return 'large' if ad > 0.474 else 'medium' if ad > 0.33 else 'small' if ad > 0.147 else 'negligible'


def sig_stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def winsorize(s, limit=0.01):
    lo, hi = s.quantile(limit), s.quantile(1 - limit)
    return s.clip(lower=lo, upper=hi)


def kpi_card(col, label, value, delta=None, help=None):
    col.metric(label, value, delta, help=help)


# -- Reusable Chart Builders -------------------------------------------

def chart_distribution_histogram(df, metric, title, nbins=60):
    fig = px.histogram(df, x=metric, color='profile_type',
                        category_orders={'profile_type': ORDER_3CAT},
                        color_discrete_map=COLORS_3CAT,
                        marginal='box', barmode='overlay', opacity=0.55,
                        nbins=nbins, title=title)
    fig.update_layout(height=420, legend_title_text='PFP Type')
    return fig


def chart_distribution_ecdf(df, metric, title):
    fig = px.ecdf(df, x=metric, color='profile_type',
                   category_orders={'profile_type': ORDER_3CAT},
                   color_discrete_map=COLORS_3CAT, title=title)
    fig.update_traces(line=dict(width=3))
    fig.update_layout(height=380, legend_title_text='PFP Type')
    return fig


def chart_distribution_violin(df, metric, title):
    fig = px.violin(df, x='profile_type', y=metric, color='profile_type',
                     category_orders={'profile_type': ORDER_3CAT},
                     color_discrete_map=COLORS_3CAT, box=True, points=False,
                     title=title)
    fig.update_layout(height=420, showlegend=False)
    return fig


def chart_binned_stacked(df, bin_col, bin_order, title, bin_label="Bin"):
    ct = pd.crosstab(df[bin_col], df['profile_type'], normalize='index') * 100
    ct = ct.reindex([b for b in bin_order if b in ct.index])
    ct = ct[[c for c in ORDER_3CAT if c in ct.columns]]
    long = ct.reset_index().melt(id_vars=bin_col, var_name='profile_type', value_name='pct')
    fig = px.bar(long, x=bin_col, y='pct', color='profile_type',
                  category_orders={'profile_type': ORDER_3CAT,
                                    bin_col: [b for b in bin_order if b in ct.index]},
                  color_discrete_map=COLORS_3CAT, barmode='stack',
                  title=title, text_auto='.1f',
                  labels={'pct': 'Ratio (%)', bin_col: bin_label})
    fig.update_layout(xaxis_tickangle=-25, height=440, legend_title_text='PFP Type')
    return fig


def chart_binned_lines(df, bin_col, bin_order, title, bin_label="Bin"):
    ct = pd.crosstab(df[bin_col], df['profile_type'], normalize='index') * 100
    ordered_bins = [b for b in bin_order if b in ct.index]
    ct = ct.reindex(ordered_bins)
    fig = go.Figure()
    for ptype in ORDER_3CAT:
        if ptype not in ct.columns:
            continue
        vals = ct[ptype]
        fig.add_trace(go.Scatter(
            x=vals.index, y=vals.values,
            mode='lines+markers+text',
            name=ptype,
            text=[f"{v:.1f}" for v in vals.values],
            textposition='top center',
            line=dict(color=COLORS_3CAT[ptype], width=3),
            marker=dict(size=11, line=dict(color='white', width=1.5)),
        ))
    fig.update_layout(
        title=title,
        xaxis_title=bin_label, yaxis_title='Ratio (%)',
        xaxis_tickangle=-25, height=440, legend_title_text='PFP Type',
    )
    return fig


def chart_median_iqr_grid(df, metrics, title="Median +/- IQR (Q1~Q3) Comparison", ncol=3):
    n = len(metrics)
    cols = min(ncol, n) if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics,
                         horizontal_spacing=0.10, vertical_spacing=0.18)
    for i, m in enumerate(metrics):
        r, c = i // cols + 1, i % cols + 1
        for t in ORDER_3CAT:
            s = df[df['profile_type']==t][m].dropna()
            if len(s) == 0: continue
            med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
            fig.add_trace(go.Bar(
                x=[t], y=[med],
                error_y=dict(type='data', symmetric=False,
                             array=[q3 - med], arrayminus=[med - q1]),
                marker_color=COLORS_3CAT[t], name=t, showlegend=(i == 0),
                text=f"{med:.0f}<br>IQR {q1:.0f}-{q3:.0f}",
                textposition='outside', hovertemplate='%{x}<br>median=%{y:.0f}<extra></extra>',
            ), row=r, col=c)
    fig.update_layout(height=300 * rows + 60, title_text=title, showlegend=True,
                       legend_title_text='PFP Type')
    return fig


def chart_percentile_curve(df, metric, title=None):
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    rows = []
    for t in ORDER_3CAT:
        s = df[df['profile_type']==t][metric].dropna()
        if len(s) == 0: continue
        for p in percentiles:
            rows.append({'profile_type': t, 'percentile': p, 'value': s.quantile(p/100)})
    pct_df = pd.DataFrame(rows)
    fig = px.line(pct_df, x='percentile', y='value', color='profile_type',
                   markers=True,
                   category_orders={'profile_type': ORDER_3CAT},
                   color_discrete_map=COLORS_3CAT,
                   title=title or f"{metric} Percentile Curve")
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(height=420, legend_title_text='PFP Type')
    return fig


def chart_activity_quantile_lines(df, metric, n_bins=10, title=None):
    """PFP type ratio by activity quantile — does higher activity correlate with more anime?"""
    df_s = df.copy()
    df_s['_rank'] = df_s[metric].rank(method='first')
    df_s['quantile'] = pd.qcut(df_s['_rank'], n_bins, labels=range(1, n_bins + 1))
    ct = pd.crosstab(df_s['quantile'], df_s['profile_type'], normalize='index') * 100
    ct = ct[[c for c in ORDER_3CAT if c in ct.columns]]

    fig = go.Figure()
    for ptype in ORDER_3CAT:
        if ptype not in ct.columns: continue
        vals = ct[ptype]
        fig.add_trace(go.Scatter(
            x=[f"Q{i}" for i in vals.index], y=vals.values,
            mode='lines+markers+text', name=ptype,
            text=[f"{v:.1f}" for v in vals.values],
            textposition='top center' if ptype == 'Anime' else 'bottom center',
            line=dict(color=COLORS_3CAT[ptype], width=3),
            marker=dict(size=11, line=dict(color='white', width=1.5)),
        ))
    bounds = df.groupby(pd.qcut(df[metric].rank(method='first'), n_bins,
                                  labels=range(1, n_bins + 1)),
                         observed=True)[metric].agg(['min', 'max'])
    tick_labels = [f"Q{i}<br><span style='font-size:10px'>[{int(bounds.loc[i,'min'])}-{int(bounds.loc[i,'max'])}]</span>"
                   for i in range(1, n_bins + 1)]
    fig.update_xaxes(tickvals=[f"Q{i}" for i in range(1, n_bins + 1)], ticktext=tick_labels)
    fig.update_layout(
        title=title or f"{metric} Quantile vs PFP Type Ratio — Does higher activity = more anime?",
        xaxis_title=f"{metric} Quantile (Q1=bottom 10%, Q10=top 10%)",
        yaxis_title='Ratio (%)',
        height=500, legend_title_text='PFP Type',
    )
    return fig


def chart_topN_cumulative(df, metric, title=None):
    """Top N% cumulative — X=top N%, Y=PFP type ratio"""
    sub = df[df[metric].notna()].copy()
    n = len(sub)
    if n == 0:
        return go.Figure()
    sub = sub.sort_values(metric, ascending=False).reset_index(drop=True)
    thresholds = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100]
    rows = []
    for top_pct in thresholds:
        k = max(1, int(n * top_pct / 100))
        top = sub.head(k)
        vc = top['profile_type'].value_counts(normalize=True) * 100
        for ptype in ORDER_3CAT:
            rows.append({
                'top_pct': top_pct,
                'profile_type': ptype,
                'pct': float(vc.get(ptype, 0)),
                'n': k,
            })
    plot = pd.DataFrame(rows)
    fig = go.Figure()
    for ptype in ORDER_3CAT:
        sub_p = plot[plot['profile_type'] == ptype]
        fig.add_trace(go.Scatter(
            x=sub_p['top_pct'], y=sub_p['pct'],
            mode='lines+markers+text', name=ptype,
            text=[f"{v:.1f}" for v in sub_p['pct']],
            textposition='top center' if ptype == 'Anime' else 'bottom center',
            line=dict(color=COLORS_3CAT[ptype], width=3),
            marker=dict(size=11, line=dict(color='white', width=1.5)),
            customdata=sub_p[['n']],
            hovertemplate=f"<b>{ptype}</b><br>Top %{{x}}%: %{{y:.1f}}%%<br>n=%{{customdata[0]:,}}<extra></extra>",
        ))
    fig.update_layout(
        title=title or f"Top N% ({metric}) PFP Type Ratio — Left = elite group",
        xaxis_title=f"Top N% (by {metric})",
        yaxis_title='PFP Type Ratio (%)',
        xaxis=dict(tickvals=thresholds, autorange='reversed'),
        height=500, legend_title_text='PFP Type',
    )
    return fig


def chart_effect_size(df_eff, title="Anime vs Non-Anime Effect Size"):
    fig = px.bar(df_eff, x='Metric', y="Cliff's δ", color='Effect',
                  text="Cliff's δ", title=title,
                  color_discrete_map={'large': '#d62728', 'medium': '#ff7f0e',
                                       'small': '#2ca02c', 'negligible': '#7f7f7f'})
    fig.add_hline(y=0.147, line_dash='dot', annotation_text='small', line_color='gray')
    fig.add_hline(y=0.33, line_dash='dot', annotation_text='medium', line_color='gray')
    fig.add_hline(y=0.474, line_dash='dot', annotation_text='large', line_color='gray')
    fig.update_traces(textposition='outside')
    fig.update_layout(height=380)
    return fig


def build_kw_mw_table(df, metrics):
    """Kruskal-Wallis + Mann-Whitney results table"""
    anime = df[df['profile_type'] == 'Anime']
    non_anime = df[df['profile_type'] != 'Anime']
    rows = []
    for m in metrics:
        try:
            h, p_kw = stats.kruskal(*[df[df['profile_type']==t][m].dropna() for t in ORDER_3CAT])
            d, p_mw = cliff_delta(anime[m].dropna(), non_anime[m].dropna())
            rows.append({
                'Metric': m, 'KW H': round(h, 1), 'KW p': f'{p_kw:.2e}',
                'Anime median': anime[m].median(),
                'Non-anime median': non_anime[m].median(),
                "Cliff's δ": round(d, 3), 'Effect': effect_label(d),
                'Significance': sig_stars(p_mw),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def build_posthoc_table(df, metric):
    """Pairwise post-hoc test (Bonferroni correction)"""
    rows = []
    for g1, g2 in combinations(ORDER_3CAT, 2):
        x = df[df['profile_type']==g1][metric].dropna()
        y = df[df['profile_type']==g2][metric].dropna()
        if len(x) == 0 or len(y) == 0: continue
        d, p = cliff_delta(x, y)
        p_bon = min(p * 3, 1.0)
        rows.append({
            'Comparison': f'{g1} vs {g2}',
            f'{g1} median': x.median(),
            f'{g2} median': y.median(),
            "Cliff's δ": round(d, 3),
            'p (Bonf.)': f'{p_bon:.2e}',
            'Significance': sig_stars(p_bon),
        })
    return pd.DataFrame(rows)


def kpi_row(df, total_label="Filtered Users"):
    col1, col2, col3, col4 = st.columns(4)
    kpi_card(col1, total_label, f"{len(df):,}")
    for col, ptype in zip([col2, col3, col4], ORDER_3CAT):
        n = (df['profile_type'] == ptype).sum()
        kpi_card(col, f"{ptype}", f"{n:,}", f"{n/max(len(df),1)*100:.1f}%")


# -- Data Loaders -------------------------------------------------------
@st.cache_data(show_spinner="Loading GitHub data...")
def load_gh():
    if not CSV_3CAT_PATH.exists():
        return None
    df = remap_labels(pd.read_csv(CSV_3CAT_PATH))

    if ENRICHED_PATH.exists():
        with open(ENRICHED_PATH) as f:
            enriched = json.load(f)
        metrics_df = pd.DataFrame([
            {
                'uid': u['user_id'],
                'followers': u.get('followers', 0),
                'public_repos': u.get('public_repos', 0),
                'total_stars': u.get('repos', {}).get('total_stars_received', 0),
                'total_forks': u.get('repos', {}).get('total_forks_received', 0),
                'activity_grade': u.get('activity_grade'),
                'sampling_group': u.get('sampling_group'),
            }
            for u in enriched.values()
        ])
        df = df.merge(metrics_df, on='uid', how='left')

    if CONTRIB_PATH.exists():
        with open(CONTRIB_PATH) as f:
            contributions = json.load(f)
        contrib_df = pd.DataFrame([
            {
                'uid': int(uid),
                'commits': c.get('commits', 0),
                'prs': c.get('prs', 0),
                'issues': c.get('issues', 0),
                'reviews': c.get('reviews', 0),
                'total_contributions': c.get('total', 0),
            }
            for uid, c in contributions.items()
        ])
        df = df.merge(contrib_df, on='uid', how='left')

    return df


@st.cache_data(show_spinner="Loading Codeforces data...")
def load_cf():
    if CF_CSV_PATH.exists():
        return remap_labels(pd.read_csv(CF_CSV_PATH))
    return None


gh = load_gh()
cf = load_cf()
gh_n = len(gh) if gh is not None else 0
cf_n = len(cf) if cf is not None else 0


# -- Sidebar ------------------------------------------------------------
st.sidebar.markdown("# 🎭 Profile x Skill")
st.sidebar.caption("Correlation between PFP type and developer skills")
st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("GitHub", f"{gh_n:,}")
col_b.metric("Codeforces", f"{cf_n:,}")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "GitHub Analysis (RQ1)",
    "Codeforces Analysis (RQ2)",
    "Cross-Platform",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("All personal identifiers are SHA-256 hashed.")
st.sidebar.caption("All charts always show **Anime / Photo / Default** together.")


# =====================================================================
# OVERVIEW
# =====================================================================
if page == "Overview":
    st.title("🎭 Do Anime PFP Developers Actually Code Better?")

    st.markdown("""
    <div class="caption-box">
      <b>Research Question</b>: Is there a statistically significant <b>correlation</b>
      between a developer's profile image type (Anime / Default / Photo) and their
      activity & skill metrics?
      <br><br>
      This study analyzes <b>correlations only</b> and does not claim causation.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    kpi_card(col1, "GitHub Users", f"{gh_n:,}", help="6 stratified sampling groups")
    kpi_card(col2, "Codeforces Users", f"{cf_n:,}", help="activeOnly rated users")
    kpi_card(col3, "Total Sample", f"{gh_n + cf_n:,}")
    kpi_card(col4, "Platforms", "2")

    st.markdown("---")

    st.subheader("3-Way Classification Distribution")
    col1, col2 = st.columns(2)
    for col, df, name in [(col1, gh, f"GitHub (n={gh_n:,})"),
                           (col2, cf, f"Codeforces (n={cf_n:,})")]:
        if df is None: continue
        vc = df['profile_type'].value_counts().reindex(ORDER_3CAT).fillna(0).reset_index()
        vc.columns = ['profile_type', 'count']
        fig = px.pie(vc, names='profile_type', values='count', hole=0.55,
                      color='profile_type', color_discrete_map=COLORS_3CAT, title=name)
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        fig.update_layout(showlegend=False, height=380, margin=dict(t=50, b=10))
        col.plotly_chart(fig, width='stretch')

    if gh is not None and cf is not None:
        st.subheader("Platform Comparison Summary")
        comp = pd.DataFrame({
            'Platform': ['GitHub', 'Codeforces'],
            'Total': [gh_n, cf_n],
            'Anime (%)': [(gh['profile_type']=='Anime').mean()*100,
                        (cf['profile_type']=='Anime').mean()*100],
            'Photo (%)': [(gh['profile_type']=='Photo').mean()*100,
                        (cf['profile_type']=='Photo').mean()*100],
            'Default (%)': [(gh['profile_type']=='Default').mean()*100,
                        (cf['profile_type']=='Default').mean()*100],
        }).round(1)
        st.dataframe(comp, width='stretch', hide_index=True)

    st.markdown("---")
    st.subheader("Key Findings")
    c1, c2, c3 = st.columns(3)
    c1.info("**RQ1 (GitHub)**\n\nAnime PFP users score higher on followers / stars / repos (Cliff's δ ≈ 0.17–0.23, small effect)")
    c2.warning("**RQ2 (Codeforces)**\n\nAnime ratio rises with rank: newbie 18.4% → grandmaster 38.4% (tapers off at top tiers)")
    c3.success("**Cross-Validation**\n\nDirection is consistent across two independent platforms, though absolute anime share differs (GitHub 5.9% vs CF 22.3%)")


# =====================================================================
# GITHUB ANALYSIS (RQ1) — Unified 4-tab structure
# =====================================================================
elif page == "GitHub Analysis (RQ1)":
    st.title("GitHub Analysis (RQ1)")
    st.caption("Correlation between GitHub PFP type and open-source activity metrics")

    if gh is None:
        st.error(f"`{CSV_3CAT_PATH}` not found. Run notebook 01 first.")
        st.stop()

    BASE_METRICS = ['followers', 'total_stars', 'public_repos', 'total_forks']
    CONTRIB_METRICS = ['commits', 'prs', 'issues', 'reviews', 'total_contributions']
    available_base = [m for m in BASE_METRICS if m in gh.columns]
    available_contrib = [m for m in CONTRIB_METRICS if m in gh.columns]
    contrib_coverage = 0
    if available_contrib:
        contrib_coverage = gh[available_contrib[0]].notna().sum() / len(gh) * 100
        if contrib_coverage > 0:
            st.info(
                f"Contribution data coverage: **{contrib_coverage:.1f}%** "
                f"({gh[available_contrib[0]].notna().sum():,}/{len(gh):,} users). "
                "Users without contribution data are excluded from contribution-based charts."
            )

    all_metrics = available_base + available_contrib
    default_m = 'commits' if 'commits' in all_metrics else all_metrics[0]
    main_metric = st.selectbox("Main Metric", all_metrics,
                                 index=all_metrics.index(default_m), key='gh_main_m')

    gf = gh.copy()
    if main_metric in available_contrib:
        gf = gf[gf[main_metric].notna()]
    metrics = available_base + available_contrib
    kpi_row(gf)

    tabs = st.tabs(["Distribution", "Group Comparison", "Statistical Tests", "Cross Analysis"])

    # Tab 1: Distribution
    with tabs[0]:
        st.subheader(f"1-1. {main_metric} Distribution (Winsorized 1%)")
        gf_w = gf.copy()
        gf_w[main_metric] = winsorize(gf_w[main_metric], 0.01)
        st.plotly_chart(chart_distribution_histogram(gf_w, main_metric,
                         f"{main_metric} Histogram + Marginal Box"),
                         width='stretch')

        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_distribution_violin(gf_w, main_metric,
                          f"{main_metric} Violin"), width='stretch')
        col2.plotly_chart(chart_distribution_ecdf(gf_w, main_metric,
                          f"{main_metric} ECDF"), width='stretch')

        st.subheader(f"1-2. {main_metric} Top N% PFP Type Distribution")
        st.caption("Percentile-based bins (left = low activity, right = top percentile).")
        gf_bin = gf.copy()
        pct_breaks = [0, 0.25, 0.50, 0.75, 0.90, 1.00]
        labels = ['Bottom 25%', '25-50%', '50-75%', 'Top 10-25%', 'Top 10%']
        gf_bin['pct_rank'] = gf_bin[main_metric].rank(method='first', pct=True)
        gf_bin['bin'] = pd.cut(gf_bin['pct_rank'], bins=pct_breaks,
                                labels=labels, include_lowest=True)

        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_binned_stacked(gf_bin, 'bin', labels,
                          f"{main_metric} Top N% Ratio (%) — Stacked", "Percentile Bin"),
                          width='stretch')
        col2.plotly_chart(chart_binned_lines(gf_bin, 'bin', labels,
                          f"{main_metric} Top N% Line — All 3 Types", "Percentile Bin"),
                          width='stretch')

    # Tab 2: Group Comparison
    with tabs[1]:
        st.subheader(f"2-1. Median + IQR ({len(metrics)} metrics)")
        st.plotly_chart(chart_median_iqr_grid(gf, metrics),
                          width='stretch')

        st.subheader(f"2-2. {main_metric} Percentile Curve")
        st.plotly_chart(chart_percentile_curve(gf, main_metric),
                          width='stretch')

        st.subheader(f"2-3. {main_metric} Quantile vs PFP Type Ratio")
        st.caption(
            f"How does the ratio of each PFP type change as {main_metric} quantile increases? "
            "Can be interpreted as an activity proxy similar to commit count."
        )
        st.plotly_chart(chart_activity_quantile_lines(gf, main_metric),
                          width='stretch')

        st.subheader(f"2-4. Top N% ({main_metric}) PFP Type Ratio")
        st.caption(
            "X-axis: Top N% by metric (further right = more inclusive). "
            "Y-axis: PFP type ratio within that group."
        )
        st.plotly_chart(chart_topN_cumulative(gf, main_metric),
                          width='stretch')

        with st.expander(f"All {len(metrics)} metrics — Quantile & Top N% charts"):
            for m in metrics:
                col1, col2 = st.columns(2)
                col1.plotly_chart(chart_activity_quantile_lines(gf, m,
                                    title=f"{m} Quantile vs PFP Ratio"),
                                    width='stretch')
                col2.plotly_chart(chart_topN_cumulative(gf, m,
                                    title=f"{m} Top N% PFP Ratio"),
                                    width='stretch')

        st.subheader("2-5. Effect Size (Cliff's δ)")
        eff_rows = []
        anime = gf[gf['profile_type']=='Anime']
        non_anime = gf[gf['profile_type']!='Anime']
        for m in metrics:
            d, p = cliff_delta(anime[m].dropna(), non_anime[m].dropna())
            eff_rows.append({'Metric': m, "Cliff's δ": round(d, 3), 'Effect': effect_label(d)})
        st.plotly_chart(chart_effect_size(pd.DataFrame(eff_rows)),
                          width='stretch')

    # Tab 3: Statistical Tests
    with tabs[2]:
        st.subheader("3-1. Kruskal-Wallis + Mann-Whitney U")
        st.dataframe(build_kw_mw_table(gf, metrics),
                       width='stretch', hide_index=True)

        st.subheader(f"3-2. Pairwise Post-Hoc — {main_metric} (Bonferroni)")
        st.dataframe(build_posthoc_table(gf, main_metric),
                       width='stretch', hide_index=True)

        st.subheader("3-3. Chi-Square Independence Test — PFP Type x Activity Grade")
        cont = pd.crosstab(gf['profile_type'], gf['activity_grade'])
        chi2, p_chi, dof, _ = stats.chi2_contingency(cont)
        n = cont.sum().sum()
        k = min(cont.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k))
        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Chi-Square", f"{chi2:.1f}")
        kpi_card(c2, "p-value", f"{p_chi:.2e}", sig_stars(p_chi))
        kpi_card(c3, "Cramer's V", f"{cramers_v:.3f}")

    # Tab 4: Cross Analysis
    with tabs[3]:
        grades_order = ['high', 'medium', 'low', 'dormant', 'active']
        grades_in = [g for g in grades_order if g in gf['activity_grade'].unique()]
        grades_in += [g for g in gf['activity_grade'].dropna().unique() if g not in grades_in]

        st.subheader("4-1. PFP Type x Activity Grade")
        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_binned_stacked(gf, 'activity_grade', grades_in,
                          "PFP Type Ratio by Activity Grade", "Activity Grade"),
                          width='stretch')
        col2.plotly_chart(chart_binned_lines(gf, 'activity_grade', grades_in,
                          "PFP Type Lines by Activity Grade", "Activity Grade"),
                          width='stretch')

        st.subheader("4-2. PFP Type x Primary Language (Top 10)")
        top_langs = gf['top_language'].value_counts().head(10).index.tolist()
        lang_df = gf[gf['top_language'].isin(top_langs)]
        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_binned_stacked(lang_df, 'top_language', top_langs,
                          "PFP Type Ratio by Language", "Language"),
                          width='stretch')
        col2.plotly_chart(chart_binned_lines(lang_df, 'top_language', top_langs,
                          "PFP Type Lines by Language", "Language"),
                          width='stretch')

        st.subheader("4-3. PFP Type x Sampling Group")
        grps = gf['sampling_group'].unique().tolist()
        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_binned_stacked(gf, 'sampling_group', grps,
                          "PFP Type Ratio by Sampling Group", "Group"),
                          width='stretch')
        col2.plotly_chart(chart_binned_lines(gf, 'sampling_group', grps,
                          "PFP Type Lines by Sampling Group", "Group"),
                          width='stretch')


# =====================================================================
# CODEFORCES ANALYSIS (RQ2) — Same 4-tab structure
# =====================================================================
elif page == "Codeforces Analysis (RQ2)":
    st.title("Codeforces Analysis (RQ2)")
    st.caption("Testing the 'Anime PFP = Pro' hypothesis")

    if cf is None:
        st.error(f"`{CF_CSV_PATH}` not found. Run notebook 05 first.")
        st.stop()

    main_metric = st.selectbox("Main Metric", ['rating', 'maxRating'], key='cf_main_m')
    cff = cf.copy()
    metrics = ['rating', 'maxRating']
    kpi_row(cff)

    tabs = st.tabs(["Distribution", "Group Comparison", "Statistical Tests", "Cross Analysis"])

    # Tab 1: Distribution
    with tabs[0]:
        st.subheader(f"1-1. {main_metric} Distribution")
        st.plotly_chart(chart_distribution_histogram(cff, main_metric,
                         f"{main_metric} Histogram + Marginal Box", nbins=60),
                         width='stretch')

        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_distribution_violin(cff, main_metric,
                          f"{main_metric} Violin"), width='stretch')
        col2.plotly_chart(chart_distribution_ecdf(cff, main_metric,
                          f"{main_metric} ECDF"), width='stretch')

        st.subheader(f"1-2. {main_metric} Binned PFP Type Distribution")
        cff_bin = cff.copy()
        rating_bins = [0, 1200, 1400, 1600, 1900, 2100, 2400, 4000]
        rating_labels = ['<1200', '1200-1400', '1400-1600', '1600-1900',
                         '1900-2100', '2100-2400', '2400+']
        cff_bin['rating_bin'] = pd.cut(cff_bin[main_metric], bins=rating_bins,
                                         labels=rating_labels, include_lowest=True)
        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_binned_stacked(cff_bin, 'rating_bin', rating_labels,
                          f"{main_metric} Binned Ratio (%)", "Rating Bin"),
                          width='stretch')
        col2.plotly_chart(chart_binned_lines(cff_bin, 'rating_bin', rating_labels,
                          f"{main_metric} Binned PFP Lines", "Rating Bin"),
                          width='stretch')

    # Tab 2: Group Comparison
    with tabs[1]:
        st.subheader("2-1. Median + IQR (rating / maxRating)")
        st.plotly_chart(chart_median_iqr_grid(cff, metrics),
                          width='stretch')

        st.subheader(f"2-2. {main_metric} Percentile Curve")
        st.plotly_chart(chart_percentile_curve(cff, main_metric),
                          width='stretch')

        st.subheader(f"2-3. {main_metric} Quantile vs PFP Type Ratio")
        st.caption(f"How does each PFP type ratio change as {main_metric} quantile increases?")
        st.plotly_chart(chart_activity_quantile_lines(cff, main_metric),
                          width='stretch')

        st.subheader(f"2-4. Top N% ({main_metric}) PFP Type Ratio")
        st.caption(
            "X-axis: Top N% by metric (left = elite). "
            "Y-axis: PFP type ratio. "
            "Intuitive test of the 'Anime PFP = Pro' hypothesis."
        )
        st.plotly_chart(chart_topN_cumulative(cff, main_metric),
                          width='stretch')

        st.subheader("2-5. Effect Size (Cliff's δ)")
        eff_rows = []
        anime = cff[cff['profile_type']=='Anime']
        non_anime = cff[cff['profile_type']!='Anime']
        for m in metrics:
            d, p = cliff_delta(anime[m].dropna(), non_anime[m].dropna())
            eff_rows.append({'Metric': m, "Cliff's δ": round(d, 3), 'Effect': effect_label(d)})
        st.plotly_chart(chart_effect_size(pd.DataFrame(eff_rows)),
                          width='stretch')

    # Tab 3: Statistical Tests
    with tabs[2]:
        st.subheader("3-1. Kruskal-Wallis + Mann-Whitney U")
        st.dataframe(build_kw_mw_table(cff, metrics),
                       width='stretch', hide_index=True)

        st.subheader(f"3-2. Pairwise Post-Hoc — {main_metric} (Bonferroni)")
        st.dataframe(build_posthoc_table(cff, main_metric),
                       width='stretch', hide_index=True)

        st.subheader("3-3. Chi-Square Independence Test — PFP Type x Rank")
        cff_rank = cff[cff['rank'].isin(RANK_ORDER)]
        cont = pd.crosstab(cff_rank['profile_type'], cff_rank['rank'])
        chi2, p_chi, dof, _ = stats.chi2_contingency(cont)
        n = cont.sum().sum()
        k = min(cont.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k))
        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Chi-Square", f"{chi2:.1f}")
        kpi_card(c2, "p-value", f"{p_chi:.2e}", sig_stars(p_chi))
        kpi_card(c3, "Cramer's V", f"{cramers_v:.3f}")

    # Tab 4: Cross Analysis
    with tabs[3]:
        st.subheader("4-1. PFP Type x Rank (Key Finding!)")
        cff_rank = cff[cff['rank'].isin(RANK_ORDER)]
        col1, col2 = st.columns(2)
        col1.plotly_chart(chart_binned_stacked(cff_rank, 'rank', RANK_ORDER,
                          "PFP Type Ratio by Rank (%)", "Rank"),
                          width='stretch')
        col2.plotly_chart(chart_binned_lines(cff_rank, 'rank', RANK_ORDER,
                          "PFP Type Lines by Rank — Anime share rises with rank (tapers at top tiers)", "Rank"),
                          width='stretch')

        st.subheader("4-2. Rank x PFP Type Heatmap")
        pivot = pd.crosstab(cff_rank['rank'], cff_rank['profile_type'], normalize='index') * 100
        ranks_avail = [r for r in RANK_ORDER if r in pivot.index]
        pivot = pivot.reindex(ranks_avail)
        cols_avail = [c for c in ORDER_3CAT if c in pivot.columns]
        pivot = pivot[cols_avail]
        fig = px.imshow(pivot, text_auto='.1f', aspect='auto',
                         color_continuous_scale='RdBu_r',
                         labels=dict(x="PFP Type", y="Rank", color="Ratio %"),
                         title='Rank x PFP Type Ratio Heatmap')
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')


# =====================================================================
# CROSS-PLATFORM COMPARISON
# =====================================================================
elif page == "Cross-Platform":
    st.title("GitHub x Codeforces Cross-Platform Comparison")
    st.caption("Is the same pattern consistently observed across two independent platforms?")

    if gh is None or cf is None:
        st.error("Classification or Codeforces data not found.")
        st.stop()

    st.subheader("1. PFP Type Distribution Comparison")
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                         subplot_titles=[f"GitHub (n={gh_n:,})", f"Codeforces (n={cf_n:,})"])
    for i, src in enumerate([gh, cf]):
        vc = src['profile_type'].value_counts().reindex(ORDER_3CAT).fillna(0)
        fig.add_trace(go.Pie(labels=vc.index, values=vc.values, hole=0.55,
                              marker=dict(colors=[COLORS_3CAT[t] for t in vc.index]),
                              textinfo='percent+label'), row=1, col=i+1)
    fig.update_layout(height=450)
    st.plotly_chart(fig, width='stretch')

    st.subheader("2. Anime vs Non-Anime Effect Size (Cliff's δ)")
    gh_anime = gh[gh['profile_type']=='Anime']
    gh_non = gh[gh['profile_type']!='Anime']
    cf_anime = cf[cf['profile_type']=='Anime']
    cf_non = cf[cf['profile_type']!='Anime']

    eff = []
    for m in ['followers', 'total_stars', 'public_repos', 'total_forks']:
        d, p = cliff_delta(gh_anime[m].dropna(), gh_non[m].dropna())
        eff.append({'Platform': 'GitHub', 'Metric': m, "Cliff's δ": round(d, 3),
                     'Effect': effect_label(d), 'p': f'{p:.1e}'})
    for m in ['rating', 'maxRating']:
        d, p = cliff_delta(cf_anime[m].dropna(), cf_non[m].dropna())
        eff.append({'Platform': 'Codeforces', 'Metric': m, "Cliff's δ": round(d, 3),
                     'Effect': effect_label(d), 'p': f'{p:.1e}'})
    eff_df = pd.DataFrame(eff)

    fig = px.bar(eff_df, x='Metric', y="Cliff's δ", color='Platform', barmode='group',
                  text="Cliff's δ",
                  title="Positive on both platforms → Anime PFP users consistently score higher",
                  color_discrete_map={'GitHub': '#2b7a78', 'Codeforces': '#e76f51'})
    fig.add_hline(y=0.147, line_dash='dot', annotation_text='small', line_color='gray')
    fig.add_hline(y=0.33, line_dash='dot', annotation_text='medium', line_color='gray')
    fig.add_hline(y=0.474, line_dash='dot', annotation_text='large', line_color='gray')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, width='stretch')
    st.dataframe(eff_df, width='stretch', hide_index=True)

    st.subheader("3. Top 10% Users — PFP Type Ratio")
    col1, col2 = st.columns(2)
    with col1:
        thresh = gh['followers'].quantile(0.9)
        top_gh = gh[gh['followers'] >= thresh]
        vc_gh = top_gh['profile_type'].value_counts(normalize=True).reindex(ORDER_3CAT).fillna(0) * 100
        fig = px.bar(x=vc_gh.index, y=vc_gh.values, color=vc_gh.index,
                      color_discrete_map=COLORS_3CAT,
                      labels={'x': 'PFP Type', 'y': 'Ratio (%)'},
                      title=f'GitHub Top 10% by Followers (n={len(top_gh):,})',
                      text=[f"{v:.1f}%" for v in vc_gh.values])
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, width='stretch')
    with col2:
        thresh = cf['rating'].quantile(0.9)
        top_cf = cf[cf['rating'] >= thresh]
        vc_cf = top_cf['profile_type'].value_counts(normalize=True).reindex(ORDER_3CAT).fillna(0) * 100
        fig = px.bar(x=vc_cf.index, y=vc_cf.values, color=vc_cf.index,
                      color_discrete_map=COLORS_3CAT,
                      labels={'x': 'PFP Type', 'y': 'Ratio (%)'},
                      title=f'Codeforces Top 10% by Rating (n={len(top_cf):,})',
                      text=[f"{v:.1f}%" for v in vc_cf.values])
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, width='stretch')

    st.subheader("4. Quantile PFP Ratio Trend (Both Platforms)")
    bins = 10
    rows = []
    for name, src, metric in [('GitHub', gh, 'followers'), ('Codeforces', cf, 'rating')]:
        s = src.assign(percentile=pd.qcut(src[metric].rank(method='first'),
                                            bins, labels=range(bins)))
        for ptype in ORDER_3CAT:
            pct = s.groupby('percentile', observed=True).apply(
                lambda x: (x['profile_type']==ptype).mean()*100).reset_index(name='pct')
            pct['Platform'] = name
            pct['profile_type'] = ptype
            pct['percentile'] = pct['percentile'].astype(int) * 10 + 10
            rows.append(pct)
    combined = pd.concat(rows)

    fig = px.line(combined, x='percentile', y='pct', color='profile_type',
                   line_dash='Platform', markers=True,
                   category_orders={'profile_type': ORDER_3CAT},
                   color_discrete_map=COLORS_3CAT,
                   title='PFP type ratio by quantile — solid=GitHub, dashed=Codeforces',
                   labels={'percentile': 'Percentile', 'pct': 'Ratio (%)'})
    fig.update_traces(line=dict(width=3), marker=dict(size=9))
    fig.update_layout(height=500, legend_title_text='PFP Type')
    st.plotly_chart(fig, width='stretch')

    st.subheader("5. Conclusions")
    st.markdown("""
    | Observation | GitHub | Codeforces |
    |-------------|--------|-----------|
    | Anime PFP ratio | 5.9% | 22.3% |
    | Anime vs Non-anime effect | small (δ ≈ 0.17–0.23) | small (δ ≈ 0.21) |
    | Higher quantile → Anime ratio | Increasing trend | newbie 18.4% → GM 38.4% (tapers at top tiers) |
    """)
    st.success("**Conclusion**: 'Anime PFP tends toward higher activity/skill metrics' — direction is consistent across two independent platforms, though the effect size is small (Cliff's δ ≈ 0.17–0.23).")
    st.warning("**Caveat**: Correlation ≠ Causation. Confounding variables (culture, experience, region) may exist.")



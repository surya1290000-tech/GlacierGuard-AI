"""
GlacierGuard-AI — Operational Demonstration Dashboard
======================================================
Phase 6 Streamlit dashboard presenting real research evidence from the
five-lake NRT surveillance study.

Launch:
    uv run streamlit run app/dashboard.py

Scientific integrity rules:
  - All displayed values come from verified report artifacts
  - No fabricated data, predictions, sensor observations, or GLOF alerts
  - Frozen NB09 model is loaded for informational inspection only
  - System does NOT claim 24/7 real-time monitoring or GLOF prediction
"""

import sys
from pathlib import Path

# Ensure src/ is importable when launched from the repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import pandas as pd
import numpy as np

# Plotly for interactive visualisations
import plotly.graph_objects as go
import plotly.express as px

# ── GlacierGuard-AI core imports ─────────────────────────────────────────
from glacierguard.dashboard_data import (
    load_qa_csv,
    validate_qa_schema,
    get_lake_display_name,
    get_lake_qa_record,
    compute_staleness_days,
    classify_staleness,
    get_water_mask_path,
    get_available_figures,
    LAKE_METADATA,
    CANONICAL_LAKE_ORDER,
    FROZEN_BENCHMARK,
    FIGURES_DIR,
)

# ─────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlacierGuard-AI Dashboard",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS for a clean scientific dashboard
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font and background */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Remove default top padding */
.block-container {
    padding-top: 1rem;
}

/* Metric card styling */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.metric-card h3 {
    color: #7eb8da;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}

.metric-card .value {
    color: #e8eaf6;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.2;
}

.metric-card .unit {
    color: #90a4ae;
    font-size: 0.8rem;
    font-weight: 400;
}

/* Status badge styling */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.status-NORMAL {
    background: rgba(76, 175, 80, 0.15);
    color: #66bb6a;
    border: 1px solid rgba(76, 175, 80, 0.3);
}

.status-WATCH {
    background: rgba(255, 183, 77, 0.15);
    color: #ffb74d;
    border: 1px solid rgba(255, 183, 77, 0.3);
}

.status-UNUSUAL_CHANGE {
    background: rgba(239, 83, 80, 0.15);
    color: #ef5350;
    border: 1px solid rgba(239, 83, 80, 0.3);
}

.status-INSUFFICIENT_DATA {
    background: rgba(158, 158, 158, 0.15);
    color: #bdbdbd;
    border: 1px solid rgba(158, 158, 158, 0.3);
}

/* Section dividers */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(126, 184, 218, 0.3), transparent);
    margin: 1.5rem 0;
}

/* Lake selector card */
.lake-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(126, 184, 218, 0.12);
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
}

.lake-card:hover {
    border-color: rgba(126, 184, 218, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

/* Provenance disclaimer */
.provenance-box {
    background: rgba(13, 27, 42, 0.8);
    border-radius: 10px;
    padding: 1.2rem;
    border-left: 3px solid #7eb8da;
    margin: 1rem 0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
}

/* Table styling */
.dataframe {
    font-size: 0.82rem;
}

/* Quality badge */
.quality-HIGH { color: #66bb6a; }
.quality-MEDIUM { color: #ffb74d; }
.quality-LOW { color: #ef5350; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_qa():
    """Load and validate QA data (cached for the session)."""
    try:
        df = load_qa_csv()
        warnings = validate_qa_schema(df)
        return df, warnings, None
    except (FileNotFoundError, ValueError) as exc:
        return None, [], str(exc)


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render sidebar navigation and lake selector."""
    st.sidebar.markdown("# 🏔️ GlacierGuard-AI")
    st.sidebar.markdown(
        '<span style="color: #90a4ae; font-size: 0.82rem;">'
        "Glacial Lake Monitoring &amp; ML Prediction"
        "</span>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    section = st.sidebar.radio(
        "Navigation",
        [
            "🌐 Project Overview",
            "🗺️ Five-Lake Explorer",
            "📊 ML Benchmark Results",
            "🛰️ Satellite QA Status",
            "📋 Data Provenance",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Lake selector (available in explorer views)
    selected_lake = st.sidebar.selectbox(
        "Select Lake",
        options=CANONICAL_LAKE_ORDER,
        format_func=get_lake_display_name,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="color: #546e7a; font-size: 0.72rem; text-align: center;">'
        "Phase 6 Demonstration Dashboard<br>"
        "Research evidence from verified artifacts<br>"
        "No fabricated data or real-time claims"
        "</div>",
        unsafe_allow_html=True,
    )

    return section, selected_lake


# ─────────────────────────────────────────────────────────────────────────
# Section A: Project Overview
# ─────────────────────────────────────────────────────────────────────────
def render_overview():
    """Render the project overview section."""
    st.markdown("## 🏔️ GlacierGuard-AI Project Overview")
    st.markdown(
        '<p style="color: #90a4ae; font-size: 0.95rem; margin-top: -0.5rem;">'
        "Satellite-based glacial lake monitoring and annual area-change prediction"
        "</p>",
        unsafe_allow_html=True,
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<h3>Monitored Lakes</h3>'
            '<div class="value">5</div>'
            '<div class="unit">priority Himalayan lakes</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<h3>Sensor Sources</h3>'
            '<div class="value">2</div>'
            '<div class="unit">Sentinel-1 SAR + Sentinel-2 Optical</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<h3>Best Model MAE</h3>'
            f'<div class="value">{FROZEN_BENCHMARK["mae_km2"]:.6f}</div>'
            '<div class="unit">km² (frozen 2023 test set)</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            '<div class="metric-card">'
            '<h3>System Tiers</h3>'
            '<div class="value">2</div>'
            '<div class="unit">ML Prediction + NRT Surveillance</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Two-column layout: system description
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Dual-Tier Architecture")
        st.markdown("""
**Tier 1 — Annual Area-Change Prediction (ML)**
- Uses historical GLO v1.02 inventories (2017–2024)
- Features: glacier proximity (RGI), climate drivers (ERA5-Land, CHIRPS)
- Target: next-year area change (AREA_{t+1} − AREA_t)
- Validated on frozen 2023 test set

**Tier 2 — Near-Real-Time Multi-Sensor Surveillance**
- Sentinel-1 (SAR) + Sentinel-2 (Optical) imagery
- Automated water-mask extraction and cross-sensor QA
- Conservative status tiers: NORMAL → WATCH → UNUSUAL_CHANGE
        """)

    with col_right:
        st.markdown("### Scientific Scope")
        st.warning(
            "**This system does NOT claim:**\n"
            "- 24/7 continuous live sensing\n"
            "- Instantaneous GLOF detection or prediction\n"
            "- Automated flood alarms\n\n"
            "Lake area changes serve as **decision-support indicators** "
            "for glaciological review, not automated warnings.",
            icon="⚠️",
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Lake map overview
    st.markdown("### Priority Lake Locations")
    map_data = pd.DataFrame([
        {
            "lat": meta["lat"],
            "lon": meta["lon"],
            "name": meta["display_name"],
        }
        for meta in LAKE_METADATA.values()
    ])

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=map_data["lat"],
        lon=map_data["lon"],
        mode="markers+text",
        marker=dict(size=14, color="#7eb8da", opacity=0.9),
        text=[LAKE_METADATA[lid]["display_name"] for lid in CANONICAL_LAKE_ORDER],
        textposition="top right",
        textfont=dict(size=11, color="#e8eaf6"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Lat: %{lat:.4f}°<br>"
            "Lon: %{lon:.4f}°<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=28.0, lon=86.2),
            zoom=6.5,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width='stretch')


# ─────────────────────────────────────────────────────────────────────────
# Section B: Five-Lake Explorer
# ─────────────────────────────────────────────────────────────────────────
def render_lake_explorer(df: pd.DataFrame, selected_lake: str):
    """Render the five-lake monitoring explorer."""
    st.markdown("## 🗺️ Five-Lake Monitoring Explorer")

    record = get_lake_qa_record(df, selected_lake)
    meta = LAKE_METADATA.get(selected_lake, {})
    display_name = meta.get("display_name", selected_lake)

    # Header with status badge
    status = record.get("monitoring_status", "UNKNOWN")
    quality = record.get("data_quality", "UNKNOWN")
    st.markdown(
        f'### {display_name} '
        f'<span class="status-badge status-{status}">{status}</span> '
        f'<span style="margin-left: 8px; color: #90a4ae; font-size: 0.8rem;">'
        f"Data Quality: "
        f'<span class="quality-{quality}">{quality}</span>'
        f"</span>",
        unsafe_allow_html=True,
    )

    # Lake info
    st.markdown(
        f'<div style="color: #90a4ae; font-size: 0.85rem; margin-bottom: 1rem;">'
        f'{meta.get("region", "")} · '
        f'{meta.get("dam_type", "")} · '
        f'{meta.get("elevation_m", "?")}m ASL · '
        f'{meta.get("glo_id", "")}'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    glo_area = record.get("glo_2023_vector_ref_km2", np.nan)
    primary_area = record.get("primary_monitoring_area_km2", np.nan)
    diff_pct = record.get("diff_from_glo_2023_pct", np.nan)
    disagreement = record.get("cross_sensor_disagreement_pct", np.nan)

    with col1:
        st.metric(
            "GLO 2023 Baseline",
            f"{glo_area:.4f} km²" if not np.isnan(glo_area) else "N/A",
        )

    with col2:
        st.metric(
            "Current Primary Area",
            f"{primary_area:.4f} km²" if not np.isnan(primary_area) else "N/A",
        )

    with col3:
        diff_str = f"{diff_pct:+.2f}%" if not np.isnan(diff_pct) else "N/A"
        st.metric(
            "Change vs. GLO 2023",
            diff_str,
            delta=diff_str if diff_str != "N/A" else None,
            delta_color="inverse",
        )

    with col4:
        dis_str = f"{disagreement:.1f}%" if not np.isnan(disagreement) else "N/A"
        agreement_status = record.get("sensor_agreement_status", "N/A")
        st.metric(
            "Sensor Disagreement",
            dis_str,
            delta=agreement_status,
            delta_color="off",
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Two-column layout: details + water mask
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Observation Details")

        s2_date = record.get("latest_s2_date", None)
        s1_date = record.get("latest_s1_date", None)
        s2_staleness = compute_staleness_days(s2_date)
        s1_staleness = compute_staleness_days(s1_date)

        details = pd.DataFrame([
            {
                "Sensor": "Sentinel-2 (Optical)",
                "Last Date": str(s2_date) if s2_date else "N/A",
                "Area (0m)": f'{record.get("s2_area_0m_km2", np.nan):.4f}',
                "Area (50m buffer)": f'{record.get("s2_area_50m_km2", np.nan):.4f}',
                "Staleness": f"{s2_staleness}d ({classify_staleness(s2_staleness)})" if s2_staleness is not None else "N/A",
            },
            {
                "Sensor": "Sentinel-1 (SAR)",
                "Last Date": str(s1_date) if s1_date else "N/A",
                "Area (0m)": f'{record.get("s1_area_0m_km2", np.nan):.4f}',
                "Area (50m buffer)": f'{record.get("s1_area_50m_km2", np.nan):.4f}',
                "Staleness": f"{s1_staleness}d ({classify_staleness(s1_staleness)})" if s1_staleness is not None else "N/A",
            },
        ])
        st.dataframe(details, hide_index=True, width='stretch')

        # Temporal matching
        pair_days = record.get("temporal_pair_days", None)
        match_flag = record.get("temporal_matching_flag", "N/A")
        st.markdown(
            f"**Temporal Match:** {match_flag} "
            f"({pair_days} days)" if pair_days is not None else
            f"**Temporal Match:** {match_flag}"
        )

    with col_right:
        st.markdown("#### Water Mask Visualisation")
        mask_path = get_water_mask_path(selected_lake)
        if mask_path and mask_path.is_file():
            st.image(
                str(mask_path),
                caption=f"{display_name} — Sentinel-derived water mask",
                width='stretch',
            )
        else:
            st.info(f"No water mask figure available for {display_name}")

    # Status rationale
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Status Rationale")
    st.markdown(
        f'<div class="provenance-box">'
        f'{record.get("status_rationale", "No rationale available.")}'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Known limitations
    limitations = record.get("extraction_limitations", "")
    if limitations and str(limitations) != "nan":
        st.markdown("#### Extraction Limitations")
        st.markdown(f"_{limitations}_")

    # Known issues from metadata
    issues = meta.get("known_issues", [])
    if issues:
        with st.expander("⚡ Known Issues & Caveats"):
            for issue in issues:
                st.markdown(f"- {issue}")


# ─────────────────────────────────────────────────────────────────────────
# Section C: ML Benchmark Results
# ─────────────────────────────────────────────────────────────────────────
def render_ml_benchmarks():
    """Render the ML benchmark results section."""
    st.markdown("## 📊 ML Benchmark Results")
    st.markdown(
        '<p style="color: #90a4ae; font-size: 0.9rem; margin-top: -0.5rem;">'
        "Frozen model performance from NB09 validation on held-out 2023 test set"
        "</p>",
        unsafe_allow_html=True,
    )

    # Benchmark metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<h3>Test MAE</h3>'
            f'<div class="value">{FROZEN_BENCHMARK["mae_km2"]:.6f}</div>'
            '<div class="unit">km²</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<h3>Test RMSE</h3>'
            f'<div class="value">{FROZEN_BENCHMARK["rmse_km2"]:.6f}</div>'
            '<div class="unit">km²</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<h3>Test R²</h3>'
            f'<div class="value">{FROZEN_BENCHMARK["r2"]:.4f}</div>'
            '<div class="unit">coefficient of determination</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            '<div class="metric-card">'
            '<h3>Model</h3>'
            '<div class="value" style="font-size: 1.1rem;">Fused XGBoost</div>'
            '<div class="unit">S1 SAR + S2 Optical dual-sensor</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Model details
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Training & Evaluation Protocol")
        st.markdown(f"""
| Property | Value |
|:---------|:------|
| **Training Period** | {FROZEN_BENCHMARK['training_period']} |
| **Validation Period** | {FROZEN_BENCHMARK['validation_period']} |
| **Test Period** | {FROZEN_BENCHMARK['test_period']} |
| **Target Variable** | NEXT_AREA_CHANGE (km²) |
| **Temporal Split** | Chronological (no leakage) |
| **Feature Sources** | S1 SAR, S2 Optical, RGI 6.0/7.0, ERA5-Land, CHIRPS |
        """)

    with col_right:
        st.markdown("### Interpretation Note")
        st.info(
            f"**R² = {FROZEN_BENCHMARK['r2']:.4f}**\n\n"
            "This reflects the inherent difficulty of predicting year-to-year "
            "glacial lake area changes at sub-0.01 km² scale. Most lakes are "
            "stable; predicting small deviations from zero change is hard.\n\n"
            "**MAE of 0.0048 km²** is scientifically useful for identifying "
            "lakes with anomalous expansion rates.",
            icon="📌",
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Frozen model validation plots
    st.markdown("### NB09 Validation Diagnostics")
    diagnostic_figs = [
        ("09_predicted_vs_actual.png", "Predicted vs Actual"),
        ("09_residuals.png", "Residual Distribution"),
        ("09_shap_summary.png", "SHAP Feature Importance"),
        ("09_bootstrap_mae_ci.png", "Bootstrap MAE CI"),
    ]

    available_cols = st.columns(len(diagnostic_figs))
    for i, (fig_name, caption) in enumerate(diagnostic_figs):
        fig_path = FIGURES_DIR / fig_name
        with available_cols[i]:
            if fig_path.is_file():
                st.image(str(fig_path), caption=caption, width='stretch')
            else:
                st.caption(f"_{caption}: not available_")


# ─────────────────────────────────────────────────────────────────────────
# Section D: Satellite QA Status
# ─────────────────────────────────────────────────────────────────────────
def render_satellite_qa(df: pd.DataFrame):
    """Render the satellite QA and operational status overview."""
    st.markdown("## 🛰️ Satellite QA & Operational Status")

    # Summary table for all five lakes
    summary_rows = []
    for lake_id in CANONICAL_LAKE_ORDER:
        if lake_id not in df.index:
            continue
        rec = get_lake_qa_record(df, lake_id)
        display_name = get_lake_display_name(lake_id)
        s2_stale = compute_staleness_days(rec.get("latest_s2_date"))

        summary_rows.append({
            "Lake": display_name,
            "Status": rec.get("monitoring_status", "?"),
            "Quality": rec.get("data_quality", "?"),
            "GLO 2023 (km²)": f'{rec.get("glo_2023_vector_ref_km2", np.nan):.4f}',
            "Current (km²)": f'{rec.get("primary_monitoring_area_km2", np.nan):.4f}',
            "Δ GLO (%)": f'{rec.get("diff_from_glo_2023_pct", np.nan):+.1f}',
            "Temporal Match": rec.get("temporal_matching_flag", "?"),
            "S2 Staleness": f"{s2_stale}d" if s2_stale is not None else "?",
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df,
        hide_index=True,
        width='stretch',
        column_config={
            "Status": st.column_config.TextColumn(
                "Status", help="Conservative operational status tier"
            ),
        },
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Comparative bar chart: GLO 2023 baseline vs current area
    st.markdown("### Area Comparison: GLO 2023 Baseline vs Current")
    chart_data = []
    for lake_id in CANONICAL_LAKE_ORDER:
        if lake_id not in df.index:
            continue
        rec = get_lake_qa_record(df, lake_id)
        name = get_lake_display_name(lake_id)
        glo = rec.get("glo_2023_vector_ref_km2", np.nan)
        current = rec.get("primary_monitoring_area_km2", np.nan)
        if not np.isnan(glo):
            chart_data.append({"Lake": name, "Area (km²)": glo, "Source": "GLO 2023 Baseline"})
        if not np.isnan(current):
            chart_data.append({"Lake": name, "Area (km²)": current, "Source": "Current Observation"})

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        fig = px.bar(
            chart_df,
            x="Lake",
            y="Area (km²)",
            color="Source",
            barmode="group",
            color_discrete_map={
                "GLO 2023 Baseline": "#546e7a",
                "Current Observation": "#7eb8da",
            },
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf6",
            height=380,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Status distribution
    st.markdown("### Status Distribution")
    col1, col2 = st.columns([1, 2])

    with col1:
        status_counts = df["monitoring_status"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=status_counts.index.tolist(),
            values=status_counts.values.tolist(),
            hole=0.5,
            marker=dict(colors=[
                "#66bb6a" if s == "NORMAL" else
                "#ffb74d" if s == "WATCH" else
                "#ef5350" if s == "UNUSUAL_CHANGE" else
                "#bdbdbd"
                for s in status_counts.index
            ]),
            textinfo="label+value",
            textfont=dict(size=12),
        )])
        fig_pie.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf6",
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col2:
        st.markdown("""
**Status Tier Definitions:**

| Tier | Meaning |
|:-----|:--------|
| 🟢 **NORMAL** | Lake extent matches baseline within expected variance |
| 🟡 **WATCH** | Single-sensor reliance, stale data, or moderate disagreement |
| 🔴 **UNUSUAL_CHANGE** | Both sensors confirm significant deviation from baseline |
| ⚪ **INSUFFICIENT_DATA** | Not enough valid observations for assessment |
        """)

    # Five-lake extent QA composite figure
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    composite_fig = FIGURES_DIR / "10_2_five_lake_extent_qa.png"
    if composite_fig.is_file():
        st.markdown("### Five-Lake Extent QA Composite")
        st.image(
            str(composite_fig),
            caption="Composite QA overview of all five monitored lakes",
            width='stretch',
        )


# ─────────────────────────────────────────────────────────────────────────
# Section E: Data Provenance & Limitations
# ─────────────────────────────────────────────────────────────────────────
def render_provenance():
    """Render the data provenance and limitations section."""
    st.markdown("## 📋 Data Provenance & Limitations")
    st.markdown(
        '<p style="color: #90a4ae; font-size: 0.9rem; margin-top: -0.5rem;">'
        "Transparency about data sources, known limitations, and scientific caveats"
        "</p>",
        unsafe_allow_html=True,
    )

    with st.expander("📂 Data Sources", expanded=True):
        st.markdown("""
| Source | Description | Period |
|:-------|:------------|:-------|
| **GLO v1.02** | Global Lakes & Oceans glacial lake inventory (S1 SAR & S2 Optical) | 2017–2024 |
| **RGI 6.0/7.0** | Randolph Glacier Inventory — glacier proximity features | Static |
| **ERA5-Land** | ECMWF reanalysis — temperature and snowfall drivers | 2017–2024 |
| **CHIRPS** | Precipitation satellite estimates | 2017–2024 |
| **Sentinel-1 GRD** | C-band SAR backscatter (IW mode, VV polarisation) | Near-real-time |
| **Sentinel-2 L2A** | Multi-spectral optical (MNDWI water index, SCL cloud mask) | Near-real-time |
        """)

    with st.expander("⚠️ Known Limitations"):
        st.markdown("""
**ML Prediction Tier:**
- R² of 0.34 reflects inherent difficulty of sub-0.01 km² annual change prediction
- Model trained/validated on 2017–2022 GLO data; 2023 held-out test only
- Predicts area *change*, not absolute area or GLOF probability
- No GLOF prediction capability — area change is a decision-support indicator only

**NRT Surveillance Tier:**
- **Thulagi:** 135-day gap between S2 (May) and S1 (Sep) due to monsoon cloud cover
- **Tsho Rolpa:** SAR extraction unreliable due to steep gorge terrain layover/shadow
- **Lower Barun:** 50m search buffer underestimates calving bay expansion
- **Sabai Tsho:** Glacial flour sediment plumes affect water reflectance
- **General:** Monsoon season (Jun–Sep) severely limits optical coverage across all sites

**System-Wide:**
- Not a 24/7 real-time monitoring system
- No automated GLOF detection or early warning capability
- Observation frequency depends on satellite revisit schedules (6–12 day repeat cycles)
- All status assignments are conservative and require expert glaciological review
        """)

    with st.expander("🔬 Model Artifact Integrity"):
        st.markdown("""
| Artifact | Status |
|:---------|:-------|
| `reports/models/09_final_fused_xgboost.pkl` | **Frozen** — never retrained |
| Expected SHA-256 | `5c5bec942c316a8c4a3f40d7098f7d50a4d7ac34d0f4ba5e308c7d19fc75525f` |
| Training data | 2017–2021 (chronological split, no temporal leakage) |
| Validation | 2022 for hyperparameter selection |
| Test | 2023 held-out (never seen during training or tuning) |
        """)

    with st.expander("📝 Methodology Notes"):
        st.markdown("""
**Water Extraction:**
- Sentinel-2: MNDWI > 0.0 with SCL cloud/shadow masking
- Sentinel-1: Lee 5×5 speckle filter, calibrated threshold at −12 dB
- Spatial envelope: GLO geometry prior + configurable search buffer (0m, 50m)

**Cross-Sensor QA Rules:**
- SAME_DAY: S1 and S2 acquired within 0 days
- CLOSE_PAIR: 1–3 days apart
- NEAR_PAIR: 4–10 days apart
- TEMPORALLY_UNMATCHED: >10 days — cross-sensor agreement not computed

**Status Assignment:**
- Conservative rules codified in `src/glacierguard/satellite/qa.py`
- Status escalation: NORMAL → WATCH → UNUSUAL_CHANGE
- INSUFFICIENT_DATA assigned when observations are inadequate
        """)


# ─────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────
def main():
    """Main application entry point."""
    section, selected_lake = render_sidebar()

    # Load data
    df, warnings, error = cached_load_qa()

    if error:
        st.error(
            f"**Failed to load QA data:** {error}\n\n"
            "Ensure the repository is intact and "
            "`reports/10_2_five_lake_monitoring_qa.csv` exists.",
            icon="🚨",
        )
        return

    if warnings:
        for w in warnings:
            st.warning(w, icon="⚠️")

    # Route to selected section
    if section == "🌐 Project Overview":
        render_overview()
    elif section == "🗺️ Five-Lake Explorer":
        render_lake_explorer(df, selected_lake)
    elif section == "📊 ML Benchmark Results":
        render_ml_benchmarks()
    elif section == "🛰️ Satellite QA Status":
        render_satellite_qa(df)
    elif section == "📋 Data Provenance":
        render_provenance()


if __name__ == "__main__":
    main()

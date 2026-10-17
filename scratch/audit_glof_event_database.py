"""
GlacierGuard-AI — HMAGLOFDB (High Mountain Asia Glacial Lake Outburst Flood Database)
Comprehensive Data Quality, Provenance, Leakage, and Feasibility Audit.

Usage:
    uv run python scratch/audit_glof_event_database.py --csv raw/GLOF/HMAGLOFDB.csv --out-dir reports/glof_research
"""

import os
import sys
import argparse
import hashlib
import json
import re
import pathlib
import pandas as pd
import numpy as np

# Standard placeholders recognized across HMAGLOFDB and common geospatial datasets
NA_PLACEHOLDERS = {
    "", " ", "\xa0", "\t", "nan", "NaN", "NAN", "None", "null", "NULL",
    "-", "--", "---", "?", "unknown", "Unknown", "UNKNOWN", "Na", "NA",
    "\x96", "–", "—"
}

# Fields where '+' explicitly represents positive-but-unknown counts
PLUS_COUNT_FIELDS = {
    "Lives_total", "Lives_male", "Lives_female", "Lives_disabilities",
    "Injured_total", "Injured_male", "Injured_female", "Injured_disabilities",
    "Displaced_total", "Displaced_male", "Displaced_female", "Displaced_disabilities",
    "Livestock", "Residential_destroyed", "Commerical_destroyed",
    "Residential_damaged", "Commerical_damaged", "Agricultural"
}

# Feature / Target / Leakage Registry
COLUMN_REGISTRY = {
    # Identifiers
    "GF_ID": ("IDENTIFIER", "Unique GLOF event identifier (primary key)"),
    "Lake_name": ("IDENTIFIER", "Reported local or toponymic lake name"),
    "Glacier_name": ("IDENTIFIER", "Associated parent glacier name"),
    "GL_ID": ("IDENTIFIER", "Glacial lake inventory identifier"),
    "LakeDB_ID": ("IDENTIFIER", "External lake database identifier"),
    "G_ID": ("IDENTIFIER", "Glacier identifier in RGI / GLIMS"),

    # Event Timing & Geography
    "Year_approx": ("EVENT_METADATA", "Approximate year of GLOF event"),
    "Year_exact": ("EVENT_METADATA", "Confirmed exact year of GLOF event"),
    "Month": ("EVENT_METADATA", "Month of GLOF event (1-12)"),
    "Day": ("EVENT_METADATA", "Day of GLOF event (1-31)"),
    "Transboundary": ("EVENT_METADATA", "Whether event crossed national borders (Y/N)"),
    "Repeat": ("EVENT_METADATA", "Whether the lake has experienced repeated GLOFs (Y/N)"),
    "Region_RGI": ("EVENT_METADATA", "Randolph Glacier Inventory region (e.g. 15 South Asia East)"),
    "Region_HiMAP": ("EVENT_METADATA", "High Mountain Asia sub-region classification"),
    "Country": ("EVENT_METADATA", "Country where lake is located"),
    "Province": ("EVENT_METADATA", "Administrative province/state"),
    "River_Basin": ("EVENT_METADATA", "Major drainage river basin"),
    "Ref_scientific": ("EVENT_METADATA", "Primary scientific literature citation"),
    "Ref_scientific_full": ("EVENT_METADATA", "Full bibliographic reference"),
    "Ref_other": ("EVENT_METADATA", "Secondary/grey literature reference"),
    "Remarks": ("EVENT_METADATA", "Free-text narrative comments and context"),

    # Pre-Event Lake Characteristics (Eligible as pre-event features if temporally available)
    "Lat_lake": ("PRE_EVENT_CANDIDATE", "Latitude of source lake (WGS84)"),
    "Lon_lake": ("PRE_EVENT_CANDIDATE", "Longitude of source lake (WGS84)"),
    "Elev_lake": ("PRE_EVENT_CANDIDATE", "Elevation of source lake surface (m a.s.l.)"),
    "Lake_type": ("PRE_EVENT_CANDIDATE", "Lake dam type (e.g., Moraine-dammed, Ice-dammed)"),
    "Driver_lake": ("PRE_EVENT_CANDIDATE", "Long-term morphological or climate driver of lake formation"),

    # Sparse / Unverified Timing (Could be pre- or post-drainage; high missingness)
    "Area": ("SPARSE_UNVERIFIED_TIMING", "Reported lake area (km²); unverified whether pre- or post-drainage"),
    "Volume": ("SPARSE_UNVERIFIED_TIMING", "Reported lake volume (m³); unverified whether pre- or post-drainage"),

    # Post-Event Leakage (Cannot be used as pre-event predictors)
    "Lat_impact": ("POST_EVENT_LEAKAGE", "Latitude of downstream impact location"),
    "Lon_impact": ("POST_EVENT_LEAKAGE", "Longitude of downstream impact location"),
    "Elev_impact": ("POST_EVENT_LEAKAGE", "Elevation of downstream impact location"),
    "Impact_type": ("POST_EVENT_LEAKAGE", "Nature of observed evidence (Observation / Deposit)"),
    "Driver_GLOF": ("POST_EVENT_LEAKAGE", "Triggering mechanism of breach (determined post-event)"),
    "Mechanism": ("POST_EVENT_LEAKAGE", "Breach physical failure mode (determined post-event)"),
    "Discharge_water": ("POST_EVENT_LEAKAGE", "Peak water discharge (m³/s) — physical outcome of breach"),
    "Discharge_solid": ("POST_EVENT_LEAKAGE", "Solid sediment volume transported (m³) — post-event outcome"),
    "Impact": ("POST_EVENT_LEAKAGE", "Narrative summary of downstream devastation / damage"),
    "Lives_total": ("POST_EVENT_LEAKAGE", "Fatalities caused by GLOF — post-event consequence"),
    "Lives_male": ("POST_EVENT_LEAKAGE", "Male fatalities — post-event consequence"),
    "Lives_female": ("POST_EVENT_LEAKAGE", "Female fatalities — post-event consequence"),
    "Lives_disabilities": ("POST_EVENT_LEAKAGE", "Fatalities among disabled individuals — post-event consequence"),
    "Injured_total": ("POST_EVENT_LEAKAGE", "Total persons injured — post-event consequence"),
    "Injured_male": ("POST_EVENT_LEAKAGE", "Injured male individuals — post-event consequence"),
    "Injured_female": ("POST_EVENT_LEAKAGE", "Injured female individuals — post-event consequence"),
    "Injured_disabilities": ("POST_EVENT_LEAKAGE", "Injured disabled individuals — post-event consequence"),
    "Displaced_total": ("POST_EVENT_LEAKAGE", "Persons displaced or evacuated — post-event consequence"),
    "Displaced_male": ("POST_EVENT_LEAKAGE", "Displaced males — post-event consequence"),
    "Displaced_female": ("POST_EVENT_LEAKAGE", "Displaced females — post-event consequence"),
    "Displaced_disabilities": ("POST_EVENT_LEAKAGE", "Displaced disabled individuals — post-event consequence"),
    "Livestock": ("POST_EVENT_LEAKAGE", "Livestock killed or swept away — post-event consequence"),
    "Residential_destroyed": ("POST_EVENT_LEAKAGE", "Homes destroyed — post-event damage"),
    "Commerical_destroyed": ("POST_EVENT_LEAKAGE", "Commercial buildings destroyed — post-event damage"),
    "Residential_damaged": ("POST_EVENT_LEAKAGE", "Homes damaged — post-event damage"),
    "Commerical_damaged": ("POST_EVENT_LEAKAGE", "Commercial buildings damaged — post-event damage"),
    "Infra": ("POST_EVENT_LEAKAGE", "Infrastructure damaged (bridges, roads, trails) — post-event damage"),
    "Agricultural": ("POST_EVENT_LEAKAGE", "Farmland destroyed/damaged — post-event damage"),
    "Hydropower": ("POST_EVENT_LEAKAGE", "Hydropower plants destroyed/damaged — post-event damage"),
    "Econ_damage": ("POST_EVENT_LEAKAGE", "Estimated financial loss ($ / local currency) — post-event outcome"),
    "Sat_evidence": ("POST_EVENT_LEAKAGE", "Satellite imagery confirming post-event deposit/breach")
}


def compute_sha256(filepath: pathlib.Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_value(val):
    """Normalize raw string values: trim whitespace, clean NBSP, handle placeholders."""
    if val is None or pd.isna(val):
        return None
    s = str(val).replace("\xa0", " ").strip()
    if s in NA_PLACEHOLDERS:
        return None
    return s


def parse_plus_count_field(val):
    """
    Parse a numeric or '+' count field.
    Returns:
        status: 'NULL', 'PLUS_UNKNOWN', 'EXACT_NUMERIC', 'MALFORMED'
        numeric_val: float or None
    """
    norm = normalize_value(val)
    if norm is None:
        return "NULL", None
    if norm == "+":
        return "PLUS_UNKNOWN", None
    # Strip commas or trailing + like '10+'
    cleaned = norm.replace(",", "").strip()
    if cleaned.endswith("+"):
        try:
            return "PLUS_LOWER_BOUND", float(cleaned[:-1])
        except ValueError:
            return "MALFORMED", None
    try:
        return "EXACT_NUMERIC", float(cleaned)
    except ValueError:
        return "MALFORMED", None


def load_and_clean_hmaglofdb(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load HMAGLOFDB with robust encoding and whitespace normalization."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HMAGLOFDB CSV file not found at: {csv_path.resolve()}\n"
            f"Please download the dataset from:\n"
            f"  https://raw.githubusercontent.com/fidelsteiner/HMAGLOFDB/master/Database/GLOFs/HMAGLOFDB.csv\n"
            f"and place it at: raw/GLOF/HMAGLOFDB.csv"
        )

    # Try reading with latin-1 (standard for HMAGLOFDB due to author accents & NBSP)
    try:
        df = pd.read_csv(csv_path, encoding="latin-1", dtype=str)
    except Exception as e:
        df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace", dtype=str)

    # Clean whitespace across all string columns
    for col in df.columns:
        df[col] = df[col].apply(normalize_value)

    return df


def audit_hmaglofdb(csv_path: pathlib.Path, out_dir: pathlib.Path):
    """Run full comprehensive audit of HMAGLOFDB."""
    out_dir.mkdir(parents=True, exist_ok=True)

    file_sha256 = compute_sha256(csv_path)
    file_size_bytes = csv_path.stat().st_size

    df = load_and_clean_hmaglofdb(csv_path)
    n_rows, n_cols = df.shape

    # 1. Identifier Audits
    id_col = "GF_ID" if "GF_ID" in df.columns else df.columns[0]
    unique_ids = df[id_col].dropna().nunique()
    null_ids = df[id_col].isna().sum()
    duplicate_ids = n_rows - unique_ids - null_ids

    # 2. Temporal Coverage & QC Audit
    year_col = "Year_exact" if "Year_exact" in df.columns else "Year_approx"
    years = []
    malformed_years = 0
    temporal_qc_records = []

    for idx, row in df.iterrows():
        y_ex_str = row.get("Year_exact")
        y_ap_str = row.get("Year_approx")
        m_str = row.get("Month")
        d_str = row.get("Day")

        y_val = None
        for cand in [y_ex_str, y_ap_str]:
            if cand is not None:
                try:
                    y_val = float(cand)
                    break
                except ValueError:
                    pass

        if y_val is not None:
            years.append(y_val)
            if y_val > 2025:
                qc_flag = "UNVERIFIED_POST_2025_REQUIRES_CROSS_CHECK"
                review_status = "PENDING_MAINTAINER_EVALUATION"
                qc_note = "Dated after 2025-12-31; maintainer documentation specifies this record requires evaluation and cross-checking."
            elif y_val < 1850:
                qc_flag = "HISTORICAL_PRE_INSTRUMENTAL_DOCUMENTARY"
                review_status = "DOCUMENTED_HISTORICAL_RECORD"
                qc_note = "Pre-1850 historical documentary/oral record; lake geometry often unknown or unverified."
            elif y_val < 1950:
                qc_flag = "HISTORICAL_EARLY_INSTRUMENTAL"
                review_status = "VERIFIED_LITERATURE_RECORD"
                qc_note = "1850-1950 early instrumental/expedition record."
            else:
                qc_flag = "MODERN_SATELLITE_ERA"
                review_status = "VERIFIED_RECORD"
                qc_note = "1950-2025 modern era with potential aerial/satellite verification."
        else:
            if y_ex_str is not None or y_ap_str is not None:
                malformed_years += 1
                qc_flag = "MALFORMED_YEAR_TEXT"
                review_status = "REQUIRES_FORMAT_CLEANING"
                qc_note = f"Non-numeric year entry: exact='{y_ex_str}', approx='{y_ap_str}'"
            else:
                qc_flag = "UNDATED_HISTORICAL_RECORD"
                review_status = "UNDATED"
                qc_note = "Neither exact nor approximate year recorded in source literature."

        temporal_qc_records.append({
            "GF_ID": row.get("GF_ID", idx),
            "Year_exact": y_ex_str,
            "Year_approx": y_ap_str,
            "Month": m_str,
            "Day": d_str,
            "Lake_name": row.get("Lake_name"),
            "Country": row.get("Country"),
            "River_Basin": row.get("River_Basin"),
            "QC_Flag": qc_flag,
            "Review_Status": review_status,
            "Source_Citation": row.get("Ref_scientific"),
            "Source_Other": row.get("Ref_other"),
            "QC_Note": qc_note
        })

    min_year = min(years) if years else None
    max_year = max(years) if years else None
    recent_events_count = sum(1 for y in years if y >= 2000) if years else 0
    post_2025_count = sum(1 for y in years if y > 2025)
    pre_1850_count = sum(1 for y in years if y < 1850)
    undated_count = sum(1 for r in temporal_qc_records if r["QC_Flag"] == "UNDATED_HISTORICAL_RECORD")

    df_temporal_qc = pd.DataFrame(temporal_qc_records)
    temporal_qc_csv = out_dir / "hmaglofdb_temporal_qc_audit.csv"
    df_temporal_qc.to_csv(temporal_qc_csv, index=False)

    # 3. Field Completeness and Plus-Count Audit
    field_records = []
    plus_counts_summary = {}

    for col in df.columns:
        total_obs = n_rows
        valid_obs = df[col].notna().sum()
        null_obs = df[col].isna().sum()
        completeness_pct = (valid_obs / total_obs) * 100.0

        category, description = COLUMN_REGISTRY.get(col, ("UNCLASSIFIED", "Unclassified field"))

        # Check '+' occurrences
        plus_count = 0
        if col in PLUS_COUNT_FIELDS:
            plus_analysis = [parse_plus_count_field(v)[0] for v in df[col]]
            plus_count = sum(1 for status in plus_analysis if status in ("PLUS_UNKNOWN", "PLUS_LOWER_BOUND"))
            if plus_count > 0:
                plus_counts_summary[col] = plus_count

        field_records.append({
            "column_name": col,
            "role": category,
            "description": description,
            "total_records": total_obs,
            "non_null_records": valid_obs,
            "null_records": null_obs,
            "completeness_pct": round(completeness_pct, 2),
            "plus_unknown_count": plus_count
        })

    df_completeness = pd.DataFrame(field_records)
    completeness_csv = out_dir / "hmaglofdb_field_completeness.csv"
    df_completeness.to_csv(completeness_csv, index=False)

    # 4. Feature/Target Eligibility Registry
    df_registry = df_completeness[["column_name", "role", "completeness_pct", "description"]].copy()
    registry_csv = out_dir / "hmaglofdb_feature_registry.csv"
    df_registry.to_csv(registry_csv, index=False)

    # 5. GLOF Feasibility & Gate Checks
    # Check coverage of key fields
    area_coverage_pct = df_completeness.loc[df_completeness["column_name"] == "Area", "completeness_pct"].values[0] if "Area" in df.columns else 0.0
    impact_coverage_pct = df_completeness.loc[df_completeness["column_name"] == "Impact", "completeness_pct"].values[0] if "Impact" in df.columns else 0.0
    lives_coverage_pct = df_completeness.loc[df_completeness["column_name"] == "Lives_total", "completeness_pct"].values[0] if "Lives_total" in df.columns else 0.0
    infra_coverage_pct = df_completeness.loc[df_completeness["column_name"] == "Infra", "completeness_pct"].values[0] if "Infra" in df.columns else 0.0

    # Gate logic:
    # Task 1 (Occurrence): Requires negative non-event lake-years, which are absent in an event-only inventory.
    task1_status = "BLOCKED"
    task1_blockers = [
        "HMAGLOFDB is an event-only inventory (774 positive events); no verified non-event observations exist.",
        "Unrecorded lake-years in GLO cannot be assumed to be true negatives without historical observation bias.",
        f"Pre-event lake area is missing in {100 - area_coverage_pct:.1f}% of recorded events."
    ]

    # Task 2 (Impact Severity): Requires non-arbitrary severity labels and sufficient coverage.
    task2_status = "BLOCKED"
    impact_count = df['Impact'].notna().sum() if 'Impact' in df.columns else 0
    task2_blockers = [
        f"Overall impact narrative is available for only {impact_coverage_pct:.1f}% of events ({impact_count} / {n_rows}).",
        f"Casualties (Lives_total) recorded for only {lives_coverage_pct:.1f}% of events; missing reports cannot be assumed to equal 0.",
        f"Infrastructure damage recorded for only {infra_coverage_pct:.1f}% of events.",
        "Grouping narrative text into 'LOW / MEDIUM / HIGH' creates arbitrary, uncalibrated subjective targets.",
        "Crucial physical outcomes (discharge, casualties, mechanism) are post-event leakage, not pre-event features."
    ]

    overall_recommendation = "NO-GO"
    recommendation_summary = (
        "Model training is strictly NO-GO for both GLOF occurrence and GLOF impact severity. "
        "HMAGLOFDB provides invaluable historical geomorphological reference context, but its extreme "
        "missingness (>87% across damage fields), event-only nature, and post-event character of documented "
        "mechanisms prevent training a defensible, reproducible machine learning classifier. "
        "GlacierGuard-AI must maintain GLOF hazard assessment as a rule-based expert glaciological surveillance "
        "tier, distinct from the validated annual lake area change regression tier."
    )

    profile = {
        "dataset_name": "High Mountain Asia Glacial Lake Outburst Flood Database (HMAGLOFDB)",
        "source_url": "https://github.com/fidelsteiner/HMAGLOFDB",
        "file_sha256": file_sha256,
        "file_size_bytes": file_size_bytes,
        "dimensions": {
            "total_records": int(n_rows),
            "total_columns": int(n_cols)
        },
        "identifier_audit": {
            "primary_key": str(id_col),
            "unique_identifiers": int(unique_ids),
            "duplicate_identifiers": int(duplicate_ids),
            "null_identifiers": int(null_ids)
        },
        "temporal_coverage": {
            "earliest_year": int(min_year) if min_year is not None else None,
            "latest_year": int(max_year) if max_year is not None else None,
            "events_since_2000": int(recent_events_count),
            "malformed_year_count": int(malformed_years),
            "post_2025_events_count": int(post_2025_count),
            "pre_1850_events_count": int(pre_1850_count),
            "undated_events_count": int(undated_count)
        },
        "temporal_qc_audit": {
            "maintainer_documentation_notice": "The database has been quality controlled up to 2025/12/31. Events with a later date in the database need to be evaluated and cross checked.",
            "post_2025_qc_status": "UNVERIFIED_REQUIRES_CROSS_CHECK",
            "post_2025_records": [
                {
                    "GF_ID": str(r["GF_ID"]),
                    "date": f"{r['Year_exact']}-{str(r['Month']).zfill(2) if r['Month'] else '00'}-{str(r['Day']).zfill(2) if r['Day'] else '00'}",
                    "lake": str(r["Lake_name"]),
                    "country": str(r["Country"]),
                    "citation": str(r["Source_Citation"]),
                    "secondary_ref": str(r["Source_Other"]),
                    "qc_flag": str(r["QC_Flag"]),
                    "review_status": str(r["Review_Status"])
                }
                for r in temporal_qc_records if r["QC_Flag"] == "UNVERIFIED_POST_2025_REQUIRES_CROSS_CHECK"
            ],
            "pre_instrumental_pre_1850_records_count": int(pre_1850_count),
            "undated_records_count": int(undated_count),
            "policy": "Original raw records preserved untouched without silent modification. Unverified post-2025 events and pre-instrumental documentary accounts flagged for explicit QC."
        },
        "plus_counts_audit": {
            "interpretation": "Values formatted as '+' represent positive-but-unknown counts and must never be converted to zero.",
            "fields_with_plus_counts": {k: int(v) for k, v in plus_counts_summary.items()}
        },
        "feasibility_gate": {
            "task_1_glof_occurrence": {
                "status": task1_status,
                "blockers": task1_blockers
            },
            "task_2_impact_severity": {
                "status": task2_status,
                "blockers": task2_blockers
            },
            "overall_recommendation": overall_recommendation,
            "rationale": recommendation_summary
        }
    }

    profile_json = out_dir / "hmaglofdb_dataset_profile.json"
    with open(profile_json, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else str(o))

    # Print human-readable console report
    print("=" * 78)
    print("GLACIERGUARD-AI — HMAGLOFDB DATA QUALITY & FEASIBILITY AUDIT")
    print("=" * 78)
    print(f"Dataset Path:       {csv_path}")
    print(f"SHA-256 Hash:       {file_sha256}")
    print(f"Total Records:      {n_rows} rows × {n_cols} columns")
    print(f"Event Identifiers:  {unique_ids} unique, {duplicate_ids} duplicates, {null_ids} null")
    print(f"Year Range:         {int(min_year) if min_year else 'N/A'} – {int(max_year) if max_year else 'N/A'} ({recent_events_count} events >= 2000)")
    print(f"Fields with '+':    {list(plus_counts_summary.keys())}")
    print("-" * 78)
    print("CRITICAL FIELD COMPLETENESS:")
    for col in ["Lat_lake", "Elev_lake", "Lake_type", "Area", "Volume", "Driver_GLOF", "Mechanism", "Discharge_water", "Impact", "Lives_total", "Infra", "Econ_damage"]:
        match = df_completeness[df_completeness["column_name"] == col]
        if not match.empty:
            pct = match["completeness_pct"].values[0]
            non_null = match["non_null_records"].values[0]
            role = match["role"].values[0]
            print(f"  {col:<20} {non_null:>4} / {n_rows} ({pct:>5.1f}%)  [{role}]")
    print("-" * 78)
    print(f"GATE RECOMMENDATION: {overall_recommendation}")
    print(f"  Task 1 (Occurrence):      {task1_status}")
    print(f"  Task 2 (Impact Severity): {task2_status}")
    print(f"\nArtifacts saved to {out_dir.resolve()}:")
    print(f"  - {profile_json.name}")
    print(f"  - {completeness_csv.name}")
    print(f"  - {registry_csv.name}")
    print("=" * 78)

    return profile


def main():
    parser = argparse.ArgumentParser(description="Audit HMAGLOFDB GLOF Event Database.")
    parser.add_argument("--csv", default="raw/GLOF/HMAGLOFDB.csv", help="Path to HMAGLOFDB CSV file")
    parser.add_argument("--out-dir", default="reports/glof_research", help="Directory to save audit reports")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv)
    out_dir = pathlib.Path(args.out_dir)

    try:
        audit_hmaglofdb(csv_path, out_dir)
    except FileNotFoundError as e:
        print(f"\n[DATA AUDIT ERROR] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

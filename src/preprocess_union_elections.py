"""
Union Election Preprocessing Pipeline
======================================
Inputs:
  - Raw NLRB SQLite DB (built into preliminary parquet by build_election_focus_dataset.py)
  - WRDS / Compustat comp.names  (requires WRDS credentials)
  - Old match reference file:
      datasets_processed/union/20220319_union_election_merge_with_gvkey.pkl

Outputs  (all under projects/union/outputs/):
  preliminary_election_focus.parquet
      39-column focused dataset, all 33,477 elections

  preliminary_election_focus_with_votes.parquet
      Same, filtered to elections with actual vote records (31,416 rows)

  preliminary_election_focus_with_votes_rc_compustat_match.parquet
      RC-with-votes elections plus hybrid fuzzy match columns (26,523 rows)

  union_election_rc_votes_matched_combined.parquet
      Final sample: new matches (score >= 80) supplemented by old match file.
      Key fields: gvkey_final, gvkey_source ('new_match' | 'old_match')

Usage:
  python src/preprocess_union_elections.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import wrds
from name_matching.name_matcher import NameMatcher
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUT_DIR = PROJECT_ROOT / "outputs"
OLD_MATCH_PATH = Path(
    "/data/disk4/workspace/datasets_processed/union"
    "/20220319_union_election_merge_with_gvkey.pkl"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

WRDS_USERNAME = "wangyouan"
SCORE_THRESHOLD = 80   # minimum fuzzy score to accept a new match


# ---------------------------------------------------------------------------
# Step 1: build focused election dataset  (runs build_election_focus_dataset.py)
# ---------------------------------------------------------------------------
def build_focus_dataset():
    script_path = SRC_DIR / "build_election_focus_dataset.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Build script not found: {script_path}")
    exec(script_path.read_text(), {"__name__": "__main__"})
    print("Focused dataset build completed.")


# ---------------------------------------------------------------------------
# Step 2: load focused dataset
# ---------------------------------------------------------------------------
def load_focus_dataset() -> pd.DataFrame:
    pq = OUT_DIR / "preliminary_election_focus.parquet"
    csv = OUT_DIR / "preliminary_election_focus.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Focused output not found. Run build_focus_dataset() first.")
    print(f"Loaded focus dataset: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 3: filter to RC elections with vote records
# ---------------------------------------------------------------------------
def filter_rc_with_votes(df: pd.DataFrame) -> pd.DataFrame:
    # keep elections that have actual vote counts
    if "total_valid_votes" in df.columns:
        mask_votes = pd.to_numeric(df["total_valid_votes"], errors="coerce").fillna(0) > 0
    else:
        vf = pd.to_numeric(df.get("votes_for_union", 0), errors="coerce").fillna(0)
        va = pd.to_numeric(df.get("votes_against_union", 0), errors="coerce").fillna(0)
        mask_votes = (vf + va) > 0

    df_votes = df.loc[mask_votes].copy()
    df_votes.to_parquet(OUT_DIR / "preliminary_election_focus_with_votes.parquet", index=False)
    df_votes.to_csv(OUT_DIR / "preliminary_election_focus_with_votes.csv", index=False)
    print(f"With-votes: {len(df_votes):,} rows  (removed {len(df) - len(df_votes):,})")

    # keep only RC elections
    rc_mask = df_votes["case_number"].astype("string").str.contains("-RC-", na=False)
    df_rc = df_votes.loc[rc_mask].copy()
    print(f"RC with votes: {len(df_rc):,} rows")
    return df_rc


# ---------------------------------------------------------------------------
# Step 4: build employer name candidates (primary + auxiliary from pipe-splits)
# ---------------------------------------------------------------------------
def build_employer_candidates(df_rc: pd.DataFrame) -> pd.DataFrame:
    name_cols = [c for c in ["employer_name", "employer_names_raw"] if c in df_rc.columns]
    if not name_cols:
        raise RuntimeError("No employer name columns found in df_rc")

    emp_long = df_rc[["election_id", "case_number"] + name_cols].copy()
    for c in name_cols:
        emp_long[c] = emp_long[c].astype("string")

    records = []
    for _, row in emp_long.iterrows():
        raw_values = []
        for c in name_cols:
            v = row[c]
            if pd.notna(v):
                raw_values.extend([x.strip() for x in str(v).split(" | ") if x.strip()])
        seen: set = set()
        dedup = []
        for x in raw_values:
            if x not in seen:
                seen.add(x)
                dedup.append(x)
        for i, nm in enumerate(dedup):
            records.append(
                {
                    "election_id": row["election_id"],
                    "case_number": row["case_number"],
                    "employer_candidate_name": nm,
                    "candidate_rank": i + 1,
                    "is_primary_candidate": i == 0,
                }
            )

    df_cand = pd.DataFrame(records)
    print(f"Employer candidates: {len(df_cand):,} rows, {df_cand['employer_candidate_name'].nunique():,} unique names")
    return df_cand


# ---------------------------------------------------------------------------
# Step 5: load Compustat company names from WRDS
# ---------------------------------------------------------------------------
def load_compustat() -> pd.DataFrame:
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    cols_df = db.raw_sql(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'comp' AND table_name = 'names' ORDER BY ordinal_position"
    )
    avail_cols = cols_df["column_name"].tolist()

    preferred = ["gvkey", "conm", "tic", "cik", "cusip", "fic", "loc", "naics", "sic"]
    select_cols = [c for c in preferred if c in avail_cols]
    if "gvkey" not in select_cols or "conm" not in select_cols:
        raise RuntimeError(f"Required columns missing in comp.names. available={avail_cols}")

    df_comp = db.raw_sql(f"SELECT {', '.join(select_cols)} FROM comp.names WHERE conm IS NOT NULL")
    df_comp["comp_name_for_match"] = df_comp["conm"].astype("string")
    df_comp = df_comp.dropna(subset=["comp_name_for_match"]).drop_duplicates(
        subset=["gvkey", "comp_name_for_match"]
    ).copy()
    print(f"Compustat loaded: {len(df_comp):,} rows")
    return df_comp


# ---------------------------------------------------------------------------
# Step 6: hybrid fuzzy match (name_matching + rapidfuzz, take best score)
# ---------------------------------------------------------------------------
def normalize_company_name(x) -> str:
    if x is None:
        return ""
    s = str(x).upper().replace("\n", " ").strip()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LLC|LTD|LIMITED|PLC|LP|LLP|PC)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def hybrid_fuzzy_match(df_cand: pd.DataFrame, df_comp: pd.DataFrame) -> pd.DataFrame:
    meta_cols_preferred = ["tic", "cik", "cusip", "fic", "loc", "naics", "sic"]
    meta_cols = [c for c in meta_cols_preferred if c in df_comp.columns]
    ref_cols = ["gvkey", "comp_name_for_match"] + meta_cols

    # query names
    q = df_cand[["employer_candidate_name"]].drop_duplicates().copy()
    q["employer_candidate_name"] = q["employer_candidate_name"].astype("string").str.strip()
    q = q[q["employer_candidate_name"].notna() & (q["employer_candidate_name"] != "")].reset_index(drop=True)
    q["query_for_match"] = q["employer_candidate_name"].map(normalize_company_name)
    q = q[q["query_for_match"].str.len() > 1].reset_index(drop=True)

    # reference table
    r = df_comp[ref_cols].dropna(subset=["comp_name_for_match"]).copy()
    r["comp_name_for_match"] = r["comp_name_for_match"].astype("string").str.strip()
    r = r[r["comp_name_for_match"] != ""].reset_index(drop=True)
    r = r.reset_index().rename(columns={"index": "master_row_id"})
    r["ref_for_match"] = r["comp_name_for_match"].map(normalize_company_name)
    r = r[r["ref_for_match"].str.len() > 1].reset_index(drop=True)

    print(f"Candidate names after normalization: {len(q):,}")

    # --- Track A: name_matching ---
    matcher = NameMatcher(number_of_matches=1, legal_suffixes=True, common_words=False, top_n=50, verbose=False)
    matcher.set_distance_metrics(["bag", "typo", "refined_soundex"])
    matcher.load_and_process_master_data(
        column="ref_for_match",
        df_matching_data=r[["master_row_id", "ref_for_match"]],
        transform=True,
    )
    nm_matches = matcher.match_names(to_be_matched=q[["query_for_match"]], column_matching="query_for_match")

    nm_res = q.reset_index().rename(columns={"index": "q_row_id"})
    nm_res = nm_res.merge(nm_matches.reset_index().rename(columns={"index": "q_row_id"}), on="q_row_id", how="left")

    nm_score_col = next((c for c in ["score", "match_score", "distance_score"] if c in nm_res.columns), None)
    nm_idx_col = next((c for c in ["match_index", "index_match", "master_index", "index_0"] if c in nm_res.columns), None)
    if nm_idx_col is None:
        cands = [c for c in nm_res.columns if "index" in c.lower() and c != "q_row_id"]
        nm_idx_col = cands[0] if cands else None
    if nm_score_col is None:
        cands = [c for c in nm_res.columns if "score" in c.lower() or "sim" in c.lower()]
        nm_score_col = cands[0] if cands else None
    if nm_idx_col is None:
        raise RuntimeError(f"Cannot find name_matching index column. columns={nm_res.columns.tolist()}")
    if nm_score_col is None:
        nm_res["nm_score"] = pd.NA
        nm_score_col = "nm_score"

    nm_res[nm_idx_col] = pd.to_numeric(nm_res[nm_idx_col], errors="coerce")
    nm_res[nm_score_col] = pd.to_numeric(nm_res[nm_score_col], errors="coerce")
    nm_res = nm_res.merge(
        r[["master_row_id", "gvkey", "comp_name_for_match"] + meta_cols],
        left_on=nm_idx_col, right_on="master_row_id", how="left",
    )

    nm_out = pd.DataFrame({
        "employer_candidate_name": nm_res["employer_candidate_name"],
        "nm_match_score": nm_res[nm_score_col],
        "nm_matched_gvkey": nm_res["gvkey"],
        "nm_matched_conm": nm_res["comp_name_for_match"],
    })
    for c in meta_cols:
        nm_out[f"nm_matched_{c}"] = nm_res[c]

    # --- Track B: rapidfuzz (first-char blocked) ---
    ref_by_first: dict = {}
    for _, rr in r[["master_row_id", "ref_for_match"]].iterrows():
        ref_by_first.setdefault(rr["ref_for_match"][0], []).append((rr["master_row_id"], rr["ref_for_match"]))
    all_pool = list(zip(r["master_row_id"].tolist(), r["ref_for_match"].tolist()))

    rf_records = []
    for _, row in q[["employer_candidate_name", "query_for_match"]].iterrows():
        qn = row["query_for_match"]
        pool = ref_by_first.get(qn[0]) or all_pool
        hit = process.extractOne(qn, [x[1] for x in pool], scorer=fuzz.token_sort_ratio)
        if hit is None:
            rf_records.append({"employer_candidate_name": row["employer_candidate_name"], "rf_match_score": pd.NA, "rf_master_row_id": pd.NA})
        else:
            _, sc, pos = hit
            rf_records.append({"employer_candidate_name": row["employer_candidate_name"], "rf_match_score": float(sc), "rf_master_row_id": int(pool[pos][0])})

    rf_out = pd.DataFrame(rf_records)
    rf_out = rf_out.merge(
        r[["master_row_id", "gvkey", "comp_name_for_match"] + meta_cols],
        left_on="rf_master_row_id", right_on="master_row_id", how="left",
    )
    rf_out = rf_out.rename(columns={"gvkey": "rf_matched_gvkey", "comp_name_for_match": "rf_matched_conm"})
    for c in meta_cols:
        rf_out = rf_out.rename(columns={c: f"rf_matched_{c}"})

    # --- Hybrid: best score wins ---
    hyb = nm_out.merge(rf_out, on="employer_candidate_name", how="outer")
    hyb["nm_match_score"] = pd.to_numeric(hyb["nm_match_score"], errors="coerce")
    hyb["rf_match_score"] = pd.to_numeric(hyb["rf_match_score"], errors="coerce")
    use_rf = hyb["rf_match_score"].fillna(-1) > hyb["nm_match_score"].fillna(-1)

    hyb["match_score"] = hyb["nm_match_score"]
    hyb.loc[use_rf, "match_score"] = hyb.loc[use_rf, "rf_match_score"]
    hyb["matched_gvkey"] = hyb["nm_matched_gvkey"]
    hyb["matched_conm"] = hyb["nm_matched_conm"]
    hyb.loc[use_rf, "matched_gvkey"] = hyb.loc[use_rf, "rf_matched_gvkey"]
    hyb.loc[use_rf, "matched_conm"] = hyb.loc[use_rf, "rf_matched_conm"]
    for c in meta_cols:
        out_col = f"matched_{c}"
        hyb[out_col] = hyb.get(f"nm_matched_{c}", pd.NA)
        if f"rf_matched_{c}" in hyb.columns:
            hyb.loc[use_rf, out_col] = hyb.loc[use_rf, f"rf_matched_{c}"]

    out_cols = ["employer_candidate_name", "match_score", "matched_gvkey", "matched_conm"] + [
        f"matched_{c}" for c in meta_cols if f"matched_{c}" in hyb.columns
    ]
    df_name_match = hyb[out_cols].drop_duplicates(subset=["employer_candidate_name"], keep="first").copy()
    print(f"Name-level matches built (hybrid): {len(df_name_match):,}  (rf wins: {int(use_rf.fillna(False).sum()):,})")
    return df_name_match, meta_cols


# ---------------------------------------------------------------------------
# Step 7: collapse to election-level best match, export intermediate
# ---------------------------------------------------------------------------
def collapse_to_election_level(
    df_rc: pd.DataFrame,
    df_cand: pd.DataFrame,
    df_name_match: pd.DataFrame,
    meta_cols: list,
) -> pd.DataFrame:
    df_emp_match = df_cand.merge(df_name_match, on="employer_candidate_name", how="left")
    df_best = (
        df_emp_match.sort_values(["election_id", "match_score"], ascending=[True, False])
        .drop_duplicates(subset=["election_id"], keep="first")
        .copy()
    )
    keep_cols = [
        "election_id", "case_number", "employer_candidate_name", "candidate_rank", "is_primary_candidate",
        "match_score", "matched_gvkey", "matched_conm",
    ] + [f"matched_{c}" for c in meta_cols if f"matched_{c}" in df_best.columns]
    df_best = df_best[[c for c in keep_cols if c in df_best.columns]]

    df_matched = df_rc.merge(df_best, on=["election_id", "case_number"], how="left")

    pq_out = OUT_DIR / "preliminary_election_focus_with_votes_rc_compustat_match.parquet"
    csv_out = OUT_DIR / "preliminary_election_focus_with_votes_rc_compustat_match.csv"
    df_matched.to_parquet(pq_out, index=False)
    df_matched.to_csv(csv_out, index=False)

    s = pd.to_numeric(df_matched["match_score"], errors="coerce")
    print(f"RC matched shape: {df_matched.shape}  |  score>=80: {int((s >= 80).fillna(False).sum()):,}")
    return df_matched


# ---------------------------------------------------------------------------
# Step 8: supplement with old match file, export final combined sample
# ---------------------------------------------------------------------------
def supplement_with_old_match(df_matched: pd.DataFrame) -> pd.DataFrame:
    df_old = pd.read_pickle(OLD_MATCH_PATH)
    df_old["case_number"] = df_old["casenumber"].astype("string").str.strip()
    df_old["gvkey_old"] = df_old["gvkey"].apply(
        lambda x: str(int(x)).zfill(6) if pd.notna(x) and x > 0 else pd.NA
    )
    old_lookup = (
        df_old[["case_number", "gvkey_old"]]
        .dropna(subset=["gvkey_old"])
        .drop_duplicates(subset=["case_number"], keep="first")
    )
    print(f"Old file: {len(df_old):,} rows, {old_lookup['case_number'].nunique():,} cases with gvkey")

    df_new = df_matched.copy()
    df_new["case_number"] = df_new["case_number"].astype("string").str.strip()
    df_new["match_score_num"] = pd.to_numeric(df_new["match_score"], errors="coerce")
    new_high_conf = df_new["match_score_num"].fillna(-1) >= SCORE_THRESHOLD

    df_new = df_new.merge(old_lookup, on="case_number", how="left")
    df_new["gvkey_final"] = pd.NA
    df_new["gvkey_source"] = pd.NA
    df_new.loc[new_high_conf, "gvkey_final"] = df_new.loc[new_high_conf, "matched_gvkey"]
    df_new.loc[new_high_conf, "gvkey_source"] = "new_match"
    old_mask = ~new_high_conf & df_new["gvkey_old"].notna()
    df_new.loc[old_mask, "gvkey_final"] = df_new.loc[old_mask, "gvkey_old"]
    df_new.loc[old_mask, "gvkey_source"] = "old_match"

    n_total = len(df_new)
    n_new = int((df_new["gvkey_source"] == "new_match").sum())
    n_old = int((df_new["gvkey_source"] == "old_match").sum())
    n_unmatched = int(df_new["gvkey_final"].isna().sum())
    print(f"Combined: total={n_total:,}  new_match={n_new:,}  old_match={n_old:,}  unmatched={n_unmatched:,}")
    print(f"Total with gvkey: {n_total - n_unmatched:,}  ({(n_total - n_unmatched)/n_total:.2%})")

    df_combined = df_new[df_new["gvkey_final"].notna()].copy()
    df_combined["election_year"] = pd.to_datetime(df_combined["election_date"], errors="coerce").dt.year
    cy = df_combined.dropna(subset=["gvkey_final", "election_year"]).drop_duplicates(subset=["gvkey_final", "election_year"])
    print(f"Unique companies: {df_combined['gvkey_final'].nunique():,}  |  Company-year obs: {len(cy):,}")

    pq_out = OUT_DIR / "union_election_rc_votes_matched_combined.parquet"
    csv_out = OUT_DIR / "union_election_rc_votes_matched_combined.csv"
    df_new.to_parquet(pq_out, index=False)
    df_new.to_csv(csv_out, index=False)
    print(f"Exported: {pq_out}")
    print(f"Exported: {csv_out}")
    return df_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Build focused election dataset")
    print("=" * 60)
    build_focus_dataset()

    print("\n" + "=" * 60)
    print("Step 2: Load focused dataset")
    print("=" * 60)
    df_focus = load_focus_dataset()

    print("\n" + "=" * 60)
    print("Step 3: Filter to RC elections with votes")
    print("=" * 60)
    df_rc = filter_rc_with_votes(df_focus)

    print("\n" + "=" * 60)
    print("Step 4: Build employer candidates")
    print("=" * 60)
    df_cand = build_employer_candidates(df_rc)

    print("\n" + "=" * 60)
    print("Step 5: Load Compustat from WRDS")
    print("=" * 60)
    df_comp = load_compustat()

    print("\n" + "=" * 60)
    print("Step 6: Hybrid fuzzy match")
    print("=" * 60)
    df_name_match, meta_cols = hybrid_fuzzy_match(df_cand, df_comp)

    print("\n" + "=" * 60)
    print("Step 7: Collapse to election level")
    print("=" * 60)
    df_matched = collapse_to_election_level(df_rc, df_cand, df_name_match, meta_cols)

    print("\n" + "=" * 60)
    print("Step 8: Supplement with old match file")
    print("=" * 60)
    supplement_with_old_match(df_matched)

    print("\nDone.")

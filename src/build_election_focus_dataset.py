from pathlib import Path
import sqlite3
import pandas as pd


IN_PARQUET = Path("/data/disk4/workspace/projects/union/outputs/preliminary_election_level.parquet")
IN_CSV = Path("/data/disk4/workspace/projects/union/outputs/preliminary_election_level.csv")
OUT_DIR = Path("/data/disk4/workspace/projects/union/outputs")
RAW_DB_PATH = Path("/data/disk4/workspace/datasets_raw/union/nlrb/nlrb.db")


def first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coalesce_series(df: pd.DataFrame, candidates, default=pd.NA):
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series([default] * len(df), index=df.index)

    s = df[existing[0]]
    for c in existing[1:]:
        s = s.combine_first(df[c])
    return s


def normalize_join_values(*values):
    out = []
    seen = set()
    for v in values:
        if pd.isna(v):
            continue
        for piece in str(v).split(" | "):
            item = piece.strip()
            if not item:
                continue
            if item not in seen:
                seen.add(item)
                out.append(item)
    return " | ".join(out) if out else pd.NA


def join_unique_series(s: pd.Series):
    vals = [str(v).strip() for v in pd.unique(s.dropna()) if str(v).strip()]
    return " | ".join(vals) if vals else pd.NA


def enrich_employer_fields_from_raw(df_focus: pd.DataFrame) -> pd.DataFrame:
    """Add richer employer-related fields from raw participant records by case_number."""
    if not RAW_DB_PATH.exists():
        print(f"Raw DB not found, skip employer enrichment: {RAW_DB_PATH}")
        return df_focus

    conn = sqlite3.connect(f"file:{RAW_DB_PATH}?mode=ro", uri=True)
    try:
        participant_cols = pd.read_sql("PRAGMA table_info(participant)", conn)["name"].tolist()
        needed = [
            "case_number",
            "participant",
            "type",
            "subtype",
            "address",
            "address_1",
            "address_2",
            "city",
            "state",
            "zip",
            "phone_number",
        ]
        use_cols = [c for c in needed if c in participant_cols]
        if not {"case_number", "type"}.issubset(set(use_cols)):
            print("participant table missing case_number/type, skip employer enrichment")
            return df_focus

        sql = "SELECT " + ", ".join(f"[{c}]" for c in use_cols) + " FROM participant"
        p = pd.read_sql(sql, conn)

        # Filter employer-like participant rows (same rule as notebook)
        t = p["type"].astype("string").str.lower().fillna("")
        mask = t.str.contains("employer|company|business|management", regex=True)
        p = p.loc[mask].copy()

        # Build a case-level employer summary with more than just names
        agg_map = {c: join_unique_series for c in p.columns if c != "case_number"}
        summary = p.groupby("case_number", dropna=False, as_index=False).agg(agg_map)

        rename_map = {
            "participant": "employer_names_raw",
            "type": "employer_types_raw",
            "subtype": "employer_subtypes_raw",
            "address": "employer_address_raw",
            "address_1": "employer_address_1_raw",
            "address_2": "employer_address_2_raw",
            "city": "employer_city_raw",
            "state": "employer_state_raw",
            "zip": "employer_zip_raw",
            "phone_number": "employer_phone_raw",
        }
        summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})

        out = df_focus.merge(summary, on="case_number", how="left", validate="m:1")

        # Prefer raw participant employer names if available
        out["employer_name"] = out["employer_name"].combine_first(out.get("employer_names_raw"))
        return out
    finally:
        conn.close()


def add_vote_detail_fields(df_focus: pd.DataFrame, df_wide: pd.DataFrame) -> pd.DataFrame:
    """Add explicit for/against/support-rate vote metrics from tally columns."""
    out = df_focus.copy()

    # Find no-union tally column (usually tally_votes_no_union)
    no_union_col = first_existing(df_wide, ["tally_votes_no_union", "tally_votes_no_union_"])
    tally_vote_cols = [
        c
        for c in df_wide.columns
        if c.startswith("tally_votes_") and c != "tally_votes_total_reconstructed"
    ]
    union_vote_cols = [c for c in tally_vote_cols if c != no_union_col]

    against = pd.to_numeric(df_wide[no_union_col], errors="coerce").fillna(0.0) if no_union_col else 0.0

    if union_vote_cols:
        union_votes = df_wide[union_vote_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        votes_for = union_votes.sum(axis=1)
        lead_union_col = union_votes.idxmax(axis=1)
        lead_union_votes = union_votes.max(axis=1)
        lead_union_name = lead_union_col.str.replace("tally_votes_", "", regex=False).str.replace("_", " ")
    else:
        votes_for = pd.Series([0.0] * len(df_wide), index=df_wide.index)
        lead_union_votes = pd.Series([pd.NA] * len(df_wide), index=df_wide.index)
        lead_union_name = pd.Series([pd.NA] * len(df_wide), index=df_wide.index)

    total_valid = votes_for + against
    support_rate = votes_for / total_valid.replace(0, pd.NA)

    out["votes_for_union"] = votes_for
    out["votes_against_union"] = against
    out["total_valid_votes"] = total_valid
    out["union_support_rate"] = support_rate
    out["lead_union_name"] = lead_union_name
    out["lead_union_votes"] = lead_union_votes

    return out


def load_wide_dataset() -> pd.DataFrame:
    if IN_PARQUET.exists():
        print(f"Reading parquet: {IN_PARQUET}")
        return pd.read_parquet(IN_PARQUET)
    if IN_CSV.exists():
        print(f"Reading csv: {IN_CSV}")
        return pd.read_csv(IN_CSV)
    raise FileNotFoundError("No preliminary_election_level input file found")


def build_focus_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Core election identifiers
    df_focus = pd.DataFrame(index=df.index)
    if "election_id" not in df.columns:
        raise KeyError("election_id is required in source dataset")

    df_focus["election_id"] = df["election_id"]
    df_focus["case_number"] = coalesce_series(df, ["case_number", "case_number_x", "case_number_y"])

    # Election date and status
    df_focus["election_date"] = coalesce_series(df, ["date", "election_date", "filing__date_filed"])
    df_focus["tally_type"] = coalesce_series(df, ["tally_type"])
    df_focus["ballot_type"] = coalesce_series(df, ["ballot_type"])
    df_focus["runoff_required"] = coalesce_series(df, ["runoff_required"])

    # Election outcome
    df_focus["union_to_certify"] = coalesce_series(df, ["union_to_certify"])
    df_focus["total_ballots_counted"] = coalesce_series(
        df,
        ["total_ballots_counted", "tally_votes_total_reconstructed"],
    )
    df_focus["void_ballots"] = coalesce_series(df, ["void_ballots"])
    df_focus["challenged_ballots"] = coalesce_series(df, ["challenged_ballots"])
    df_focus["challenges_are_determinative"] = coalesce_series(df, ["challenges_are_determinative"])

    # Employer information (base)
    df_focus["employer_name"] = coalesce_series(
        df,
        ["participant__employer_names", "filing__name"],
    )

    # Union participation information
    df_focus["participant_union_names"] = coalesce_series(df, ["participant__union_names"])
    df_focus["participant_types_observed"] = coalesce_series(df, ["participant__types_observed"])

    # Merge union signals into a single human-readable field
    df_focus["unions_involved"] = [
        normalize_join_values(a, b)
        for a, b in zip(df_focus["participant_union_names"], df_focus["union_to_certify"])
    ]

    # Filing context (useful for filtering / QA)
    df_focus["filing_status"] = coalesce_series(df, ["filing__status"])
    df_focus["filing_date_filed"] = coalesce_series(df, ["filing__date_filed"])
    df_focus["filing_date_closed"] = coalesce_series(df, ["filing__date_closed"])
    df_focus["filing_city"] = coalesce_series(df, ["filing__city"])
    df_focus["filing_state"] = coalesce_series(df, ["filing__state"])

    # Quality diagnostics columns
    df_focus["participant_row_count"] = coalesce_series(df, ["participant__row_count"])
    df_focus["result_row_count"] = coalesce_series(df, ["result_row_count"])
    df_focus["tally_row_count"] = coalesce_series(df, ["tally_row_count"])

    # Add explicit voting detail fields requested by user
    df_focus = add_vote_detail_fields(df_focus, df)

    # Add richer employer fields from raw participant table
    df_focus = enrich_employer_fields_from_raw(df_focus)

    # Keep one row per election_id
    dup_rows = int(df_focus.duplicated(subset=["election_id"]).sum())
    print(f"Duplicate rows in focus dataset before drop: {dup_rows}")
    if dup_rows > 0:
        df_focus = df_focus.drop_duplicates(subset=["election_id"], keep="first").copy()

    return df_focus


def export_focus(df_focus: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "preliminary_election_focus.csv"
    out_parquet = OUT_DIR / "preliminary_election_focus.parquet"

    df_focus.to_csv(out_csv, index=False)
    print(f"Exported: {out_csv}")

    try:
        df_focus.to_parquet(out_parquet, index=False)
        print(f"Exported: {out_parquet}")
    except Exception as e:
        print(f"Parquet export skipped: {e}")


def main():
    df_wide = load_wide_dataset()
    print(f"Wide shape: {df_wide.shape}")

    df_focus = build_focus_dataset(df_wide)
    print(f"Focus shape: {df_focus.shape}")
    print(f"Focus columns: {df_focus.columns.tolist()}")
    print(df_focus.head(5).to_string(index=False))

    export_focus(df_focus)


if __name__ == "__main__":
    main()

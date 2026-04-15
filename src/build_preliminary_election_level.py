from pathlib import Path
import sqlite3
import pandas as pd


DB_PATH = Path("/data/disk4/workspace/datasets_raw/union/nlrb/nlrb.db")
OUT_DIR = Path("/data/disk4/workspace/projects/union/outputs")


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    print(f"Connected to: {db_path}")
    return conn


def load_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
    print(f"{table_name:<16} shape={df.shape}")
    return df


def agg_series_preserve_values(s: pd.Series):
    non_null = s.dropna()
    if non_null.empty:
        return pd.NA
    vals = pd.unique(non_null)
    if len(vals) == 1:
        return vals[0]
    return " | ".join(map(str, vals))


def detect_vote_col(df_tally: pd.DataFrame) -> str:
    candidates = ["votes", "vote_count", "count", "ballots", "n_votes"]
    lower_map = {c.lower(): c for c in df_tally.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    num_cols = [c for c in df_tally.columns if pd.api.types.is_numeric_dtype(df_tally[c])]
    num_cols = [c for c in num_cols if c not in {"election_id", "voting_unit_id"}]
    if num_cols:
        return num_cols[0]

    raise ValueError("Could not detect a numeric vote count column in tally")


def safe_col(x: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(x)).strip("_")


def pick_case_col(df: pd.DataFrame):
    if "case_number" in df.columns:
        return "case_number"
    case_like = [c for c in df.columns if str(c).lower().startswith("case_number")]
    if not case_like:
        return None
    return min(case_like, key=lambda c: df[c].isna().sum())


def aggregate_to_key_prefixed(df: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    if key_col not in df.columns:
        raise KeyError(f"{key_col} not found in source table")

    agg_map = {c: agg_series_preserve_values for c in df.columns if c != key_col}
    out = df.groupby(key_col, dropna=False, as_index=False).agg(agg_map)
    out[f"{prefix}__row_count"] = df.groupby(key_col, dropna=False).size().values

    rename_map = {c: f"{prefix}__{c}" for c in out.columns if c != key_col}
    return out.rename(columns=rename_map)


def step_a_load_tables(conn: sqlite3.Connection):
    print("\n--- Step A: Load main tables ---")
    df_election = load_table(conn, "election")
    df_result = load_table(conn, "election_result")
    df_tally = load_table(conn, "tally")
    df_voting_unit = load_table(conn, "voting_unit")
    df_filing = load_table(conn, "filing")
    df_participant = load_table(conn, "participant")
    return df_election, df_result, df_tally, df_voting_unit, df_filing, df_participant


def step_b_base(df_election: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step B: Base from election ---")
    df_work = df_election.copy()
    if "election_id" not in df_work.columns:
        raise KeyError("election_id not found in election table")

    base_dup_rows = int(df_work.duplicated(subset=["election_id"]).sum())
    base_dup_ids = int(
        df_work.loc[df_work.duplicated(subset=["election_id"], keep=False), "election_id"].nunique()
    )
    print(f"Base shape: {df_work.shape}")
    print(f"Duplicate rows by election_id: {base_dup_rows}")
    print(f"Distinct duplicated election_id values: {base_dup_ids}")
    return df_work


def step_c_merge_result(df_work: pd.DataFrame, df_result: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step C: Merge election_result ---")
    if "election_id" not in df_result.columns:
        raise KeyError("election_id not found in election_result table")

    result_counts = df_result.groupby("election_id", dropna=False).size().rename("n_result_rows")
    result_multi = result_counts[result_counts > 1]
    print(f"election_result rows: {len(df_result):,}")
    print(f"election_id groups: {result_counts.shape[0]:,}")
    print(f"election_id with multiple result rows: {result_multi.shape[0]:,}")

    agg_map_result = {c: agg_series_preserve_values for c in df_result.columns if c != "election_id"}
    df_result_1row = df_result.groupby("election_id", dropna=False, as_index=False).agg(agg_map_result)
    df_result_1row["result_row_count"] = df_result.groupby("election_id", dropna=False).size().values

    rows_before = len(df_work)
    cols_before = df_work.shape[1]
    df_work = df_work.merge(df_result_1row, on="election_id", how="left", validate="m:1")

    print(f"Shape before merge: ({rows_before}, {cols_before})")
    print(f"Shape after merge : {df_work.shape}")
    print(f"Duplicate rows by election_id after merge: {int(df_work.duplicated('election_id').sum())}")
    return df_work


def step_d_merge_tally(df_work: pd.DataFrame, df_tally: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step D: Summarize and merge tally ---")
    required_tally_cols = {"election_id", "option"}
    missing_tally_cols = required_tally_cols - set(df_tally.columns)
    if missing_tally_cols:
        raise KeyError(f"tally missing required columns: {missing_tally_cols}")

    print("Observed tally.option values:")
    print(df_tally["option"].value_counts(dropna=False).to_string())

    vote_col = detect_vote_col(df_tally)
    print(f"Using vote count column: {vote_col}")

    t = df_tally[["election_id", "option", vote_col]].copy()
    t["option"] = t["option"].astype("string").fillna("<NA>")

    pivot = (
        t.pivot_table(
            index="election_id",
            columns="option",
            values=vote_col,
            aggfunc="sum",
            fill_value=0,
        )
        .rename_axis(columns=None)
        .reset_index()
    )

    pivot.columns = ["election_id"] + [f"tally_votes_{safe_col(c)}" for c in pivot.columns[1:]]
    vote_option_cols = [c for c in pivot.columns if c.startswith("tally_votes_")]
    pivot["tally_votes_total_reconstructed"] = pivot[vote_option_cols].sum(axis=1)

    tally_counts = df_tally.groupby("election_id", dropna=False).size().rename("tally_row_count").reset_index()
    df_tally_summary = pivot.merge(tally_counts, on="election_id", how="left")

    df_work = df_work.merge(df_tally_summary, on="election_id", how="left", validate="m:1")
    print(f"Tally summary shape: {df_tally_summary.shape}")
    print(f"Main shape after tally merge: {df_work.shape}")
    print(f"Duplicate election_id rows after tally merge: {int(df_work.duplicated('election_id').sum())}")
    return df_work


def step_e_merge_voting_unit_and_filing(
    df_work: pd.DataFrame,
    df_voting_unit: pd.DataFrame,
    df_filing: pd.DataFrame,
) -> pd.DataFrame:
    print("\n--- Step E: Merge voting_unit and filing (robust) ---")

    df_work = df_work.loc[:, ~df_work.columns.duplicated()].copy()

    if "voting_unit_id" in df_work.columns and "voting_unit_id" in df_voting_unit.columns:
        vu_summary = aggregate_to_key_prefixed(df_voting_unit, "voting_unit_id", "voting_unit")
        rows_before = len(df_work)
        df_work = df_work.merge(vu_summary, on="voting_unit_id", how="left", validate="m:1")
        print(f"After voting_unit merge: rows={len(df_work):,} (change {len(df_work)-rows_before:+d})")
        print(f"Duplicated election_id rows: {int(df_work.duplicated('election_id').sum()):,}")
    else:
        print("Skip voting_unit merge: voting_unit_id missing on one side.")

    left_case_col = pick_case_col(df_work)
    if left_case_col is not None and "case_number" in df_filing.columns:
        filing_summary = aggregate_to_key_prefixed(df_filing, "case_number", "filing")
        rows_before = len(df_work)
        df_work = df_work.merge(
            filing_summary,
            left_on=left_case_col,
            right_on="case_number",
            how="left",
            validate="m:1",
        )
        print(
            f"After filing merge: rows={len(df_work):,} (change {len(df_work)-rows_before:+d}), "
            f"left key={left_case_col}"
        )
        print(f"Duplicated election_id rows: {int(df_work.duplicated('election_id').sum()):,}")
    else:
        print("Skip filing merge: no case_number-like key in df_work or missing in filing.")

    return df_work


def step_f_merge_participant(
    df_work: pd.DataFrame,
    df_election: pd.DataFrame,
    df_participant: pd.DataFrame,
) -> pd.DataFrame:
    print("\n--- Step F: Merge participant summary (robust) ---")

    df_work = df_work.loc[:, ~df_work.columns.duplicated()].copy()

    left_case_col = pick_case_col(df_work)
    if left_case_col is None and {"election_id", "case_number"}.issubset(df_election.columns):
        bridge = (
            df_election[["election_id", "case_number"]]
            .dropna(subset=["election_id"])
            .drop_duplicates(subset=["election_id"], keep="first")
        )
        df_work = df_work.merge(bridge, on="election_id", how="left", validate="m:1")
        left_case_col = pick_case_col(df_work)

    if left_case_col is None:
        raise KeyError("No case_number-like column found in df_work")
    if "case_number" not in df_participant.columns or "type" not in df_participant.columns:
        raise KeyError("participant table must contain case_number and type")

    rows_per_case = df_participant.groupby("case_number", dropna=False).size()
    print(f"participant rows: {len(df_participant):,}")
    print(f"case_number groups: {rows_per_case.shape[0]:,}")
    print(f"case_number with >1 participant row: {int((rows_per_case > 1).sum()):,}")

    ptype_counts = df_participant["type"].astype("string").value_counts(dropna=False)
    ptype_lookup = pd.DataFrame({"type_raw": ptype_counts.index.astype("string")})
    ptype_lookup["type_norm"] = ptype_lookup["type_raw"].str.lower().fillna("")
    ptype_lookup["is_employer_like"] = ptype_lookup["type_norm"].str.contains(
        "employer|company|business|management", regex=True
    )
    ptype_lookup["is_union_like"] = ptype_lookup["type_norm"].str.contains(
        "union|labor|employee|petitioner", regex=True
    )

    employer_types = set(ptype_lookup.loc[ptype_lookup["is_employer_like"], "type_raw"].tolist())
    union_types = set(ptype_lookup.loc[ptype_lookup["is_union_like"], "type_raw"].tolist())

    name_candidates = [
        "name",
        "participant_name",
        "organization_name",
        "entity_name",
        "full_name",
        "participant",
    ]
    name_col = next((c for c in name_candidates if c in df_participant.columns), None)
    if name_col is None:
        fallback_cols = [
            c
            for c in df_participant.columns
            if c not in {"case_number", "election_id", "type", "voting_unit_id"}
        ]
        if not fallback_cols:
            raise ValueError("No suitable participant name column found")
        name_col = fallback_cols[0]

    p = df_participant.copy()
    p["type"] = p["type"].astype("string")
    p[name_col] = p[name_col].astype("string")
    p["employer_name"] = p[name_col].where(p["type"].isin(employer_types))
    p["union_name"] = p[name_col].where(p["type"].isin(union_types))

    def join_unique(s: pd.Series):
        vals = [str(v) for v in pd.unique(s.dropna()) if str(v).strip() != ""]
        return " | ".join(vals) if vals else pd.NA

    participant_summary = (
        p.groupby("case_number", dropna=False)
        .agg(
            participant__employer_names=("employer_name", join_unique),
            participant__union_names=("union_name", join_unique),
            participant__types_observed=("type", join_unique),
            participant__row_count=("type", "size"),
            participant__employer_record_count=("employer_name", lambda s: int(s.notna().sum())),
            participant__union_record_count=("union_name", lambda s: int(s.notna().sum())),
        )
        .reset_index()
        .rename(columns={"case_number": "participant_case_number"})
    )

    rows_before = len(df_work)
    df_work = df_work.merge(
        participant_summary,
        left_on=left_case_col,
        right_on="participant_case_number",
        how="left",
        validate="m:1",
    )

    print(f"After participant merge: rows={len(df_work):,} (change {len(df_work)-rows_before:+d})")
    print(f"Duplicated election_id rows: {int(df_work.duplicated('election_id').sum()):,}")
    print(f"Missing participant employer names: {int(df_work['participant__employer_names'].isna().sum()):,}")
    print(f"Missing participant union names: {int(df_work['participant__union_names'].isna().sum()):,}")
    return df_work


def step_g_finalize(df_work: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step G: Finalize election-level frame ---")
    dup_rows = int(df_work.duplicated(subset=["election_id"]).sum())
    print(f"Rows before finalization: {len(df_work):,}")
    print(f"Duplicate rows by election_id before finalization: {dup_rows:,}")

    if dup_rows > 0:
        agg_map = {c: agg_series_preserve_values for c in df_work.columns if c != "election_id"}
        df_election_level = df_work.groupby("election_id", dropna=False, as_index=False).agg(agg_map)
        df_election_level["rows_collapsed_per_election_id"] = (
            df_work.groupby("election_id", dropna=False).size().values
        )
    else:
        df_election_level = df_work.copy()

    print(f"Final shape: {df_election_level.shape}")
    print(f"Duplicate rows by election_id (final): {int(df_election_level.duplicated('election_id').sum()):,}")
    return df_election_level


def step_h_export(df_election_level: pd.DataFrame, out_dir: Path):
    print("\n--- Step H: Export ---")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "preliminary_election_level.csv"
    parquet_path = out_dir / "preliminary_election_level.parquet"

    df_election_level.to_csv(csv_path, index=False)
    print(f"CSV exported: {csv_path}")

    try:
        df_election_level.to_parquet(parquet_path, index=False)
        print(f"Parquet exported: {parquet_path}")
    except Exception as e:
        print(f"Parquet export skipped due to: {e}")


def main():
    conn = connect_db(DB_PATH)
    try:
        (
            df_election,
            df_result,
            df_tally,
            df_voting_unit,
            df_filing,
            df_participant,
        ) = step_a_load_tables(conn)

        df_work = step_b_base(df_election)
        df_work = step_c_merge_result(df_work, df_result)
        df_work = step_d_merge_tally(df_work, df_tally)
        df_work = step_e_merge_voting_unit_and_filing(df_work, df_voting_unit, df_filing)
        df_work = step_f_merge_participant(df_work, df_election, df_participant)

        df_election_level = step_g_finalize(df_work)
        step_h_export(df_election_level, OUT_DIR)

        print("\nDone.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

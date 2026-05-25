# Union Data Processing

This repository processes NLRB union-election data into election-level datasets and matched firm identifiers. It starts from the raw NLRB SQLite database, constructs election-level files, filters to election-focused observations, attaches vote information, and links employer names to Compustat `gvkey` using WRDS/Compustat names plus an older matched dataset.

## Current Status

- Git branch: `main`, tracking `origin/main`.
- Working tree was clean before this README update.
- The repository already has generated outputs for the preliminary election panel, vote-filtered election focus file, Compustat-name matching, and a `gvkey`-only election file.
- The main downstream input for the union/Glassdoor merge is `outputs/union_election_rc_votes_gvkey_only.parquet`.

## Directory Layout

- `src/`: Python scripts for building and matching union-election datasets.
- `data/`: project-local data files, if any.
- `outputs/`: generated csv/parquet artifacts.
- `logs/`: run logs.
- `notebooks/`: exploratory checks, currently including `preprocessing_union_file.ipynb`.

## Main Pipeline

Typical order:

```bash
python src/build_preliminary_election_level.py
python src/build_election_focus_dataset.py
python src/preprocess_union_elections.py
```

`preprocess_union_elections.py` orchestrates several later steps, including vote filtering, employer-name matching, election-level collapse, and combination with older matched records.

## Key Scripts

- `build_preliminary_election_level.py`: reads `/data/disk4/workspace/datasets_raw/union/nlrb/nlrb.db`, merges election/result/tally/voting-unit/filing/participant tables, and exports `preliminary_election_level.csv` and `.parquet`.
- `build_election_focus_dataset.py`: reads the preliminary election-level file, keeps election-focused records, derives vote variables and union-name fields, and exports `preliminary_election_focus.csv` and `.parquet`.
- `preprocess_union_elections.py`: filters to elections with usable votes, uses WRDS Compustat company names for employer matching, collapses candidate matches to election level, combines with the legacy matched dataset, and exports final matched files.

## Important Inputs

- Raw NLRB SQLite database: `/data/disk4/workspace/datasets_raw/union/nlrb/nlrb.db`
- Legacy matched union file: `/data/disk4/workspace/datasets_processed/union/20220319_union_election_merge_with_gvkey.pkl`
- WRDS Compustat names are loaded inside `preprocess_union_elections.py`. The current script has `WRDS_USERNAME = "wangyouan"` hardcoded (line 50); consider making this configurable via environment variable for portability and credential security.

## Important Outputs

- `outputs/preliminary_election_level.parquet`
- `outputs/preliminary_election_focus.parquet`
- `outputs/preliminary_election_focus_with_votes.parquet`
- `outputs/preliminary_election_focus_with_votes_rc_compustat_match.parquet`
- `outputs/union_election_rc_votes_matched_combined.parquet`
- `outputs/union_election_rc_votes_gvkey_only.parquet`

## Notes for AI Handoff

- Treat `outputs/` as generated artifacts unless the task explicitly asks to inspect or regenerate them.
- WRDS access may require credentials or an active configured WRDS environment.
- Be careful with absolute paths under `/data/disk4/workspace/`; several scripts depend on the current server layout.
- This repository feeds `union_glassdoor`, especially through `outputs/union_election_rc_votes_gvkey_only.parquet`.
- Keep code changes small and verify with `git status` before and after edits.

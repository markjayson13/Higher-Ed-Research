# IPEDS Mapping & Extraction Tools

Utilities to map human-readable IPEDS variable titles to database `tableName` and
`varName`, then export tidy CSVs for analysis. Supports two modes:

- JDBC / UCanAccess (Java required)
- CSV fallback (no Java; uses exported `VARTABLE##.csv`)

## Setup

Create and activate a virtual environment, then install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ipeds/requirements.txt
```

If using JDBC / UCanAccess:

1) Install a JDK (e.g., OpenJDK 11+):
```bash
brew install openjdk
```
2) Download UCanAccess (e.g., UCanAccess-5.0.1-bin.zip) and place all JARs into:
```
/Users/markjaysonfarol13/lib/ucanaccess
```
Expected files include `ucanaccess-*.jar`, `jackcess-*.jar`, `hsqldb-*.jar`,
`commons-logging-*.jar`, `commons-lang-*.jar`.

## Mapping IPEDS Variable Titles

JDBC / UCanAccess (Java required):

```bash
python3 ipeds/map_ipeds_vars.py \
  --db-dir "/Users/markjaysonfarol13/Desktop/IPEDS Panel Dataset/IPEDS DATABASE1" \
  --out-dir "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/IPEDS/IPEDS Panels/Panels" \
  --titles "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/titles_2023.txt" \
  --years 2004-2023 \
  --ucanaccess-lib "/Users/markjaysonfarol13/lib/ucanaccess"
```

CSV fallback (no Java; export VARTABLE##.csv with MDB ACCDB Viewer):

```bash
python3 ipeds/map_ipeds_vars.py \
  --csv-vartable-root "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/exports" \
  --out-dir "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/IPEDS/IPEDS Panels/Panels" \
  --titles "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/titles_2023.txt" \
  --years 2004-2023
```

Notes:
- In the Viewer, set CSV export to International/UTF-8. You can export multiple
  tables at once. Folder structure: `exports/YYYY/VARTABLE##.csv`.
- Quote paths containing spaces.

Outputs:
- `ipeds_var_map_<start>_<end>.csv` and `ipeds_extract_sql_<start>_<end>.sql` (JDBC)
- `ipeds_var_map_2004_2023.csv` and `ipeds_extract_sql_2004_2023.sql` (CSV fallback)

## Extracting Tidy Tables

Convert the mapping into per-table CSV exports for a specific year:

```bash
python3 ipeds/extract_ipeds_data.py \
  --year 2023 \
  --map-csv "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/IPEDS/IPEDS Panels/Panels/ipeds_var_map_2004_2023.csv" \
  --out-dir "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/IPEDS/IPEDS Panels/Panels"
```

Use `--tables` with a comma-separated list to restrict which mapped tables to export.
Missing columns are logged and dropped; `year` and `UNITID` are included.

## Acceptance Checks

Syntax check:

```bash
python3 -m compileall ipeds
```

CSV fallback smoke test (no Java):
- With one label (e.g., `Tuition and fees - Total`) and only a single exported
  `exports/2023/VARTABLE23.csv` present, the script should run and either create
  `ipeds_var_map_2004_2023.csv` with 2023 rows or warn about missing years if only
  2023 is present.

JDBC mode:
- Using the Desktop DB folder above, the script should build the mapping without
  complaining about missing JARs (assuming they exist in the configured folder).

Both modes print resolved paths (`db-dir`, `out-dir`, `titles`, and
`ucanaccess-lib` or `csv-vartable-root`).

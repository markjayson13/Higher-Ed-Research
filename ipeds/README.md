# IPEDS Mapping & Extraction Tools

These utilities help you translate checked IPEDS variable titles into their exact
`tableName` and `varName` identifiers using the authoritative `VARTABLE##`
metadata stored in each yearly Access database. They also support exporting tidy
CSVs for further analysis without copying any database files into the repository.

## Prerequisites (macOS)

1. Install Java (UCanAccess requires a JVM):
   ```bash
   brew install openjdk@11
   ```
   If you install a different version, ensure the `JAVA_HOME` environment variable
   points to the correct JDK.
2. Download the latest [UCanAccess](http://ucanaccess.sourceforge.net/site.html)
   release and extract all JAR files into `~/lib/ucanaccess` (or another folder of
   your choice). The directory should include files similar to:
   - `ucanaccess-<version>.jar`
   - `jackcess-<version>.jar`
   - `hsqldb-<version>.jar`
   - `commons-logging-<version>.jar`
   - `commons-lang-<version>.jar`

   The tools scan this directory and build the `CLASSPATH` automatically.
3. Create a virtual environment and install Python dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r ipeds/requirements.txt
   ```

### Optional configuration

You can override the default UCanAccess JAR directory by setting `UCANACCESS_LIB`
in `ipeds/.env` (copy `ipeds/.env.example`) or via the environment when launching
the scripts. All other paths default to the locations provided in the project
requirements, but you can override them through CLI flags.

## Mapping IPEDS variable titles

`map_ipeds_vars.py` reads a list of human-readable titles (one per line) and
searches every requested IPEDS Access database for matching entries in the
appropriate `VARTABLE##`. By default it covers 2004–2023 and expects titles in
`../titles_2023.txt`.

```bash
python ipeds/map_ipeds_vars.py \
  --db-dir "/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace/IPEDS COMPLETE DATABASE/IPEDS DATABASE" \
  --out-dir "/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace/IPEDS EXPORTS"
```

Key options:

- `--years`: Comma-separated list or ranges (e.g., `2007-2010,2012`).
- `--titles`: Path to a custom titles file.
- `--fuzzy`: Enable ≥90 token-set ratio fuzzy matching with RapidFuzz.
- `--ucanaccess-lib`: Override the UCanAccess JAR directory (same as `UCANACCESS_LIB`).
- `--verbose`: Print debug logs.

Outputs are written to the export directory:

- `ipeds_var_map_<start>_<end>.csv`: Mapping of `varTitle → tableName/varName` per year.
- `ipeds_extract_sql_<start>_<end>.sql`: SELECT templates with `<year> AS year`,
  `UNITID`, and all discovered variables.
- `missing_titles_<start>_<end>.csv`: Titles that were not found in the VARTABLE for each year.

The mapper trusts each year’s `VARTABLE##`, ensuring the proper finance form (F1,
F2, F3, etc.) is automatically selected according to the institution’s accounting
standards.

## Extracting tidy tables

`extract_ipeds_data.py` converts the mapping into CSV exports for a specific
year. It only selects columns confirmed to exist in the target table and keeps a
clean `year` plus `UNITID` identifier for panel analyses.

```bash
python ipeds/extract_ipeds_data.py \
  --year 2023 \
  --map-csv "/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace/IPEDS EXPORTS/ipeds_var_map_2004_2023.csv" \
  --out-dir "/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace/IPEDS EXPORTS"
```

Use `--tables` to provide a comma-separated list of table names or leave it as
`all` to export every mapped table for the year. Missing columns are logged and
silently dropped so that exports always reflect the database schema.

## Data locations

The scripts operate directly on the Access databases located at:
```
/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace/IPEDS COMPLETE DATABASE/IPEDS DATABASE
```
Exports are written to:
```
/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace/IPEDS EXPORTS
```
These paths are customizable via CLI options when needed.

## Safety notes

- The repository never stores `.mdb`/`.accdb` files or generated exports.
- Finance tables differ by sector and accounting standards; relying on
  `VARTABLE##` guarantees that the correct form (F1, F2, F3, etc.) is selected
  for each combination of year and institution type.

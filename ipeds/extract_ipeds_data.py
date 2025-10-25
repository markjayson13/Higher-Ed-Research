"""Extract IPEDS tables or build wide panel slices using a generated mapping."""
from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import jaydebeapi
import pandas as pd

from map_ipeds_vars import (
    DEFAULT_WORKSPACE,
    DEFAULT_DB_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_TITLES,
    YEAR_RX,
    collect_jars,
    connect_to_database,
    determine_ucanaccess_lib,
)

try:
    from ipeds import schema
except ImportError:  # pragma: no cover - adjust path when executed as script
    import sys
    from pathlib import Path as _Path

    package_root = _Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from ipeds import schema  # type: ignore

DEFAULT_MAP_PATH = DEFAULT_OUT_ROOT / "ipeds_var_map_2004_2023.csv"


@dataclass
class BuildYearResult:
    year: int
    data: pd.DataFrame
    canonical_sources: Dict[str, List[Tuple[str, str]]]
    coverage: Dict[str, int]
    finance_forms: Dict[str, str]


def find_db_for_year(db_root: Path, year: int) -> Path | None:
    """Search recursively for IPEDSYYYYMM.accdb (preferred) or IPEDSYYYYMM.mdb for a given year."""
    for path in db_root.rglob("IPEDS*.accdb"):
        m = YEAR_RX.search(path.name)
        if not m:
            continue
        if int(m.group(1)[:4]) == year:
            return path.resolve()
    # Fallback to .mdb by scanning digits
    for path in db_root.rglob("IPEDS*.mdb"):
        name = path.name
        digits = ''.join(ch for ch in name if ch.isdigit())
        if len(digits) >= 6 and int(digits[:4]) == year:
            return path.resolve()
    return None


def parse_years(years: str | None, fallback_year: int | None = None) -> List[int]:
    if years and years.strip():
        parts = [p.strip() for p in years.split(",") if p.strip()]
        result: set[int] = set()
        for part in parts:
            if "-" in part:
                a, b = part.split("-", 1)
                start, end = int(a), int(b)
                if start > end:
                    start, end = end, start
                result.update(range(start, end + 1))
            else:
                result.add(int(part))
        years_list = [y for y in result if 1900 <= y <= 2100]
        years_list.sort()
        return years_list
    if fallback_year is not None:
        return [fallback_year]
    return []


def load_mapping(map_path: Path, year: int) -> Dict[str, List[str]]:
    if not map_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {map_path}")
    grouped: Dict[str, List[str]] = {}
    with map_path.open("r", encoding="utf-8", newline="") as handle:
        reader = pd.read_csv(handle)
    if not {"year", "tableName", "varName"}.issubset(reader.columns):
        raise ValueError("Mapping CSV missing required columns.")
    filtered = reader[reader["year"] == year]
    for table_name, group in filtered.groupby("tableName"):
        grouped[table_name] = group["varName"].dropna().astype(str).tolist()
    return grouped


def fetch_table_columns(connection, table_name: str) -> List[str]:
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT * FROM {table_name} WHERE 1=0")
    except jaydebeapi.DatabaseError as exc:
        logging.warning("Unable to introspect %s: %s", table_name, exc)
        return []
    description = cursor.description or []
    return [column[0] for column in description]


def extract_table(
    connection,
    year: int,
    table_name: str,
    requested_columns: Sequence[str],
    out_dir: Path,
) -> None:
    available_columns = fetch_table_columns(connection, table_name)
    if not available_columns:
        logging.warning("Skipping %s for %s: no columns found", table_name, year)
        return

    selected = ["UNITID"] + [col for col in requested_columns if col in available_columns]
    missing = [col for col in requested_columns if col not in available_columns]
    if missing:
        logging.warning(
            "%s %s: dropping %s missing columns", year, table_name, ", ".join(missing)
        )

    if "UNITID" not in available_columns:
        logging.warning("%s %s: UNITID not available; skipping table", year, table_name)
        return

    query = f"SELECT {', '.join(dict.fromkeys(selected))} FROM {table_name}"
    frame = pd.read_sql_query(query, connection)
    frame.insert(0, "year", year)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"ipeds_{year}_{table_name}.csv"
    frame.to_csv(output_path, index=False)
    logging.info("Exported %s", output_path)


SAFE_COLUMN_RX = re.compile(r"[^0-9A-Za-z]+")


def canonicalize_title(value: str, *, default: str = "value") -> str:
    canonical = SAFE_COLUMN_RX.sub("_", str(value).strip().lower()).strip("_")
    return canonical or default


def load_table_subset(
    table_name: str,
    columns: Sequence[str],
    *,
    year: int,
    connection=None,
    tables_csv_root: Optional[Path] = None,
    primary_keys: Sequence[str] = ("UNITID",),
) -> pd.DataFrame:
    """Load a subset of columns from an IPEDS table via JDBC or CSV fallback."""
    if tables_csv_root is not None:
        csv_path = tables_csv_root / str(year) / f"{table_name}.csv"
        if csv_path.exists():
            try:
                frame = pd.read_csv(csv_path, dtype=str)
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("Failed to read %s: %s", csv_path, exc)
            else:
                available = [col for col in columns if col in frame.columns]
                keys = [key for key in primary_keys if key in frame.columns]
                if not keys:
                    logging.warning(
                        "%s %s: primary keys %s missing in CSV fallback.",
                        year,
                        table_name,
                        ", ".join(primary_keys),
                    )
                    return pd.DataFrame(columns=primary_keys)
                subset = frame[[*keys, *available]].copy()
                for key in keys:
                    subset[key] = subset[key].astype(str)
                return subset

    if connection is None:
        raise ValueError(
            "A database connection is required when --tables-csv-root is not supplied or the table CSV is missing."
        )

    available_columns = fetch_table_columns(connection, table_name)
    if not available_columns:
        logging.warning("Skipping %s for %s: no columns found", table_name, year)
        return pd.DataFrame(columns=primary_keys)

    for key in primary_keys:
        if key not in available_columns:
            logging.warning("%s %s: %s not available; skipping table", year, table_name, key)
            return pd.DataFrame(columns=primary_keys)

    selected = list(primary_keys) + [col for col in columns if col in available_columns]
    missing = [col for col in columns if col not in available_columns]
    if missing:
        logging.warning(
            "%s %s: dropping %s missing columns", year, table_name, ", ".join(missing)
        )

    query = f"SELECT {', '.join(dict.fromkeys(selected))} FROM {table_name}"
    frame = pd.read_sql_query(query, connection)
    for key in primary_keys:
        frame[key] = frame[key].astype(str)
    return frame


def build_wide_for_year(
    mapping_df: pd.DataFrame,
    year: int,
    *,
    connection=None,
    tables_csv_root: Optional[Path] = None,
    primary_keys: Sequence[str] = ("UNITID",),
) -> BuildYearResult:
    """Return canonical wide data, coverage, and finance form metadata for a year."""

    years_numeric = pd.to_numeric(mapping_df["year"], errors="coerce")
    mapping_year = mapping_df.loc[years_numeric == int(year)].copy()
    if mapping_year.empty:
        logging.warning("No mapping rows for %s; skipping wide build.", year)
        empty = pd.DataFrame(columns=[*primary_keys])
        return BuildYearResult(year, empty, {}, {}, {})

    mapping_year = mapping_year.fillna("")
    mapping_year = mapping_year[mapping_year["canonical"].astype(bool)]
    if mapping_year.empty:
        logging.warning("No canonical mappings available for %s.", year)
        empty = pd.DataFrame(columns=[*primary_keys])
        return BuildYearResult(year, empty, {}, {}, {})

    table_requirements: Dict[str, Set[str]] = defaultdict(set)
    for table_name, var_name in mapping_year[["tableName", "varName"]].itertuples(index=False):
        table_requirements[str(table_name)].add(str(var_name))

    table_frames: Dict[str, pd.DataFrame] = {}
    for table_name, var_names in table_requirements.items():
        subset = load_table_subset(
            table_name,
            sorted(var_names),
            year=year,
            connection=connection,
            tables_csv_root=tables_csv_root,
            primary_keys=primary_keys,
        )
        if subset.empty:
            logging.debug("%s %s: table returned no rows", year, table_name)
        table_frames[table_name] = subset

    wide_df: Optional[pd.DataFrame] = None
    canonical_sources: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    coverage: Dict[str, int] = {}
    finance_forms_map: Dict[str, Set[str]] = defaultdict(set)

    for canonical, group in mapping_year.groupby("canonical"):
        partitions: List[pd.DataFrame] = []
        for _, row in group.iterrows():
            table_name = str(row["tableName"])
            var_name = str(row["varName"])
            table_df = table_frames.get(table_name)
            if table_df is None or table_df.empty:
                continue
            if var_name not in table_df.columns:
                logging.debug("%s %s.%s not present in table", year, table_name, var_name)
                continue
            subset = table_df[[*primary_keys, var_name]].copy()
            subset = subset.rename(columns={var_name: canonical})
            partitions.append(subset)
            canonical_sources[canonical].append((table_name, var_name))

            form = schema.finance_form_from_table(table_name)
            if form:
                non_null_units = subset.loc[subset[canonical].notna(), primary_keys[0]]
                for unit in non_null_units.astype(str):
                    finance_forms_map[unit].add(form)

        if not partitions:
            continue

        combined = partitions[0]
        for extra in partitions[1:]:
            combined = combined.merge(extra, on=list(primary_keys), how="outer", suffixes=("", "__dup"))
            dup_col = f"{canonical}__dup"
            if dup_col in combined.columns:
                combined[canonical] = combined[canonical].combine_first(combined.pop(dup_col))

        combined = combined.drop_duplicates(subset=list(primary_keys))
        combined = combined[[*primary_keys, canonical]]
        coverage[canonical] = int(combined[canonical].notna().sum())

        if wide_df is None:
            wide_df = combined
        else:
            wide_df = wide_df.merge(combined, on=list(primary_keys), how="outer")

    if wide_df is None:
        logging.warning("No tables produced wide data for %s.", year)
        empty = pd.DataFrame(columns=[*primary_keys])
        return BuildYearResult(year, empty, dict(canonical_sources), coverage, {})

    primary_key = primary_keys[0]
    wide_df[primary_key] = wide_df[primary_key].astype(str)

    resolved_forms: Dict[str, str] = {}
    if finance_forms_map:
        priority = {"F1": 0, "F2": 1, "F3": 2}
        for unitid, forms in finance_forms_map.items():
            resolved = sorted(forms, key=lambda form: priority.get(form, 99))
            if resolved:
                resolved_forms[unitid] = resolved[0]

    return BuildYearResult(
        year=year,
        data=wide_df,
        canonical_sources=dict(canonical_sources),
        coverage=coverage,
        finance_forms=resolved_forms,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_ROOT, help="Directory with IPEDS databases")
    parser.add_argument("--year", type=int, required=False, help="Year to extract (used if --years omitted)")
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help=(
            "Year range or comma list (e.g., 2004-2023 or 2019,2021,2023). If omitted, use --year."
        ),
    )
    parser.add_argument("--map-csv", type=Path, default=DEFAULT_MAP_PATH, help="Mapping CSV path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT, help="Output directory")
    parser.add_argument(
        "--tables",
        type=str,
        default="all",
        help="Comma list of table names to export or 'all'",
    )
    parser.add_argument(
        "--ucanaccess-lib",
        type=str,
        default=None,
        help="Directory containing UCanAccess JAR files (overrides UCANACCESS_LIB)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve common paths and echo configuration
    out_dir = args.out_dir.expanduser().resolve()
    # Ensure defaults and selected out dir exist
    try:
        DEFAULT_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise SystemExit(f"Cannot create output root {DEFAULT_OUT_ROOT}: {exc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    map_csv = args.map_csv.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    lib_dir = determine_ucanaccess_lib(args.ucanaccess_lib, script_dir)
    logging.info("Resolved paths:")
    logging.info("  db-dir: %s", args.db_dir.expanduser().resolve())
    logging.info("  out-dir: %s", out_dir)
    logging.info("  map-csv: %s", map_csv)
    logging.info("  ucanaccess-lib: %s", lib_dir)

    # Determine years list
    years = parse_years(args.years, args.year)
    if not years:
        logging.error("No valid year(s) provided. Use --year or --years.")
        return 1

    classpath = collect_jars(lib_dir)
    db_dir = args.db_dir.expanduser().resolve()

    # Extract year by year
    for year in years:
        mapping = load_mapping(map_csv, year)
        if args.tables.lower() != "all":
            requested_tables = {table.strip() for table in args.tables.split(",") if table.strip()}
            mapping = {table: columns for table, columns in mapping.items() if table in requested_tables}
            missing_tables = requested_tables - mapping.keys()
            for table in sorted(missing_tables):
                logging.warning("No mapping found for table %s", table)

        if not mapping:
            logging.warning("No tables to export for %s", year)
            continue

        db_path = find_db_for_year(db_dir, year)
        if not db_path:
            logging.error(
                "Database for %s not found under %s (expected IPEDSYYYYMM.accdb/.mdb)",
                year,
                db_dir,
            )
            continue

        with connect_to_database(db_path, classpath) as connection:
            for table_name, columns in sorted(mapping.items()):
                extract_table(
                    connection,
                    year,
                    table_name,
                    columns,
                    out_dir,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

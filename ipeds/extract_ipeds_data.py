"""Extract IPEDS tables to tidy CSV files using a generated mapping."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import jaydebeapi
import pandas as pd

from map_ipeds_vars import (
    DEFAULT_DB_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_TITLES,
    collect_jars,
    connect_to_database,
    determine_ucanaccess_lib,
)

DEFAULT_MAP_PATH = DEFAULT_OUT_DIR / "ipeds_var_map_2004_2023.csv"


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="Directory with IPEDS databases")
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
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

        db_path = db_dir / f"IPEDS{year}.accdb"
        if not db_path.exists():
            alternative = db_dir / f"IPEDS{year}.mdb"
            if alternative.exists():
                db_path = alternative
            else:
                logging.error("Database for %s not found in %s", year, db_dir)
                continue
        db_path = db_path.resolve()

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

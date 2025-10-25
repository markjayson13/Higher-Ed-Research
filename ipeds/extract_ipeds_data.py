"""Extract IPEDS tables to tidy CSV files using a generated mapping."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import jaydebeapi
import pandas as pd

from map_ipeds_vars import (
    DEFAULT_DB_ROOT,
    DEFAULT_OUT_ROOT,
    collect_jars,
    connect_to_database,
    determine_ucanaccess_lib,
    find_database_path,
)

DEFAULT_OUT_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_MAP_PATH = DEFAULT_OUT_ROOT / "ipeds_var_map_2004_2023.csv"


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


def extract_table_from_db(
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


def find_csv_table(csv_root: Path, year: int, table_name: str) -> Path | None:
    year_dir = csv_root / str(year)
    if not year_dir.exists():
        return None
    normalized = table_name.lower()
    for candidate in year_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".csv" and candidate.stem.lower() == normalized:
            return candidate
    return None


def extract_table_from_csv(
    csv_root: Path,
    year: int,
    table_name: str,
    requested_columns: Sequence[str],
    out_dir: Path,
) -> None:
    csv_path = find_csv_table(csv_root, year, table_name)
    if not csv_path:
        logging.warning(
            "%s %s: CSV export not found under %s", year, table_name, csv_root / str(year)
        )
        return

    frame = pd.read_csv(csv_path)
    available_columns = list(frame.columns)

    if "UNITID" not in available_columns:
        logging.warning("%s %s: UNITID not available in CSV export; skipping", year, table_name)
        return

    selected = [col for col in requested_columns if col in available_columns]
    missing = [col for col in requested_columns if col not in available_columns]
    if missing:
        logging.warning(
            "%s %s: dropping %s missing columns from CSV export", year, table_name, ", ".join(missing)
        )

    frame = frame[["UNITID", *selected]]
    frame.insert(0, "year", year)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"ipeds_{year}_{table_name}.csv"
    frame.to_csv(output_path, index=False)
    logging.info("Exported %s", output_path)


def export_tables_for_year(
    year: int,
    mapping: Dict[str, List[str]],
    out_dir: Path,
    db_root: Path,
    classpath: Sequence[str] | None = None,
    csv_data_root: Path | None = None,
) -> List[Path]:
    exported: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mapping:
        logging.warning("No tables to export for %s", year)
        return exported

    if csv_data_root is not None:
        csv_root = csv_data_root.expanduser().resolve()
        for table_name, columns in sorted(mapping.items()):
            extract_table_from_csv(csv_root, year, table_name, columns, out_dir)
            output_path = out_dir / f"ipeds_{year}_{table_name}.csv"
            if output_path.exists():
                exported.append(output_path)
        return exported

    if classpath is None:
        raise ValueError("classpath is required when csv_data_root is not provided")

    db_path = find_database_path(db_root, year)
    if not db_path:
        logging.error("Database for %s not found in %s", year, db_root)
        return exported

    with connect_to_database(db_path, classpath) as connection:
        for table_name, columns in sorted(mapping.items()):
            extract_table_from_db(connection, year, table_name, columns, out_dir)
            output_path = out_dir / f"ipeds_{year}_{table_name}.csv"
            if output_path.exists():
                exported.append(output_path)

    return exported


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT, help="Directory with IPEDS databases")
    parser.add_argument("--year", type=int, required=True, help="Year to extract")
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
    parser.add_argument(
        "--csv-data-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing exported IPEDS CSV tables structured as"
            " {root}/{year}/{tableName}.csv to avoid JDBC access"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    year = args.year
    mapping = load_mapping(args.map_csv.expanduser().resolve(), year)

    if args.tables.lower() != "all":
        requested_tables = {table.strip() for table in args.tables.split(",") if table.strip()}
        mapping = {table: columns for table, columns in mapping.items() if table in requested_tables}
        missing_tables = requested_tables - mapping.keys()
        for table in sorted(missing_tables):
            logging.warning("No mapping found for table %s", table)

    if not mapping:
        logging.error("No tables to export for %s", year)
        return 1

    csv_data_root = args.csv_data_root.expanduser().resolve() if args.csv_data_root else None

    classpath: List[str] | None = None
    if csv_data_root is None:
        script_dir = Path(__file__).resolve().parent
        lib_dir = determine_ucanaccess_lib(args.ucanaccess_lib, script_dir)
        classpath = collect_jars(lib_dir)

    out_dir = args.out_dir.expanduser().resolve()
    db_root = args.db_root.expanduser().resolve()

    export_tables_for_year(
        year=year,
        mapping=mapping,
        out_dir=out_dir,
        db_root=db_root,
        classpath=classpath,
        csv_data_root=csv_data_root,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

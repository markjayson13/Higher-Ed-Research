"""Map IPEDS variable titles to table and variable names.

This script connects to IPEDS Access databases using UCanAccess via JayDeBeApi.
It generates mapping CSV and SQL extraction templates for the requested years.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import jaydebeapi
import pandas as pd
from rapidfuzz import fuzz, process

DEFAULT_DB_DIR = Path(
    "/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace/"
    "IPEDS COMPLETE DATABASE/IPEDS DATABASE"
)
DEFAULT_OUT_DIR = Path(
    "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/IPEDS/IPEDS Panels/Panels"
)
DEFAULT_TITLES = Path(
    "/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/titles_2023.txt"
)
DEFAULT_UCAN = Path("/Users/markjaysonfarol13/lib/ucanaccess")
DEFAULT_YEARS = tuple(range(2004, 2024))
FUZZY_THRESHOLD = 90


@dataclass
class VarRecord:
    year: int
    table_name: str
    var_name: str
    var_title: str


def parse_years(years: str | None) -> List[int]:
    """Parse a years argument into a sorted list of unique integers."""
    if not years:
        return list(DEFAULT_YEARS)
    result: set[int] = set()
    for part in years.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    filtered = [year for year in result if 1900 <= year <= 2100]
    filtered.sort()
    return filtered


def read_titles(path: Path) -> List[str]:
    """Return cleaned titles from the provided file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Titles file not found: {path}. Provide a valid path with --titles."
        )
    titles: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            titles.append(cleaned)
    if not titles:
        raise ValueError(
            f"No non-blank titles were found in {path}. Ensure it lists one title per line."
        )
    return titles


def read_env_file(env_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not env_path.exists():
        return data
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def determine_ucanaccess_lib(cli_value: Path | None, script_dir: Path) -> Path:
    """Resolve the directory that contains UCanAccess JARs."""
    if cli_value and cli_value != DEFAULT_UCAN:
        return cli_value.expanduser().resolve()

    env_data = read_env_file(script_dir / ".env")
    env_data.update(read_env_file(script_dir.parent / ".env"))
    env_data.update(os.environ)

    lib_override = env_data.get("UCANACCESS_LIB")
    if lib_override:
        return Path(lib_override).expanduser().resolve()

    if cli_value:
        return cli_value.expanduser().resolve()

    return DEFAULT_UCAN.expanduser().resolve()


def collect_jars(lib_dir: Path) -> List[str]:
    if not lib_dir.exists():
        raise FileNotFoundError(
            "UCanAccess library directory not found: "
            f"{lib_dir}. Install OpenJDK (e.g., 'brew install openjdk') and copy the five "
            "JARs from the UCanAccess ZIP (ucanaccess, jackcess, hsqldb, commons-logging, "
            "commons-lang) into this folder."
        )
    jars = sorted(str(path) for path in lib_dir.glob("*.jar"))
    if not jars:
        raise FileNotFoundError(
            "No JAR files were found in "
            f"{lib_dir}. Confirm the UCanAccess ZIP contents (ucanaccess, jackcess, hsqldb, "
            "commons-logging, commons-lang JARs) are copied here."
        )
    return jars


def get_database_path(db_dir: Path, year: int) -> Path | None:
    """Return the Access database path for the given year, if it exists."""
    candidates = [
        db_dir / f"IPEDS{year}.accdb",
        db_dir / f"IPEDS{year}.mdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def connect_to_database(db_path: Path, classpath: Sequence[str]):
    url = f"jdbc:ucanaccess:///{db_path};memory=false;immediatelyReleaseResources=true"
    return jaydebeapi.connect(
        "net.ucanaccess.jdbc.UcanaccessDriver",
        url,
        jars=os.pathsep.join(classpath),
    )


def fetch_vartable_records(cursor, table_name: str) -> List[Tuple[str, str, str]]:
    query = f"SELECT tableName, varName, varTitle FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def map_titles_from_records(
    year: int,
    titles: Sequence[str],
    records: Iterable[Tuple[str, str, str]],
    use_fuzzy: bool,
) -> Tuple[List[VarRecord], List[Tuple[int, str]]]:
    by_title: Dict[str, List[VarRecord]] = defaultdict(list)
    for table_name, var_name, var_title in records:
        record = VarRecord(year=year, table_name=table_name, var_name=var_name, var_title=var_title)
        by_title[var_title].append(record)

    found_records: List[VarRecord] = []
    missing: List[Tuple[int, str]] = []

    if use_fuzzy:
        unique_titles = list(by_title)
        for title in titles:
            matches = process.extract(
                title,
                unique_titles,
                scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if not matches:
                missing.append((year, title))
                continue
            for matched_title, score, _ in matches:
                found_records.extend(by_title[matched_title])
                logging.debug(
                    "Fuzzy match (%s) → (%s) with score %s for %s",
                    title,
                    matched_title,
                    score,
                    year,
                )
    else:
        for title in titles:
            matches = by_title.get(title)
            if not matches:
                missing.append((year, title))
            else:
                found_records.extend(matches)

    unique: Dict[Tuple[str, str], VarRecord] = {}
    for record in found_records:
        unique[(record.table_name, record.var_name)] = record

    return list(unique.values()), missing


def load_vartable_records_from_csv(csv_root: Path, year: int) -> List[Tuple[str, str, str]]:
    year_folder = csv_root / str(year)
    csv_path = year_folder / f"VARTABLE{year % 100:02d}.csv"
    if not csv_path.exists():
        logging.warning("CSV VARTABLE not found for %s: %s", year, csv_path)
        return []

    frame = pd.read_csv(csv_path)
    required_columns = {"tableName", "varName", "varTitle"}
    if not required_columns.issubset(frame.columns):
        missing_cols = required_columns - set(frame.columns)
        logging.error(
            "%s is missing required columns: %s", csv_path, ", ".join(sorted(missing_cols))
        )
        return []

    return list(
        zip(
            frame["tableName"].astype(str),
            frame["varName"].astype(str),
            frame["varTitle"].astype(str),
        )
    )


def write_mapping_csv(records: Sequence[VarRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "tableName", "varName", "varTitle"])
        for record in sorted(
            records,
            key=lambda item: (item.year, item.table_name.lower(), item.var_name.lower()),
        ):
            writer.writerow([record.year, record.table_name, record.var_name, record.var_title])


def write_missing_csv(missing: Sequence[Tuple[int, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "requestedTitle"])
        for year, title in sorted(missing, key=lambda item: (item[0], item[1].lower())):
            writer.writerow([year, title])


def write_sql_template(records: Sequence[VarRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    for record in records:
        key = (record.year, record.table_name)
        grouped[key].append(record.var_name)

    with output_path.open("w", encoding="utf-8") as handle:
        for (year, table_name), var_names in sorted(grouped.items()):
            ordered_vars = sorted(dict.fromkeys(var_names), key=str.lower)
            columns = ", ".join([f"{year} AS year", "UNITID", *ordered_vars])
            handle.write(
                f"SELECT {columns}\nFROM {table_name}; -- {year}\n\n"
            )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def perform_self_test(records: Sequence[VarRecord], titles: Sequence[str]) -> None:
    sentinel = "tuition and fees - total"
    if any(sentinel in title.lower() for title in titles):
        if not any(sentinel in record.var_title.lower() for record in records):
            raise RuntimeError(
                "Self-test failed: expected to find a mapping for 'Tuition and fees - Total'."
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="Directory with IPEDS databases")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory to store outputs")
    parser.add_argument("--years", type=str, default=None, help="Comma-separated years or ranges (e.g., 2004-2006,2010)")
    parser.add_argument(
        "--titles",
        type=Path,
        default=DEFAULT_TITLES,
        help=f"Path to file listing variable titles (default: {DEFAULT_TITLES})",
    )
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy title matching with RapidFuzz")
    parser.add_argument(
        "--ucanaccess-lib",
        type=Path,
        default=DEFAULT_UCAN,
        help="Directory containing UCanAccess JAR files (overrides UCANACCESS_LIB)",
    )
    parser.add_argument(
        "--csv-vartable-root",
        type=Path,
        default=None,
        help="Optional: root folder with exported VARTABLE##.csv by year (CSV fallback, no JDBC needed)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    years = parse_years(args.years)
    if not years:
        logging.error("No valid years were provided.")
        return 1
    titles_path = args.titles.expanduser().resolve()
    try:
        titles = read_titles(titles_path)
    except (FileNotFoundError, ValueError) as exc:
        logging.error(exc)
        return 1

    logging.info("Loaded %s titles from %s", len(titles), titles_path)
    preview_count = min(5, len(titles))
    logging.info("First %s titles: %s", preview_count, titles[:preview_count])

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing outputs to: %s", out_dir)
    logging.info(
        'Tip: quote paths with spaces in shell commands (e.g., "--out-dir \"%s\"").',
        out_dir,
    )

    db_dir = args.db_dir.expanduser().resolve()
    logging.info("Searching databases in: %s", db_dir)

    csv_root: Path | None = None
    classpath: List[str] = []
    if args.csv_vartable_root:
        csv_root = args.csv_vartable_root.expanduser().resolve()
        if not csv_root.exists():
            logging.error("CSV VARTABLE root not found: %s", csv_root)
            return 1
        logging.info("Using CSV VARTABLE root: %s", csv_root)
    else:
        script_dir = Path(__file__).resolve().parent
        lib_dir = determine_ucanaccess_lib(args.ucanaccess_lib, script_dir)
        try:
            classpath = collect_jars(lib_dir)
        except FileNotFoundError as exc:
            logging.error(exc)
            return 1
        logging.info("Using UCanAccess library directory: %s", lib_dir)

    all_records: List[VarRecord] = []
    all_missing: List[Tuple[int, str]] = []
    year_stats: Dict[int, Tuple[int, int, int]] = {}
    requested_total = len(titles)

    for year in years:
        logging.info("Processing %s", year)
        raw_records: List[Tuple[str, str, str]] = []
        if csv_root is not None:
            raw_records = load_vartable_records_from_csv(csv_root, year)
            if not raw_records:
                missing = [(year, title) for title in titles]
                all_missing.extend(missing)
                year_stats[year] = (requested_total, 0, len(missing))
                logging.warning(
                    "%s: requested=%s, found=0, missing=%s (CSV fallback)",
                    year,
                    requested_total,
                    len(missing),
                )
                continue
        else:
            db_path = get_database_path(db_dir, year)
            if not db_path:
                missing = [(year, title) for title in titles]
                all_missing.extend(missing)
                year_stats[year] = (requested_total, 0, len(missing))
                logging.warning("Database for %s not found in %s", year, db_dir)
                logging.warning(
                    "%s: requested=%s, found=0, missing=%s",
                    year,
                    requested_total,
                    len(missing),
                )
                continue
            vartable_name = f"VARTABLE{year % 100:02d}"
            try:
                with connect_to_database(db_path, classpath) as connection:
                    cursor = connection.cursor()
                    raw_records = fetch_vartable_records(cursor, vartable_name)
            except jaydebeapi.DatabaseError as exc:
                logging.error("Failed to query %s in %s: %s", vartable_name, db_path, exc)
                missing = [(year, title) for title in titles]
                all_missing.extend(missing)
                year_stats[year] = (requested_total, 0, len(missing))
                continue

        records, missing = map_titles_from_records(year, titles, raw_records, args.fuzzy)
        all_records.extend(records)
        all_missing.extend(missing)
        year_stats[year] = (requested_total, len(records), len(missing))
        logging.info(
            "%s: requested=%s, found=%s, missing=%s",
            year,
            requested_total,
            len(records),
            len(missing),
        )

    mapping_path = out_dir / f"ipeds_var_map_{years[0]}_{years[-1]}.csv"
    sql_path = out_dir / f"ipeds_extract_sql_{years[0]}_{years[-1]}.sql"
    missing_path = out_dir / f"missing_titles_{years[0]}_{years[-1]}.csv"

    write_mapping_csv(all_records, mapping_path)
    write_sql_template(all_records, sql_path)
    write_missing_csv(all_missing, missing_path)

    perform_self_test(all_records, titles)

    for year in years:
        requested, found, missing = year_stats.get(year, (requested_total, 0, requested_total))
        logging.info(
            "Summary %s → requested=%s, found=%s, missing=%s",
            year,
            requested,
            found,
            missing,
        )
    logging.info("Wrote mapping to %s", mapping_path)
    logging.info("Wrote SQL template to %s", sql_path)
    logging.info("Wrote missing titles to %s", missing_path)
    logging.info("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

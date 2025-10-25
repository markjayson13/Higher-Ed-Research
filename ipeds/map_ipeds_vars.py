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
import re
from typing import Dict, List, Sequence, Tuple

import jaydebeapi
from rapidfuzz import fuzz, process

# Workspace defaults
DEFAULT_WORKSPACE = Path("/Users/markjaysonfarol13/Higher Ed research/IPEDS/IPEDS workspace")
DEFAULT_DB_ROOT = DEFAULT_WORKSPACE / "IPEDS COMPLETE DATABASE"
DEFAULT_OUT_ROOT = DEFAULT_WORKSPACE / "PANELS"
DEFAULT_TITLES = Path("/Users/markjaysonfarol13/Documents/GitHub/Higher-Ed-Research/titles_2023.txt")
DEFAULT_UCAN = Path("/Users/markjaysonfarol13/lib/ucanaccess")
DEFAULT_YEARS = tuple(range(2004, 2024))
FUZZY_THRESHOLD = 90

# Year inference from DB filename, e.g., IPEDS200405.accdb → 2004
YEAR_RX = re.compile(r"IPEDS(\d{6})\.accdb$", re.IGNORECASE)


def infer_year_from_db(db_path: Path) -> int:
    """Map IPEDSYYYYMM.accdb → start year (YYYY)."""
    m = YEAR_RX.search(db_path.name)
    if not m:
        raise ValueError(f"Cannot infer year from filename: {db_path.name}")
    yyyymm = m.group(1)
    return int(yyyymm[:4])


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
        raise FileNotFoundError(f"Titles file not found: {path}")
    titles: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            titles.append(cleaned)
    if not titles:
        logging.warning("No titles found in %s", path)
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


def determine_ucanaccess_lib(cli_value: str | None, script_dir: Path) -> Path:
    """Resolve the directory that contains UCanAccess JARs."""
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    env_data = read_env_file(script_dir / ".env")
    env_data.update(read_env_file(script_dir.parent / ".env"))
    env_data.update(os.environ)

    lib_override = env_data.get("UCANACCESS_LIB")
    if lib_override:
        return Path(lib_override).expanduser().resolve()

    # Fall back to project/user default
    return DEFAULT_UCAN.expanduser().resolve()


def collect_jars(lib_dir: Path) -> List[str]:
    if not lib_dir.exists():
        raise FileNotFoundError(
            "UCanAccess library directory not found: "
            f"{lib_dir}. Expected JARs from UCanAccess (e.g., UCanAccess-5.0.1-bin.zip).\n"
            "Download and extract, then place these in the folder: ucanaccess-*.jar, "
            "jackcess-*.jar, hsqldb-*.jar, commons-logging-*.jar, commons-lang-*.jar."
        )
    jars = sorted(str(path) for path in lib_dir.glob("*.jar"))
    if not jars:
        raise FileNotFoundError(
            f"No JAR files were found in {lib_dir}. Ensure the UCanAccess JARs are available.\n"
            "Look for files like: ucanaccess-<ver>.jar, jackcess-<ver>.jar, hsqldb-<ver>.jar, "
            "commons-logging-<ver>.jar, commons-lang-<ver>.jar from UCanAccess-5.0.1-bin.zip."
        )
    return jars


def get_database_path(db_dir: Path, year: int) -> Path | None:
    """Return the Access database path for the given year by searching common layouts."""
    # Direct names first
    for candidate in [db_dir / f"IPEDS{year}.accdb", db_dir / f"IPEDS{year}.mdb"]:
        if candidate.exists():
            return candidate.resolve()
    # Recursive search for IPEDSYYYYMM.accdb
    for path in db_dir.rglob("IPEDS*.accdb"):
        try:
            y = infer_year_from_db(path)
        except Exception:
            continue
        if y == year:
            return path.resolve()
    # Fallback for .mdb by scanning digits
    for path in db_dir.rglob("IPEDS*.mdb"):
        name = path.name
        digits = ''.join(ch for ch in name if ch.isdigit())
        if len(digits) >= 6 and int(digits[:4]) == year:
            return path.resolve()
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


def map_titles_for_year(
    year: int,
    titles: Sequence[str],
    db_dir: Path,
    classpath: Sequence[str],
    use_fuzzy: bool,
) -> Tuple[List[VarRecord], List[Tuple[int, str]]]:
    db_path = get_database_path(db_dir, year)
    if not db_path:
        logging.warning("Database for %s not found in %s", year, db_dir)
        return [], [(year, title) for title in titles]

    vartable_name = f"VARTABLE{year % 100:02d}"
    try:
        with connect_to_database(db_path, classpath) as connection:
            cursor = connection.cursor()
            records = fetch_vartable_records(cursor, vartable_name)
    except jaydebeapi.DatabaseError as exc:
        logging.error("Failed to query %s in %s: %s", vartable_name, db_path, exc)
        return [], [(year, title) for title in titles]

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
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_ROOT, help="Directory with IPEDS databases")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT, help="Directory to store outputs")
    parser.add_argument("--years", type=str, default=None, help="Comma-separated years or ranges (e.g., 2004-2006,2010)")
    parser.add_argument("--titles", type=Path, default=DEFAULT_TITLES, help="Path to file listing variable titles")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy title matching with RapidFuzz")
    parser.add_argument(
        "--ucanaccess-lib",
        type=str,
        default=None,
        help="Directory containing UCanAccess JAR files (overrides UCANACCESS_LIB)",
    )
    parser.add_argument(
        "--csv-vartable-root",
        type=Path,
        default=None,
        help=(
            "Root folder with per-year subfolders containing VARTABLE##.csv "
            "(CSV fallback; no JDBC needed)"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    # Resolve common paths and echo configuration
    titles_path = args.titles.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    # Ensure defaults and selected out dir exist
    try:
        DEFAULT_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise SystemExit(f"Cannot create output root {DEFAULT_OUT_ROOT}: {exc}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV fallback mode (no Java/JDBC needed)
    if args.csv_vartable_root:
        csv_root = args.csv_vartable_root.expanduser().resolve()
        db_dir = args.db_dir.expanduser().resolve()
        logging.info("Resolved paths:")
        logging.info("  db-dir: %s", db_dir)
        logging.info("  titles: %s", titles_path)
        logging.info("  out-dir: %s", out_dir)
        logging.info("  csv-vartable-root: %s", csv_root)

        titles = read_titles(titles_path)
        if not titles:
            raise SystemExit(f"No titles found in {titles_path}")

        import pandas as pd  # local import to make mode explicit

        rows = []
        years = parse_years(args.years)
        if not years:
            logging.error("No valid years were provided.")
            return 1
        processed_years = []
        total_hits = 0
        for year in years:
            yy = f"{year}"[-2:]
            csv_path = csv_root / str(year) / f"VARTABLE{yy}.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing {csv_path} — skipping {year}")
                continue
            try:
                df = pd.read_csv(csv_path, dtype=str).fillna("")
            except Exception as exc:
                raise SystemExit(f"Failed to read {csv_path}: {exc}")
            required = {"tableName", "varName", "varTitle"}
            if not required.issubset(df.columns):
                missing_cols = required - set(df.columns)
                raise SystemExit(f"{csv_path} is missing columns: {missing_cols}")

            hit = df[df["varTitle"].isin(titles)].copy()
            if hit.empty:
                print(f"[INFO] No matches in {csv_path.name}")
                processed_years.append(year)
                continue
            hit.insert(0, "year", year)
            sub = hit[["year", "tableName", "varName", "varTitle"]]
            rows.append(sub)
            total_hits += len(sub)
            processed_years.append(year)

        if not rows:
            raise SystemExit(
                "No mappings found in CSV fallback. Check titles and exported VARTABLE##.csv files."
            )

        out = pd.concat(rows, ignore_index=True).drop_duplicates().sort_values([
            "year", "tableName", "varName"
        ])

        map_csv = out_dir / "ipeds_var_map_2004_2023.csv"
        out.to_csv(map_csv, index=False)

        # Optional SQL template mirroring JDBC mode naming
        sql_out = out_dir / "ipeds_extract_sql_2004_2023.sql"
        with sql_out.open("w", encoding="utf-8") as f:
            for (year, table), g in out.groupby(["year", "tableName"]):
                cols = ",\n  ".join(sorted(set(g["varName"])) )
                f.write(
                    f"-- year {year}, table {table}\nSELECT\n  {year} AS year,\n  UNITID,\n  {cols}\nFROM {table};\n\n"
                )

        # Summary and exit
        print("[OK] Wrote mapping:", map_csv)
        print(
            f"[SUMMARY] years processed={len(processed_years)}, titles requested={len(titles)}, "
            f"rows written={len(out)}, distinct mappings={out.drop(columns=['varTitle']).drop_duplicates().shape[0]}"
        )
        return 0

    # JDBC / UCanAccess mode
    years = parse_years(args.years)
    if not years:
        logging.error("No valid years were provided.")
        return 1
    titles = read_titles(titles_path)
    if not titles:
        raise SystemExit(f"No titles found in {titles_path}")

    script_dir = Path(__file__).resolve().parent
    lib_dir = determine_ucanaccess_lib(args.ucanaccess_lib, script_dir)
    classpath = collect_jars(lib_dir)

    db_dir = args.db_dir.expanduser().resolve()

    logging.info("Resolved paths:")
    logging.info("  titles: %s", titles_path)
    logging.info("  out-dir: %s", out_dir)
    logging.info("  db-dir: %s", db_dir)
    logging.info("  ucanaccess-lib: %s", lib_dir)

    all_records: List[VarRecord] = []
    all_missing: List[Tuple[int, str]] = []

    for year in years:
        logging.info("Processing %s", year)
        records, missing = map_titles_for_year(year, titles, db_dir, classpath, args.fuzzy)
        logging.info("%s: requested=%s, found=%s, missing=%s", year, len(titles), len(records), len(missing))
        all_records.extend(records)
        all_missing.extend(missing)

    mapping_path = out_dir / f"ipeds_var_map_{years[0]}_{years[-1]}.csv"
    sql_path = out_dir / f"ipeds_extract_sql_{years[0]}_{years[-1]}.sql"
    missing_path = out_dir / f"missing_titles_{years[0]}_{years[-1]}.csv"

    write_mapping_csv(all_records, mapping_path)
    write_sql_template(all_records, sql_path)
    write_missing_csv(all_missing, missing_path)

    # Summary
    total_found = len(all_records)
    total_missing = len(all_missing)
    logging.info(
        "Summary: years=%s, titles requested per year=%s, total found=%s, total missing=%s",
        len(years), len(titles), total_found, total_missing,
    )
    return 0

    perform_self_test(all_records, titles)

    logging.info("Wrote mapping to %s", mapping_path)
    logging.info("Wrote SQL template to %s", sql_path)
    logging.info("Wrote missing titles to %s", missing_path)
    logging.info("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

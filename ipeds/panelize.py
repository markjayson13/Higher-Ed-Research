"""Orchestrate IPEDS mapping, extraction, and panel construction."""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

import map_ipeds_vars
import extract_ipeds_data


def discover_databases(db_root: Path) -> Dict[int, Path]:
    """Return a mapping from year to database path by crawling the root directory."""
    result: Dict[int, Path] = {}
    for path in sorted(db_root.rglob("IPEDS*.accdb")):
        try:
            year = map_ipeds_vars.infer_year_from_db(path)
        except ValueError:
            continue
        result.setdefault(year, path.resolve())
    return result


def discover_csvmeta_years(csv_root: Path | None) -> List[int]:
    if not csv_root:
        return []
    root = csv_root.expanduser().resolve()
    if not root.exists():
        return []
    years: List[int] = []
    for entry in root.iterdir():
        if entry.is_dir():
            try:
                years.append(int(entry.name))
            except ValueError:
                continue
    return sorted(set(years))


def load_vartable_records_from_csv(year: int, csv_root: Path) -> List[map_ipeds_vars.VarRecord]:
    """Load vartable metadata from CSV exports."""
    year_dir = csv_root / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Metadata directory missing: {year_dir}")

    candidates: List[Path] = [
        year_dir / f"VARTABLE{year % 100:02d}.csv",
        year_dir / f"VARTABLE{year % 100:02d}.CSV",
    ]
    if not any(candidate.exists() for candidate in candidates):
        candidates = list(year_dir.glob("VARTABLE*.csv")) + list(year_dir.glob("VARTABLE*.CSV"))
    csv_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if not csv_path:
        raise FileNotFoundError(f"VARTABLE CSV not found for {year} under {year_dir}")

    frame = pd.read_csv(csv_path)
    required = {"tableName", "varName", "varTitle"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"VARTABLE CSV {csv_path} missing required columns: {required - set(frame.columns)}"
        )

    records = [
        map_ipeds_vars.VarRecord(
            year=year,
            table_name=str(row["tableName"]),
            var_name=str(row["varName"]),
            var_title=str(row["varTitle"]),
        )
        for _, row in frame.iterrows()
    ]
    return records


def group_records_by_year(records: Iterable[map_ipeds_vars.VarRecord]) -> Dict[int, Dict[str, List[str]]]:
    grouped: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.year][record.table_name].append(record.var_name)

    normalized: Dict[int, Dict[str, List[str]]] = {}
    for year, table_map in grouped.items():
        normalized[year] = {
            table: list(dict.fromkeys(columns))
            for table, columns in table_map.items()
        }
    return normalized


def parse_years_argument(
    years_arg: str | None, available_years: Sequence[int]
) -> tuple[List[int], List[int]]:
    available_set = set(available_years)
    if years_arg:
        requested = map_ipeds_vars.parse_years(years_arg)
    else:
        requested = sorted(available_years)
    selected = [year for year in requested if year in available_set]
    missing = sorted(set(requested) - available_set)
    return selected, missing


def build_panel(out_root: Path, years: Sequence[int], year_table_mapping: Dict[int, Dict[str, List[str]]]) -> Path | None:
    data_frames: List[pd.DataFrame] = []
    for year in years:
        tables = year_table_mapping.get(year, {})
        for table_name in tables:
            csv_path = out_root / f"ipeds_{year}_{table_name}.csv"
            if not csv_path.exists():
                logging.warning("Expected export missing: %s", csv_path)
                continue
            frame = pd.read_csv(csv_path)
            if "year" not in frame.columns:
                frame.insert(0, "year", year)
            frame["source_table"] = table_name
            data_frames.append(frame)

    if not data_frames:
        logging.warning("No data files available to build panel in %s", out_root)
        return None

    panel = pd.concat(data_frames, ignore_index=True, sort=False)
    start, end = min(years), max(years)
    panel_path = out_root / f"ipeds_panel_{start}_{end}.csv"
    panel.to_csv(panel_path, index=False)
    logging.info("Wrote panel to %s (%s rows, %s columns)", panel_path, len(panel), len(panel.columns))
    return panel_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=map_ipeds_vars.DEFAULT_DB_ROOT, help="Root directory containing IPEDS Access databases")
    parser.add_argument("--out-root", type=Path, default=map_ipeds_vars.DEFAULT_OUT_ROOT, help="Directory for outputs")
    parser.add_argument("--titles", type=Path, default=map_ipeds_vars.DEFAULT_TITLES, help="Variable titles file")
    parser.add_argument("--years", type=str, default=None, help="Year list or range (e.g., 2004-2023)")
    parser.add_argument("--ucanaccess-lib", type=Path, default=None, help="Directory with UCanAccess JARs")
    parser.add_argument("--mode", choices=["jdbc", "csvmeta"], default="jdbc", help="Metadata extraction mode")
    parser.add_argument("--csv-vartable-root", type=Path, default=None, help="Root containing exported VARTABLE CSV folders per year")
    parser.add_argument("--csv-data-root", type=Path, default=None, help="Root containing exported table CSVs per year for CSV fallback")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    db_root = args.db_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    titles_path = args.titles.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    csv_vartable_root = args.csv_vartable_root.expanduser().resolve() if args.csv_vartable_root else None
    csv_data_root = args.csv_data_root.expanduser().resolve() if args.csv_data_root else None

    print("Resolved configuration:")
    print(f"  Workspace: {map_ipeds_vars.DEFAULT_WORKSPACE}")
    print(f"  DB root: {db_root}")
    print(f"  Output root: {out_root}")
    print(f"  Titles file: {titles_path}")
    print(f"  Mode: {args.mode}")
    if args.mode == "jdbc":
        resolved_ucan = (
            args.ucanaccess_lib.expanduser().resolve()
            if args.ucanaccess_lib
            else map_ipeds_vars.DEFAULT_UCAN
        )
        print(f"  UCanAccess lib: {resolved_ucan}")
    if args.mode == "csvmeta":
        print(
            "  CSV VARTABLE root: "
            f"{csv_vartable_root if csv_vartable_root else '(not provided)'}"
        )
        if csv_data_root:
            print(f"  CSV data root: {csv_data_root}")

    try:
        titles = map_ipeds_vars.read_titles(titles_path)
    except map_ipeds_vars.TitlesFileEmptyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    db_year_map = discover_databases(db_root)
    csvmeta_years = discover_csvmeta_years(csv_vartable_root) if args.mode == "csvmeta" else []
    available_years = sorted(set(db_year_map) | set(csvmeta_years))
    if not available_years:
        print("No IPEDS databases or metadata found to process.", file=sys.stderr)
        return 1

    selected_years, missing_years = parse_years_argument(args.years, available_years)
    if missing_years:
        logging.warning("Requested years not available: %s", ", ".join(str(y) for y in missing_years))
    if not selected_years:
        print("No overlapping years between requested and available datasets.", file=sys.stderr)
        return 1

    classpath: List[str] | None = None
    if args.mode == "jdbc":
        script_dir = Path(__file__).resolve().parent
        lib_dir = map_ipeds_vars.determine_ucanaccess_lib(
            str(args.ucanaccess_lib) if args.ucanaccess_lib else None,
            script_dir,
        )
        classpath = map_ipeds_vars.collect_jars(lib_dir)
    else:
        if not csv_vartable_root:
            print("CSV metadata mode requires --csv-vartable-root", file=sys.stderr)
            return 1

    all_records: List[map_ipeds_vars.VarRecord] = []
    all_missing: List[tuple[int, str]] = []

    for year in selected_years:
        if args.mode == "jdbc":
            records, missing = map_ipeds_vars.map_titles_for_year(
                year=year,
                titles=titles,
                db_root=db_root,
                classpath=classpath or [],
                use_fuzzy=False,
            )
        else:
            try:
                assert csv_vartable_root is not None
                vartable_records = load_vartable_records_from_csv(year, csv_vartable_root)
            except (FileNotFoundError, ValueError) as exc:
                logging.error("%s", exc)
                all_missing.extend((year, title) for title in titles)
                continue
            records, missing = map_ipeds_vars.match_titles(year, titles, vartable_records, use_fuzzy=False)
        all_records.extend(records)
        all_missing.extend(missing)
        logging.info(
            "%s: requested=%s, matched=%s, missing=%s",
            year,
            len(titles),
            len(records),
            len(missing),
        )

    if not all_records:
        print("No variable mappings were generated.", file=sys.stderr)
        return 1

    start_year, end_year = min(selected_years), max(selected_years)
    mapping_path = out_root / f"ipeds_var_map_{start_year}_{end_year}.csv"
    map_ipeds_vars.write_mapping_csv(all_records, mapping_path)
    missing_path = out_root / f"missing_titles_{start_year}_{end_year}.csv"
    map_ipeds_vars.write_missing_csv(all_missing, missing_path)
    logging.info("Wrote mapping to %s", mapping_path)

    year_table_mapping = group_records_by_year(all_records)

    total_exports = 0
    run_extraction = not (args.mode == "csvmeta" and csv_data_root is None)
    if not run_extraction:
        print(
            "CSV metadata mode skipped data extraction. Provide --csv-data-root to build the panel.",
            file=sys.stderr,
        )

    panel_path: Path | None = None
    if run_extraction:
        for year in selected_years:
            mapping = year_table_mapping.get(year, {})
            exports = extract_ipeds_data.export_tables_for_year(
                year=year,
                mapping=mapping,
                out_dir=out_root,
                db_root=db_root,
                classpath=classpath,
                csv_data_root=csv_data_root,
            )
            total_exports += len(exports)
            logging.info("Exported %s tables for %s", len(exports), year)

        panel_path = build_panel(out_root, selected_years, year_table_mapping)

    print("Summary:")
    print(f"  Years processed: {', '.join(str(year) for year in selected_years)}")
    print(f"  Titles requested: {len(titles)}")
    print(f"  Titles matched: {len(all_records)}")
    print(f"  Titles missing: {len(all_missing)}")
    print(f"  Tables exported: {total_exports}")
    if panel_path:
        print(f"  Panel path: {panel_path}")
    else:
        print("  Panel path: (skipped)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

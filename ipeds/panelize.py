"""Panelize IPEDS: discover DBs, map titles, extract per-year tables, build a panel CSV."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd

try:
    from ipeds import schema
except ImportError:  # pragma: no cover - adjust path when executed as script
    import sys
    from pathlib import Path as _Path

    package_root = _Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from ipeds import schema  # type: ignore

from map_ipeds_vars import (
    DEFAULT_DB_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_TITLES,
    DEFAULT_UCAN,
    collect_jars,
    connect_to_database,
    determine_ucanaccess_lib,
    infer_year_from_db,
)
from map_ipeds_vars import main as map_main
from extract_ipeds_data import build_wide_for_year, find_db_for_year, main as extract_main


def parse_years(years: str | None) -> List[int]:
    if not years:
        return list(range(2004, 2024))
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


def read_titles_file(path: Path) -> List[str]:
    titles: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            titles.append(cleaned)
    return titles


def _build_long_panel(out_dir: Path, start: int, end: int) -> Path | None:
    csvs = sorted(out_dir.glob("ipeds_*_*.csv"))
    if not csvs:
        return None
    frames = []
    for path in csvs:
        # Expect pattern: ipeds_<year>_<table>.csv
        m = re.match(r"ipeds_(\d{4})_(.+)\.csv$", path.name)
        source = m.group(2) if m else "unknown"
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:
            logging.warning("Skipping %s: %s", path.name, exc)
            continue
        df["source_table"] = source
        frames.append(df)
    if not frames:
        return None
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel_path = out_dir / f"ipeds_panel_{start}_{end}.csv"
    panel.to_csv(panel_path, index=False)
    return panel_path


def resolve_mapping_csv(out_dir: Path, start: int, end: int) -> Path | None:
    """Return the most relevant mapping CSV under out_dir."""
    candidates = [
        out_dir / f"ipeds_var_map_{start}_{end}.csv",
        out_dir / "ipeds_var_map_2004_2023.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    try:
        latest = max(
            (path for path in out_dir.glob("ipeds_var_map_*.csv") if path.is_file()),
            key=lambda item: item.stat().st_mtime,
        )
    except ValueError:
        return None
    except OSError as exc:  # pragma: no cover - defensive
        logging.debug("Unable to inspect mapping files in %s: %s", out_dir, exc)
        return None
    return latest


def discover_years(db_root: Path) -> List[int]:
    years: Set[int] = set()
    for path in db_root.rglob("IPEDS*.accdb"):
        try:
            y = infer_year_from_db(path)
        except Exception:
            continue
        years.add(y)
    ys = sorted(y for y in years if 1900 <= y <= 2100)
    return ys


def _invoke_map_main(argv: Sequence[str]) -> int:
    """Call map_main and translate SystemExit into an exit code."""
    try:
        return map_main(list(argv))
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 1


def _invoke_extract_main(argv: Sequence[str]) -> int:
    """Call extract_main and translate SystemExit into an exit code."""
    try:
        return extract_main(list(argv))
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT, help="Root with IPEDS *.accdb files")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output directory root")
    parser.add_argument("--titles", type=Path, default=DEFAULT_TITLES, help="Titles file path")
    parser.add_argument("--years", type=str, default=None, help="Optional year range or comma list to restrict")
    parser.add_argument("--ucanaccess-lib", type=str, default=str(DEFAULT_UCAN), help="UCanAccess JARs dir")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["jdbc", "csvmeta", "excel_then_jdbc"],
        default="jdbc",
        help="Mapping mode: JDBC, CSV metadata, or CSV metadata with JDBC fallback.",
    )
    parser.add_argument(
        "--csv-vartable-root",
        type=Path,
        default=None,
        help="CSV root for VARTABLE##.csv (legacy flag; alias for --metadata-csv-root).",
    )
    parser.add_argument(
        "--metadata-csv-root",
        type=Path,
        default=None,
        help="Alias for --csv-vartable-root (CSV metadata root).",
    )
    parser.add_argument(
        "--tables-csv-root",
        type=Path,
        default=None,
        help="Optional root containing exported component tables (<root>/<year>/<tableName>.csv).",
    )
    parser.add_argument(
        "--wide-column-source",
        type=str,
        choices=["title", "varname"],
        default="title",
        help="Column naming strategy for wide outputs ('title' or 'varname').",
    )
    parser.add_argument("--skip-wide", action="store_true", help="Skip building wide panel output.")
    parser.add_argument(
        "--long-panel",
        action="store_true",
        help="Also build the legacy long-form panel (concatenated per-table CSVs).",
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy title matching during the mapping step.",
    )
    parser.add_argument(
        "--single-output",
        dest="single_output",
        action="store_true",
        default=True,
        help="Write only the combined wide panel CSV (default).",
    )
    parser.add_argument(
        "--per-year-wide",
        dest="single_output",
        action="store_false",
        help="Also write per-year wide CSVs alongside the panel.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    db_root = args.db_root.expanduser().resolve()
    titles_path = args.titles.expanduser().resolve()

    csv_meta_root: Path | None = None
    if args.csv_vartable_root:
        csv_meta_root = args.csv_vartable_root.expanduser().resolve()
    if args.metadata_csv_root:
        metadata_root = args.metadata_csv_root.expanduser().resolve()
        if csv_meta_root and metadata_root != csv_meta_root:
            logging.warning(
                "Both --csv-vartable-root and --metadata-csv-root were provided; using %s", metadata_root
            )
        csv_meta_root = metadata_root

    tables_csv_root = args.tables_csv_root.expanduser().resolve() if args.tables_csv_root else None
    script_dir = Path(__file__).resolve().parent
    ucanaccess_lib = determine_ucanaccess_lib(args.ucanaccess_lib, script_dir)

    if not titles_path.exists():
        logging.error("Titles file not found: %s", titles_path)
        return 1

    discovered = discover_years(db_root) if db_root.exists() else []
    years: List[int] = []
    if args.years:
        requested = parse_years(args.years)
        if discovered:
            discovered_set = set(discovered)
            years = [year for year in requested if year in discovered_set]
            missing = sorted(set(requested) - set(years))
            if missing:
                logging.warning(
                    "Requested years not found under %s: %s", db_root, ", ".join(map(str, missing))
                )
        else:
            years = requested
    else:
        years = discovered

    if not years:
        logging.error("No years available; supply --years or ensure the databases are present.")
        return 1

    start, end = years[0], years[-1]

    logging.info("Resolved paths:")
    logging.info("  db-root: %s", db_root)
    logging.info("  out-root: %s", out_root)
    logging.info("  titles: %s", titles_path)
    if csv_meta_root:
        logging.info("  csv metadata root: %s", csv_meta_root)
    if tables_csv_root:
        logging.info("  tables csv root: %s", tables_csv_root)
    if args.mode in {"jdbc", "excel_then_jdbc"}:
        logging.info("  ucanaccess-lib: %s", ucanaccess_lib)
    logging.info("Years: %s", ", ".join(map(str, years)))

    def run_jdbc_mapping() -> int:
        argv = [
            "--db-dir",
            str(db_root),
            "--out-dir",
            str(out_root),
            "--titles",
            str(titles_path),
            "--years",
            f"{start}-{end}",
            "--ucanaccess-lib",
            str(ucanaccess_lib),
        ]
        if args.fuzzy:
            argv.append("--fuzzy")
        return _invoke_map_main(argv)

    def run_csvmeta_mapping() -> int:
        if not csv_meta_root:
            logging.error("CSV metadata mode requires --metadata-csv-root (or --csv-vartable-root).")
            return 1
        argv = [
            "--db-dir",
            str(db_root),
            "--out-dir",
            str(out_root),
            "--titles",
            str(titles_path),
            "--years",
            f"{start}-{end}",
            "--csv-vartable-root",
            str(csv_meta_root),
        ]
        if args.fuzzy:
            argv.append("--fuzzy")
        return _invoke_map_main(argv)

    map_rc = 0
    map_mode_used: str | None = None
    if args.mode == "jdbc":
        map_rc = run_jdbc_mapping()
        map_mode_used = "jdbc"
    elif args.mode == "csvmeta":
        map_rc = run_csvmeta_mapping()
        map_mode_used = "csvmeta"
    else:  # excel_then_jdbc
        if csv_meta_root:
            map_rc = run_csvmeta_mapping()
            if map_rc == 0:
                map_mode_used = "csvmeta"
            else:
                logging.warning(
                    "CSV metadata mapping failed with exit code %s; falling back to JDBC.", map_rc
                )
        else:
            logging.info("No CSV metadata root provided; using JDBC mapping.")
        if map_mode_used is None:
            map_rc = run_jdbc_mapping()
            map_mode_used = "jdbc"

    if map_rc != 0:
        logging.error("Mapper failed with exit code %s", map_rc)
        return map_rc

    map_csv_path = resolve_mapping_csv(out_root, start, end)
    if not map_csv_path:
        logging.error("Mapping CSV not found in %s", out_root)
        return 1

    run_extraction = args.long_panel and args.mode in {"jdbc", "excel_then_jdbc"}
    if run_extraction:
        ext_rc = _invoke_extract_main(
            (
                "--db-dir",
                str(db_root),
                "--out-dir",
                str(out_root),
                "--map-csv",
                str(map_csv_path),
                "--years",
                ",".join(map(str, years)),
                "--ucanaccess-lib",
                str(ucanaccess_lib),
            )
        )
        if ext_rc != 0:
            logging.error("Extractor failed with exit code %s", ext_rc)
            return ext_rc
    elif args.long_panel:
        logging.warning("Legacy long panel requested, but extractor was not run (non-JDBC mode).")

    if args.long_panel:
        long_panel = _build_long_panel(out_root, start, end)
        if long_panel:
            logging.info("Legacy long panel written: %s", long_panel)
        else:
            logging.warning("Legacy long panel not created (no per-table CSVs found).")

    if args.skip_wide:
        logging.info("Skipping wide panel build (--skip-wide set).")
        return 0

    try:
        map_df = pd.read_csv(map_csv_path, dtype=str)
    except FileNotFoundError:
        logging.error("Mapping CSV missing at %s", map_csv_path)
        return 1

    required_cols = {"year", "canonical", "tableName", "varName"}
    if not required_cols.issubset(map_df.columns):
        missing_cols = required_cols - set(map_df.columns)
        logging.error("Mapping CSV missing required columns: %s", ", ".join(sorted(missing_cols)))
        return 1

    map_df = map_df.replace({pd.NA: ""}).fillna("")
    map_df = map_df.dropna(subset=["canonical", "tableName", "varName"], how="any")

    if args.mode == "csvmeta" and tables_csv_root is None:
        logging.error("CSV metadata mode requires --tables-csv-root to build wide outputs.")
        return 1

    classpath: List[str] | None = None
    results = []
    coverage_records: List[Dict[str, object]] = []
    component_year_rows: Set[Tuple[str, int, int, int]] = set()
    canonical_source_records: List[Dict[str, object]] = []
    finance_form_lookup: Dict[Tuple[int, str], str] = {}

    for year in years:
        connection = None
        db_path: Path | None = None
        if args.mode in {"jdbc", "excel_then_jdbc"}:
            db_path = find_db_for_year(db_root, year)
            if db_path and classpath is None:
                try:
                    classpath = collect_jars(ucanaccess_lib)
                except FileNotFoundError as exc:
                    if tables_csv_root is None:
                        logging.error("UCanAccess library not found: %s", exc)
                        return 1
                    logging.warning(
                        "UCanAccess library not available (%s); relying on tables CSV fallback.", exc
                    )
                    classpath = None
            if db_path and classpath:
                try:
                    connection = connect_to_database(db_path, classpath)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to connect to %s: %s", db_path, exc)
                    if tables_csv_root is None:
                        logging.error(
                            "Cannot build wide output for %s without database access or tables CSVs.",
                            year,
                        )
                        return 1
                    connection = None
            elif db_path is None and tables_csv_root is None:
                logging.warning("Database for %s not found and no tables CSV fallback; skipping.", year)
                continue

        try:
            result = build_wide_for_year(
                map_df,
                year,
                connection=connection,
                tables_csv_root=tables_csv_root,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Wide build failed for %s: %s", year, exc)
            return 1
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:  # pragma: no cover - defensive
                    pass

        if result.data.empty and not result.coverage:
            logging.warning("Wide build produced no data for %s.", year)
        else:
            logging.info("Wide build complete for %s: %s columns", year, len(result.data.columns))

        if not args.single_output and not result.data.empty:
            year_path = out_root / f"ipeds_wide_source_{year}.csv"
            temp = result.data.copy()
            temp.insert(0, "source_year", year)
            temp.to_csv(year_path, index=False)
            logging.info("Per-year wide written: %s", year_path)

        total_units = result.data.shape[0]
        for canonical, non_null in result.coverage.items():
            offset = schema.panel_year_offset(canonical)
            coverage_records.append(
                {
                    "source_year": year,
                    "panel_year": year + offset,
                    "canonical": canonical,
                    "non_null": non_null,
                    "total_units": total_units,
                }
            )
        for canonical, sources in result.canonical_sources.items():
            offset = schema.panel_year_offset(canonical)
            component_year_rows.add((canonical, year, year + offset, offset))
            if sources:
                unique_sources = sorted({f"{table}.{var}" for table, var in sources})
                canonical_source_records.append(
                    {
                        "source_year": year,
                        "canonical": canonical,
                        "sources": ";".join(unique_sources),
                    }
                )

        finance_offset = schema.GROUP_YEAR_OFFSETS.get("finance_revenue", -1)
        for unitid, form in result.finance_forms.items():
            key = (year + finance_offset, str(unitid))
            finance_form_lookup[key] = form

        results.append(result)

    if not results:
        logging.warning("No wide data produced; skipping panel creation.")
        return 0

    panel_rows: Dict[Tuple[int, str], Dict[str, object]] = {}
    for result in results:
        data = result.data
        if data.empty or "UNITID" not in data.columns:
            continue
        for canonical in [col for col in data.columns if col != "UNITID"]:
            offset = schema.panel_year_offset(canonical)
            panel_year = result.year + offset
            component_year_rows.add((canonical, result.year, panel_year, offset))
            slice_df = data[["UNITID", canonical]].copy()
            for unitid, value in zip(slice_df["UNITID"], slice_df[canonical]):
                unitid = str(unitid)
                if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                    continue
                key = (panel_year, unitid)
                record = panel_rows.setdefault(key, {"UNITID": unitid, "year": panel_year})
                existing = record.get(canonical)
                if existing is None or (isinstance(existing, float) and pd.isna(existing)) or existing == "":
                    record[canonical] = value
                if schema.is_finance_canonical(canonical):
                    form = result.finance_forms.get(unitid)
                    if form:
                        record["finance_form_type"] = form

    for key, form in finance_form_lookup.items():
        record = panel_rows.setdefault(key, {"UNITID": key[1], "year": key[0]})
        record.setdefault("finance_form_type", form)

    if not panel_rows:
        logging.warning("Panel rows were empty after processing; nothing to write.")
        return 0

    panel_df = pd.DataFrame(panel_rows.values())
    canonical_cols = schema.canonical_columns()
    for canonical in canonical_cols:
        if canonical not in panel_df.columns:
            panel_df[canonical] = pd.NA
    if "finance_parent_child_indicator" not in panel_df.columns:
        panel_df["finance_parent_child_indicator"] = pd.NA
    if "finance_accounting_standard" not in panel_df.columns:
        panel_df["finance_accounting_standard"] = pd.NA

    column_order = ["year", "UNITID", *canonical_cols]
    for column in column_order:
        if column not in panel_df.columns:
            panel_df[column] = pd.NA
    panel_df = panel_df.reindex(columns=column_order)
    panel_df["year"] = panel_df["year"].astype(int)
    panel_df["UNITID"] = panel_df["UNITID"].astype(str)
    panel_df = panel_df.sort_values(["year", "UNITID"]).reset_index(drop=True)

    panel_path = out_root / f"ipeds_panel_{start}_{end}.csv"
    panel_df.to_csv(panel_path, index=False)
    logging.info("Wide panel written: %s (rows=%s, columns=%s)", panel_path, len(panel_df), len(panel_df.columns))

    diagnostics_dir = out_root

    if coverage_records:
        coverage_df = pd.DataFrame(coverage_records)
        coverage_df.sort_values(["canonical", "source_year"], inplace=True)
        coverage_df.to_csv(diagnostics_dir / "coverage_by_year.csv", index=False)

    if canonical_source_records:
        canonical_sources_df = pd.DataFrame(canonical_source_records).drop_duplicates()
        canonical_sources_df.sort_values(["canonical", "source_year"], inplace=True)
        canonical_sources_df.to_csv(diagnostics_dir / "canonical_sources.csv", index=False)

    if component_year_rows:
        component_df = pd.DataFrame(
            sorted(component_year_rows),
            columns=["canonical", "source_year", "panel_year", "offset"],
        )
        component_df.to_csv(diagnostics_dir / "component_year_map.csv", index=False)

    if finance_form_lookup:
        finance_form_df = pd.DataFrame(
            (
                {"panel_year": key[0], "UNITID": key[1], "finance_form_type": value}
                for key, value in finance_form_lookup.items()
            )
        )
        finance_form_df.sort_values(["panel_year", "UNITID"], inplace=True)
        finance_form_df.to_csv(diagnostics_dir / "finance_form_by_unitid.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

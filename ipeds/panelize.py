"""Panelize IPEDS: discover DBs, map titles, extract per-year tables, build a panel CSV."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Sequence, Set

import pandas as pd

from map_ipeds_vars import (
    DEFAULT_WORKSPACE,
    DEFAULT_DB_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_TITLES,
    DEFAULT_UCAN,
    infer_year_from_db,
)
from map_ipeds_vars import main as map_main
from extract_ipeds_data import main as extract_main


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


def build_panel(out_dir: Path, start: int, end: int) -> Path | None:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT, help="Root with IPEDS *.accdb files")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output directory root")
    parser.add_argument("--titles", type=Path, default=DEFAULT_TITLES, help="Titles file path")
    parser.add_argument("--years", type=str, default=None, help="Optional year range or comma list to restrict")
    parser.add_argument("--ucanaccess-lib", type=str, default=str(DEFAULT_UCAN), help="UCanAccess JARs dir")
    parser.add_argument("--mode", type=str, choices=["jdbc", "csvmeta"], default="jdbc", help="Mapping mode")
    parser.add_argument("--csv-vartable-root", type=Path, default=None, help="CSV root for VARTABLE##.csv (csvmeta mode)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    db_root = args.db_root.expanduser().resolve()
    titles = args.titles.expanduser().resolve()

    # Discover available years from DB filenames
    discovered = discover_years(db_root)
    if args.years:
        requested = parse_years(args.years)
        years = [y for y in discovered if y in set(requested)]
    else:
        years = discovered
    if not years:
        logging.error("No years discovered under %s (or restricted by --years).", db_root)
        return 1
    start, end = years[0], years[-1]

    logging.info("Resolved paths:")
    logging.info("  db-root: %s", db_root)
    logging.info("  out-root: %s", out_root)
    logging.info("  titles: %s", titles)
    if args.mode == "jdbc":
        logging.info("  ucanaccess-lib: %s", args.ucanaccess_lib)
    else:
        logging.info("  csv-vartable-root: %s", args.csv_vartable_root)
    logging.info("Years: %s", ", ".join(map(str, years)))

    # Step A: Build mapping
    if args.mode == "jdbc":
        map_rc = map_main(
            [
                "--db-dir",
                str(db_root),
                "--out-dir",
                str(out_root),
                "--titles",
                str(titles),
                "--years",
                f"{start}-{end}",
                "--ucanaccess-lib",
                str(args.ucanaccess_lib),
            ]
        )
    else:
        if not args.csv_vartable_root:
            logging.error("csvmeta mode requires --csv-vartable-root")
            return 1
        map_rc = map_main(
            [
                "--db-dir",
                str(db_root),
                "--out-dir",
                str(out_root),
                "--titles",
                str(titles),
                "--years",
                f"{start}-{end}",
                "--csv-vartable-root",
                str(args.csv_vartable_root),
            ]
        )
    if map_rc != 0:
        logging.error("Mapper failed with exit code %s", map_rc)
        return map_rc

    # Step B: Extract per-year tables (JDBC only)
    if args.mode == "jdbc":
        ext_rc = extract_main(
            [
                "--db-dir",
                str(db_root),
                "--out-dir",
                str(out_root),
                "--map-csv",
                str(out_root / f"ipeds_var_map_{start}_{end}.csv"),
                "--years",
                ",".join(map(str, years)),
                "--ucanaccess-lib",
                str(args.ucanaccess_lib),
            ]
        )
        if ext_rc != 0:
            logging.error("Extractor failed with exit code %s", ext_rc)
            return ext_rc
    else:
        logging.info("csvmeta mode: mapping created from CSV metadata; extraction requires JDBC or a CSV data fallback.")

    # Step C: Build simple panel if exports exist
    panel_path = build_panel(out_root, start, end)
    if panel_path:
        logging.info("Panel written: %s", panel_path)
    else:
        logging.info("Panel not created (no per-table CSVs found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Panelize IPEDS: mapping once, extract per-year tables, optionally build a panel CSV."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from map_ipeds_vars import (
    DEFAULT_DB_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_TITLES,
    DEFAULT_UCAN,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=str, default="2004-2023", help="Year range or comma list")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="Directory with IPEDS databases")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--titles", type=Path, default=DEFAULT_TITLES, help="Titles file path")
    parser.add_argument("--ucanaccess-lib", type=str, default=str(DEFAULT_UCAN), help="UCanAccess JARs dir")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_dir = args.db_dir.expanduser().resolve()
    titles = args.titles.expanduser().resolve()
    years = parse_years(args.years)
    if not years:
        logging.error("No valid years parsed from %s", args.years)
        return 1
    start, end = years[0], years[-1]

    logging.info("Resolved paths:")
    logging.info("  db-dir: %s", db_dir)
    logging.info("  out-dir: %s", out_dir)
    logging.info("  titles: %s", titles)
    logging.info("  ucanaccess-lib: %s", args.ucanaccess_lib)

    # Run mapper once (JDBC path by default here)
    map_rc = map_main(
        [
            "--db-dir",
            str(db_dir),
            "--out-dir",
            str(out_dir),
            "--titles",
            str(titles),
            "--years",
            f"{start}-{end}",
            "--ucanaccess-lib",
            str(args.ucanaccess_lib),
        ]
    )
    if map_rc != 0:
        logging.error("Mapper failed with exit code %s", map_rc)
        return map_rc

    # Extract per-year tables
    ext_rc = extract_main(
        [
            "--db-dir",
            str(db_dir),
            "--out-dir",
            str(out_dir),
            "--map-csv",
            str(out_dir / f"ipeds_var_map_{start}_{end}.csv"),
            "--years",
            f"{start}-{end}",
        ]
    )
    if ext_rc != 0:
        logging.error("Extractor failed with exit code %s", ext_rc)
        return ext_rc

    # Build simple panel
    panel_path = build_panel(out_dir, start, end)
    if panel_path:
        logging.info("Panel written: %s", panel_path)
    else:
        logging.info("Panel not created (no per-table CSVs found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


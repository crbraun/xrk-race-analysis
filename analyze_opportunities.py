"""Identify per-driver improvement opportunities for each race.

Builds on the same parsing, lap classification, and driver assignments as
``analyze.py``, but writes opportunity-focused outputs instead of the summary
pace CSV.

Usage:
    python analyze_opportunities.py [--input FOLDER] [--reuse-assignments]

Outputs:
    opportunities_summary.csv — one row per driver per race
    opportunities_laps.csv    — flagged laps (outliers, early pace, traffic)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze import DEFAULT_ASSIGNMENTS, DEFAULT_INPUT, load_and_classify
from driver_id import load_assignments
from lap_analysis import Thresholds
from opportunities import OpportunityConfig, analyze_opportunities

DEFAULT_SUMMARY = "opportunities_summary.csv"
DEFAULT_LAPS = "opportunities_laps.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Folder of .xrk files")
    p.add_argument(
        "--assignments",
        default=DEFAULT_ASSIGNMENTS,
        help="Driver assignments file (same format as analyze.py).",
    )
    p.add_argument(
        "--summary-output",
        default=DEFAULT_SUMMARY,
        help="Per-driver per-race opportunity summary CSV.",
    )
    p.add_argument(
        "--laps-output",
        default=DEFAULT_LAPS,
        help="Flagged opportunity laps CSV.",
    )
    p.add_argument(
        "--outlier-vs-best-s",
        type=float,
        default=3.0,
        help="Flag hot laps at least this many seconds off driver best.",
    )
    p.add_argument(
        "--outlier-vs-best-ratio",
        type=float,
        default=1.05,
        help="Flag hot laps at least this ratio vs driver best (default 1.05).",
    )
    p.add_argument(
        "--vsc-speed-mph",
        type=float,
        default=35.0,
        help="VSC speed limit in mph (default: 35).",
    )
    p.add_argument(
        "--vsc-margin-mph",
        type=float,
        default=8.0,
        help="Mph above VSC speed allowed on mean lap speed before not VSC.",
    )
    p.add_argument(
        "--formation-restart-ratio",
        type=float,
        default=1.05,
        help="Restart formation if lap after VSC exceeds race best hot x this.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    folder = Path(args.input).expanduser()
    assignments_path = Path(args.assignments).expanduser()

    if not assignments_path.exists():
        raise SystemExit(
            f"Assignments file not found: {assignments_path}\n"
            "Run analyze.py --list-sessions first, tag drivers, then re-run."
        )

    cfg = Thresholds(
        vsc_speed_mph=args.vsc_speed_mph,
        vsc_margin_mph=args.vsc_margin_mph,
        formation_restart_ratio=args.formation_restart_ratio,
    )
    opp_cfg = OpportunityConfig(
        outlier_vs_best_s=args.outlier_vs_best_s,
        outlier_vs_best_ratio=args.outlier_vs_best_ratio,
    )

    print(f"Parsing XRK files in {folder} ...")
    sessions = load_and_classify(folder, cfg)
    drivers = load_assignments(assignments_path)
    print(f"Parsed {len(sessions)} sessions.")

    summary_df, laps_df = analyze_opportunities(sessions, drivers, opp_cfg)

    summary_path = Path(args.summary_output).expanduser()
    laps_path = Path(args.laps_output).expanduser()
    summary_df.to_csv(summary_path, index=False)
    laps_df.to_csv(laps_path, index=False)

    print(f"\nWrote {len(summary_df)} summary rows to {summary_path}")
    print(f"Wrote {len(laps_df)} flagged laps to {laps_path}\n")

    if not summary_df.empty:
        display_cols = [
            "driver",
            "race",
            "best_lap",
            "gap_to_race_best",
            "potential_gain_to_median",
            "outlier_hot_laps",
            "early_pace_laps",
            "traffic_laps",
            "insight",
        ]
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""Analyze a folder of AiM XRK files for a race weekend.

Pipeline:
  1. Parse every .xrk file (session datetime, laps, GPS speed, telemetry).
  2. Classify laps (hot vs VSC/safety vs out/in/pit) using GPS speed.
  3. Attribute each stint (file) to a driver via an editable assignments file
     that lists every stint with its start/end time and best lap; names can be
     prepopulated from the file note. Edit it all at once, save, and re-run.
  4. Aggregate per driver per race and write a CSV.

Usage:
    python analyze.py [--input FOLDER] [--output CSV]
                      [--list-sessions] [--reuse-assignments] [--template-only]
                      [--profile] [--suggest-clusters] [--drivers N]

CSV columns (one row per driver per race):
    driver, race, date, best_lap_s, best_lap, avg_hot_lap_s, avg_hot_lap,
    median_hot_lap_s, median_hot_lap, p90_hot_lap_s, p90_hot_lap,
    total_time_in_car_s, total_time_in_car, total_laps, hot_laps, vsc_laps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from driver_id import (
    assign_drivers,
    build_prefill,
    load_assignments,
    print_session_list,
    session_summaries,
    write_assignments_file,
    write_session_list,
)
from lap_analysis import (
    HOT,
    VSC,
    Thresholds,
    classify_race_context,
    classify_session,
    is_stat_lap,
)
from xrk_parser import Session, find_xrk_files, parse_file

DEFAULT_INPUT = "~/Documents/Summit Point"
DEFAULT_OUTPUT = "race_analysis.csv"
DEFAULT_ASSIGNMENTS = "driver_assignments.txt"
DEFAULT_SESSIONS_CSV = "sessions.csv"


def _fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return ""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:06.3f}"


def _fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return ""
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def load_and_classify(folder: Path, cfg: Thresholds) -> list[Session]:
    """Parse and classify every file, using a race-level green pace reference."""
    files = find_xrk_files(folder)
    if not files:
        raise SystemExit(f"No .xrk files found in {folder}")

    sessions = [parse_file(f) for f in files]
    for s in sessions:
        classify_session(s, cfg)
    classify_race_context(sessions, cfg)
    return sessions


def aggregate(sessions: list[Session], drivers: dict[str, str]) -> pd.DataFrame:
    """Aggregate per (driver, race).

    Best / avg / median / P90 use ``hot`` laps only. Excluded from those
    stats: VSC, formation, out/in/pit, traffic (slow), and incomplete laps.
    """
    records: dict[tuple[str, str], dict] = {}

    for s in sessions:
        driver = drivers.get(s.file_name, "Unassigned")
        key = (driver, s.race)
        rec = records.setdefault(
            key,
            {
                "driver": driver,
                "race": s.race,
                "dates": set(),
                "hot_laps_s": [],
                "total_time_s": 0.0,
                "total_laps": 0,
                "vsc_laps": 0,
            },
        )
        if s.start_dt:
            rec["dates"].add(s.start_dt.date())
        rec["total_time_s"] += s.duration_s
        for lap in s.laps:
            rec["total_laps"] += 1
            if is_stat_lap(lap):
                rec["hot_laps_s"].append(lap.lap_time_s)
            elif lap.lap_type == VSC:
                rec["vsc_laps"] += 1

    rows = []
    for rec in records.values():
        hot = rec["hot_laps_s"]
        best = min(hot) if hot else float("nan")
        avg = float(np.mean(hot)) if hot else float("nan")
        median = float(np.median(hot)) if hot else float("nan")
        p90 = float(np.percentile(hot, 90)) if hot else float("nan")
        date_str = ", ".join(d.strftime("%m/%d/%Y") for d in sorted(rec["dates"]))
        rows.append(
            {
                "driver": rec["driver"],
                "race": rec["race"],
                "date": date_str,
                "best_lap_s": round(best, 3) if np.isfinite(best) else "",
                "best_lap": _fmt_time(best),
                "avg_hot_lap_s": round(avg, 3) if np.isfinite(avg) else "",
                "avg_hot_lap": _fmt_time(avg),
                "median_hot_lap_s": round(median, 3) if np.isfinite(median) else "",
                "median_hot_lap": _fmt_time(median),
                "p90_hot_lap_s": round(p90, 3) if np.isfinite(p90) else "",
                "p90_hot_lap": _fmt_time(p90),
                "total_time_in_car_s": round(rec["total_time_s"], 1),
                "total_time_in_car": _fmt_duration(rec["total_time_s"]),
                "total_laps": rec["total_laps"],
                "hot_laps": len(hot),
                "vsc_laps": rec["vsc_laps"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["race", "best_lap_s"], na_position="last").reset_index(
            drop=True
        )
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Folder of .xrk files")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    p.add_argument(
        "--assignments",
        default=DEFAULT_ASSIGNMENTS,
        help="Editable file mapping each stint file -> driver name.",
    )
    p.add_argument(
        "--sessions-csv",
        default=DEFAULT_SESSIONS_CSV,
        help="Where to write the per-session list (start/end + best lap).",
    )
    p.add_argument(
        "--reuse-assignments",
        action="store_true",
        help="Read the assignments file as-is without opening an editor.",
    )
    p.add_argument(
        "--list-sessions",
        action="store_true",
        help="Only write the session list + editable assignments file, then exit.",
    )
    p.add_argument(
        "--template-only",
        action="store_true",
        help="Write the editable assignments file but do not open an editor.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Suggest untagged stints from driving-style profiles built from "
        "stints you have already tagged in the assignments file.",
    )
    p.add_argument(
        "--profile-distance",
        type=float,
        default=2.5,
        help="Max fingerprint distance to accept a profile match (default: 2.5).",
    )
    p.add_argument(
        "--suggest-clusters",
        action="store_true",
        help="Pre-fill suggested Driver A/B/... names by clustering driving style.",
    )
    p.add_argument(
        "--drivers",
        type=int,
        default=None,
        help="Hint for number of drivers when using --suggest-clusters.",
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
    cfg = Thresholds(
        vsc_speed_mph=args.vsc_speed_mph,
        vsc_margin_mph=args.vsc_margin_mph,
        formation_restart_ratio=args.formation_restart_ratio,
    )

    print(f"Parsing XRK files in {folder} ...")
    sessions = load_and_classify(folder, cfg)
    print(f"Parsed {len(sessions)} sessions.")

    sessions_csv = Path(args.sessions_csv).expanduser()
    assignments_path = Path(args.assignments).expanduser()

    if args.list_sessions:
        rows = session_summaries(sessions)
        write_session_list(rows, sessions_csv)
        print_session_list(rows)
        prefill = build_prefill(sessions, load_assignments(assignments_path))
        write_assignments_file(rows, prefill, assignments_path)
        print(
            f"\nEdit {assignments_path} (one driver name per stint), then run "
            "again with --reuse-assignments to build the analysis CSV."
        )
        return

    drivers = assign_drivers(
        sessions,
        assignments_path=assignments_path,
        sessions_csv=sessions_csv,
        reuse=args.reuse_assignments,
        suggest_clusters=args.suggest_clusters,
        use_profile=args.profile,
        profile_distance=args.profile_distance,
        n_drivers=args.drivers,
        template_only=args.template_only,
    )

    df = aggregate(sessions, drivers)
    out_path = Path(args.output).expanduser()
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} rows to {out_path}\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

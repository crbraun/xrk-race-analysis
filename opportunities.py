"""Identify per-driver improvement opportunities within each race.

An opportunity is recoverable or explainable time relative to the driver's own
best pace, their typical (median) pace, or the race's best lap. Outputs:

  * summary — one row per driver per race with aggregate opportunity metrics
  * laps    — individual laps flagged as outliers, traffic, or early pace
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lap_analysis import FORMATION, HOT, SLOW, VSC, is_stat_lap
from xrk_parser import Lap, Session


@dataclass
class OpportunityConfig:
    """Thresholds for flagging laps and summarising opportunities."""

    outlier_vs_best_s: float = 3.0
    outlier_vs_best_ratio: float = 1.05
    early_pace_hot_lap_limit: int = 12


def _fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return ""
    if seconds == 0:
        return "0:00.000"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:06.3f}"


def _stint_id(file_name: str) -> str:
    if "_a_" in file_name:
        return file_name.rsplit("_a_", 1)[-1].replace(".xrk", "")
    return file_name


def _race_best(sessions: list[Session]) -> float:
    times = [
        l.lap_time_s
        for s in sessions
        for l in s.laps
        if is_stat_lap(l)
    ]
    return min(times) if times else float("nan")


def _driver_sessions(
    sessions: list[Session], drivers: dict[str, str], driver: str, race: str
) -> list[Session]:
    group = [
        s
        for s in sessions
        if s.race == race and drivers.get(s.file_name, "Unassigned") == driver
    ]
    group.sort(key=lambda s: s.start_dt or pd.Timestamp.min)
    return group


def _is_outlier_hot(lap: Lap, best: float, cfg: OpportunityConfig) -> bool:
    if not is_stat_lap(lap) or not np.isfinite(best):
        return False
    delta = lap.lap_time_s - best
    return delta >= cfg.outlier_vs_best_s or lap.lap_time_s >= best * cfg.outlier_vs_best_ratio


def _is_early_pace(
    hot_index: int,
    stint_is_first: bool,
    lap: Lap,
    best: float,
    cfg: OpportunityConfig,
) -> bool:
    if not stint_is_first or not is_stat_lap(lap) or not np.isfinite(best):
        return False
    if hot_index >= cfg.early_pace_hot_lap_limit:
        return False
    return lap.lap_time_s >= best * cfg.outlier_vs_best_ratio


def _build_insight(
    *,
    outlier_count: int,
    outlier_lost_s: float,
    early_pace_count: int,
    early_pace_lost_s: float,
    traffic_count: int,
    traffic_time_s: float,
    consistency_gap_median_s: float,
    consistency_gap_p90_s: float,
    gap_to_race_best_s: float,
) -> str:
    parts: list[str] = []

    if outlier_count and outlier_lost_s > 0:
        parts.append(
            f"{outlier_count} hot laps averaged "
            f"{outlier_lost_s / outlier_count:.1f}s off best "
            f"({outlier_lost_s:.0f}s total)"
        )
    if early_pace_count and early_pace_lost_s > 0:
        parts.append(
            f"{early_pace_count} early-race hot laps cost "
            f"{early_pace_lost_s:.0f}s vs personal best"
        )
    if traffic_count:
        parts.append(f"{traffic_count} traffic laps ({traffic_time_s:.0f}s)")
    if np.isfinite(consistency_gap_p90_s) and consistency_gap_p90_s >= 5:
        parts.append(f"P90 is {_fmt_time(consistency_gap_p90_s)} slower than best")
    elif np.isfinite(consistency_gap_median_s) and consistency_gap_median_s >= 2:
        parts.append(f"median is {_fmt_time(consistency_gap_median_s)} slower than best")
    if np.isfinite(gap_to_race_best_s) and gap_to_race_best_s > 0.5:
        parts.append(f"best lap is {_fmt_time(gap_to_race_best_s)} off race best")

    return "; ".join(parts[:3]) if parts else "Limited hot-lap sample"


def analyze_opportunities(
    sessions: list[Session],
    drivers: dict[str, str],
    cfg: OpportunityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, laps_df) of driver opportunities per race."""
    cfg = cfg or OpportunityConfig()
    races = sorted({s.race for s in sessions})
    summary_rows: list[dict] = []
    lap_rows: list[dict] = []

    for race in races:
        race_sessions = [s for s in sessions if s.race == race]
        race_best = _race_best(race_sessions)
        race_drivers = sorted(
            {
                drivers.get(s.file_name, "Unassigned")
                for s in race_sessions
                if drivers.get(s.file_name, "Unassigned") != "Unassigned"
            }
        )

        for driver in race_drivers:
            driver_stints = _driver_sessions(sessions, drivers, driver, race)
            hot_times = [
                l.lap_time_s
                for s in driver_stints
                for l in s.laps
                if is_stat_lap(l)
            ]
            if not hot_times:
                continue

            driver_best = min(hot_times)
            median = float(np.median(hot_times))
            p90 = float(np.percentile(hot_times, 90))
            avg = float(np.mean(hot_times))

            outlier_count = 0
            outlier_lost_s = 0.0
            early_pace_count = 0
            early_pace_lost_s = 0.0
            traffic_count = 0
            traffic_time_s = 0.0
            formation_count = 0
            vsc_count = 0
            total_lost_vs_best_s = 0.0
            total_lost_vs_median_s = 0.0

            for stint_i, stint in enumerate(driver_stints):
                stint_hot_idx = 0
                for lap in stint.laps:
                    if lap.lap_type == SLOW:
                        traffic_count += 1
                        traffic_time_s += lap.lap_time_s
                        lap_rows.append(
                            _lap_row(
                                driver,
                                race,
                                stint,
                                lap,
                                driver_best,
                                race_best,
                                "traffic",
                                cfg,
                            )
                        )
                    elif lap.lap_type == FORMATION:
                        formation_count += 1
                    elif lap.lap_type == VSC:
                        vsc_count += 1

                    if is_stat_lap(lap):
                        lost_best = max(0.0, lap.lap_time_s - driver_best)
                        lost_median = max(0.0, lap.lap_time_s - median)
                        total_lost_vs_best_s += lost_best
                        total_lost_vs_median_s += lost_median

                        is_early = _is_early_pace(
                            stint_hot_idx,
                            stint_is_first=(stint_i == 0),
                            lap=lap,
                            best=driver_best,
                            cfg=cfg,
                        )
                        is_outlier = _is_outlier_hot(lap, driver_best, cfg)

                        if is_early:
                            early_pace_count += 1
                            early_pace_lost_s += lost_best
                            category = "early_pace"
                        elif is_outlier:
                            outlier_count += 1
                            outlier_lost_s += lost_best
                            category = "outlier_hot"
                        else:
                            category = ""

                        if category:
                            lap_rows.append(
                                _lap_row(
                                    driver,
                                    race,
                                    stint,
                                    lap,
                                    driver_best,
                                    race_best,
                                    category,
                                    cfg,
                                )
                            )
                        stint_hot_idx += 1

            gap_to_race_best = (
                driver_best - race_best if np.isfinite(race_best) else float("nan")
            )
            consistency_gap_median = median - driver_best
            consistency_gap_p90 = p90 - driver_best

            summary_rows.append(
                {
                    "driver": driver,
                    "race": race,
                    "hot_laps": len(hot_times),
                    "best_lap_s": round(driver_best, 3),
                    "best_lap": _fmt_time(driver_best),
                    "race_best_s": round(race_best, 3) if np.isfinite(race_best) else "",
                    "race_best": _fmt_time(race_best),
                    "gap_to_race_best_s": round(gap_to_race_best, 3)
                    if np.isfinite(gap_to_race_best)
                    else "",
                    "gap_to_race_best": _fmt_time(gap_to_race_best),
                    "avg_hot_lap_s": round(avg, 3),
                    "avg_hot_lap": _fmt_time(avg),
                    "median_hot_lap_s": round(median, 3),
                    "median_hot_lap": _fmt_time(median),
                    "p90_hot_lap_s": round(p90, 3),
                    "p90_hot_lap": _fmt_time(p90),
                    "consistency_gap_median_s": round(consistency_gap_median, 3),
                    "consistency_gap_median": _fmt_time(consistency_gap_median),
                    "consistency_gap_p90_s": round(consistency_gap_p90, 3),
                    "consistency_gap_p90": _fmt_time(consistency_gap_p90),
                    "total_time_lost_vs_best_s": round(total_lost_vs_best_s, 1),
                    "total_time_lost_vs_best": _fmt_duration(total_lost_vs_best_s),
                    "potential_gain_to_median_s": round(total_lost_vs_median_s, 1),
                    "potential_gain_to_median": _fmt_duration(total_lost_vs_median_s),
                    "outlier_hot_laps": outlier_count,
                    "outlier_hot_lost_s": round(outlier_lost_s, 1),
                    "early_pace_laps": early_pace_count,
                    "early_pace_lost_s": round(early_pace_lost_s, 1),
                    "traffic_laps": traffic_count,
                    "traffic_time_s": round(traffic_time_s, 1),
                    "formation_laps": formation_count,
                    "vsc_laps": vsc_count,
                    "insight": _build_insight(
                        outlier_count=outlier_count,
                        outlier_lost_s=outlier_lost_s,
                        early_pace_count=early_pace_count,
                        early_pace_lost_s=early_pace_lost_s,
                        traffic_count=traffic_count,
                        traffic_time_s=traffic_time_s,
                        consistency_gap_median_s=consistency_gap_median,
                        consistency_gap_p90_s=consistency_gap_p90,
                        gap_to_race_best_s=gap_to_race_best,
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    laps_df = pd.DataFrame(lap_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["race", "potential_gain_to_median_s"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)
    if not laps_df.empty:
        laps_df = laps_df.sort_values(
            ["race", "driver", "delta_vs_driver_best_s"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)
    return summary_df, laps_df


def _lap_row(
    driver: str,
    race: str,
    stint: Session,
    lap: Lap,
    driver_best: float,
    race_best: float,
    category: str,
    cfg: OpportunityConfig,
) -> dict:
    delta_best = lap.lap_time_s - driver_best
    delta_race = (
        lap.lap_time_s - race_best if np.isfinite(race_best) else float("nan")
    )
    return {
        "driver": driver,
        "race": race,
        "category": category,
        "stint_id": _stint_id(stint.file_name),
        "stint_start": stint.start_dt.strftime("%H:%M:%S") if stint.start_dt else "",
        "lap_num": lap.num,
        "lap_type": lap.lap_type,
        "lap_time_s": round(lap.lap_time_s, 3),
        "lap_time": _fmt_time(lap.lap_time_s),
        "delta_vs_driver_best_s": round(delta_best, 3),
        "delta_vs_driver_best": _fmt_time(delta_best),
        "delta_vs_race_best_s": round(delta_race, 3) if np.isfinite(delta_race) else "",
        "delta_vs_race_best": _fmt_time(delta_race),
    }


def _fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return "0:00:00"
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"

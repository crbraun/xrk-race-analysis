"""Speed-based lap classification.

Each lap is labelled as one of:
  * ``out_lap``  - first lap of a session / lap leaving the pits.
  * ``in_lap``   - final lap of a session returning to the pits.
  * ``pit``      - lap containing a sustained stop (pit stop / driver change).
  * ``vsc``      - virtual safety car / full-course yellow: average lap speed
                   at or below the VSC speed limit (~35 mph by default).
  * ``formation`` - the single lap after a VSC period at race start or restart
                   when the field has not yet returned to hot pace.
  * ``slow``     - a complete green lap that is anomalously slow (traffic);
                   excluded from stats but not counted as VSC.
  * ``hot``      - a clean green-flag lap; the only type used in lap-time stats.

Only ``hot`` laps feed best / average / median / P90. ``vsc`` laps are counted
separately in the ``vsc_laps`` column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from xrk_parser import Lap, Session

MS_TO_MPH = 2.23694

HOT = "hot"
VSC = "vsc"
OUT_LAP = "out_lap"
IN_LAP = "in_lap"
PIT = "pit"
SLOW = "slow"
FORMATION = "formation"
INCOMPLETE = "incomplete"

STAT_LAP_TYPES = frozenset({HOT})


@dataclass
class Thresholds:
    """Tunable knobs for lap classification."""

    vsc_speed_mph: float = 35.0
    vsc_margin_mph: float = 8.0
    pit_stop_min_stop_s: float = 20.0
    min_speed_samples: int = 30
    # Traffic: remaining hot laps after stint best, slower than this ratio.
    hot_outlier_ratio: float = 1.15
    # Restart formation: lap after a later VSC block when still > this ratio
    # vs the race's best hot lap (race-start formation is always one lap).
    formation_restart_ratio: float = 1.05


def is_stat_lap(lap: Lap) -> bool:
    return lap.lap_type in STAT_LAP_TYPES


def stat_lap_times(laps: list[Lap]) -> list[float]:
    return [l.lap_time_s for l in laps if is_stat_lap(l)]


def _mean_mph(lap: Lap) -> float:
    if not np.isfinite(lap.speed_mean):
        return float("nan")
    return lap.speed_mean * MS_TO_MPH


def _vsc_cap_mph(cfg: Thresholds) -> float:
    return cfg.vsc_speed_mph + cfg.vsc_margin_mph


def _is_vsc_lap(lap: Lap, cfg: Thresholds) -> bool:
    mean = _mean_mph(lap)
    if not np.isfinite(mean):
        return False
    return mean <= _vsc_cap_mph(cfg)


def _above_vsc_pace(lap: Lap, cfg: Thresholds) -> bool:
    mean = _mean_mph(lap)
    return np.isfinite(mean) and mean > _vsc_cap_mph(cfg)


def classify_session(session: Session, cfg: Thresholds | None = None) -> None:
    """Per-stint pass: structural labels + VSC."""
    cfg = cfg or Thresholds()
    n = len(session.laps)
    for idx, lap in enumerate(session.laps):
        # Synthetic trailing laps are recovered pit-in laps by construction.
        if lap.synthetic:
            lap.lap_type = IN_LAP
            continue
        lap.lap_type = _classify_lap(
            lap, is_first=(idx == 0), is_last=(idx == n - 1), cfg=cfg
        )


def classify_race_context(sessions: list[Session], cfg: Thresholds) -> None:
    """Race-wide pass: formation (start + restart) and traffic laps."""
    by_race: dict[str, list[Session]] = {}
    for s in sessions:
        by_race.setdefault(s.race, []).append(s)
    for group in by_race.values():
        _classify_race(group, cfg)


def _vsc_blocks(ordered: list[Lap]) -> list[tuple[int, int]]:
    """(start, end) index pairs for each contiguous VSC block.

    ``start`` is the index of the first VSC lap in the block; ``end`` is the
    index of the first lap after the block (may equal ``len(ordered)``).
    """
    blocks: list[tuple[int, int]] = []
    i, n = 0, len(ordered)
    while i < n:
        if ordered[i].lap_type != VSC:
            i += 1
            continue
        start = i
        while i < n and ordered[i].lap_type == VSC:
            i += 1
        blocks.append((start, i))
    return blocks


def _vsc_block_tail_indices(ordered: list[Lap]) -> list[int]:
    """Index of the first lap after each contiguous VSC block."""
    return [end for _, end in _vsc_blocks(ordered) if end < len(ordered)]


def _classify_race(group: list[Session], cfg: Thresholds) -> None:
    group.sort(key=lambda s: s.start_dt or datetime.min)

    hot_times = [
        l.lap_time_s for s in group for l in s.laps if l.lap_type == HOT
    ]
    if not hot_times:
        return

    race_best = min(hot_times)
    restart_form_cutoff = race_best * cfg.formation_restart_ratio

    refs: list[tuple[Lap, datetime]] = []
    for s in group:
        for lap in s.laps:
            refs.append((lap, lap.abs_start or s.start_dt or datetime.min))
    refs.sort(key=lambda x: x[1])
    ordered = [lap for lap, _ in refs]

    for block_i, (start, end) in enumerate(_vsc_blocks(ordered)):
        # Lap leaving the caution (first lap after the block): the field is
        # still forming up, so it is a formation lap when above VSC pace.
        if end < len(ordered):
            lap = ordered[end]
            if lap.lap_type == HOT and _above_vsc_pace(lap, cfg):
                if block_i == 0 or lap.lap_time_s > restart_form_cutoff:
                    lap.lap_type = FORMATION
        # Lap entering the caution (last lap before the block): a lap run at
        # well below green pace as the field slows for a full-course yellow is
        # part of the caution sequence, not a hot lap. Gated on slowness so a
        # genuine flying lap immediately before a yellow is left untouched.
        if start > 0:
            lap = ordered[start - 1]
            if lap.lap_type == HOT and lap.lap_time_s > restart_form_cutoff:
                lap.lap_type = FORMATION

    for s in group:
        hot_laps = [(i, l) for i, l in enumerate(s.laps) if l.lap_type == HOT]
        if not hot_laps:
            continue
        best_time = min(l.lap_time_s for _, l in hot_laps)

        # Recover a dropped in-lap. libxrk's lap table sometimes ends before
        # the telemetry does, omitting the true pit-in lap and leaving a full
        # green lap as the final row -- which ``_classify_lap`` tentatively
        # tagged ``in_lap`` only because it was last. Promote that lap back to
        # ``hot`` when it is clearly a complete green lap: no pit crawl, green
        # mean speed, and a lap time within the normal band around the stint
        # best (the lower bound rejects short/partial final fragments that
        # would otherwise corrupt the best lap).
        last = s.laps[-1]
        no_crawl = (
            not np.isfinite(last.max_low_speed_run_s)
            or last.max_low_speed_run_s < cfg.pit_stop_min_stop_s
        )
        if (
            last.lap_type == IN_LAP
            and not last.synthetic
            and no_crawl
            and _above_vsc_pace(last, cfg)
            and 0.95 * best_time
            <= last.lap_time_s
            <= cfg.hot_outlier_ratio * best_time
        ):
            last.lap_type = HOT
            hot_laps = [
                (i, l) for i, l in enumerate(s.laps) if l.lap_type == HOT
            ]
            best_time = min(l.lap_time_s for _, l in hot_laps)

        # Only demote traffic laps *after* the stint has hit its best pace.
        # Early-race laps that are slower than the ultimate best but still
        # legitimate green-flag running (e.g. laps 4–9 at ~100s) stay hot.
        best_idx = next(i for i, l in hot_laps if l.lap_time_s == best_time)
        cutoff = best_time * cfg.hot_outlier_ratio
        for i, lap in hot_laps:
            if i > best_idx and lap.lap_time_s > cutoff:
                lap.lap_type = SLOW


def _classify_lap(
    lap: Lap, is_first: bool, is_last: bool, cfg: Thresholds
) -> str:
    if (
        np.isfinite(lap.max_low_speed_run_s)
        and lap.max_low_speed_run_s >= cfg.pit_stop_min_stop_s
    ):
        return OUT_LAP if is_first else IN_LAP
    if is_first:
        return OUT_LAP
    if lap.n_speed_samples < cfg.min_speed_samples:
        return INCOMPLETE
    if _is_vsc_lap(lap, cfg):
        return VSC
    if is_last:
        return IN_LAP
    return HOT

"""Parse AiM XRK files into structured session/lap data using libxrk.

Each XRK file is one logging session recorded by the dash. We read:
  * session start datetime from the logger metadata (Log Date + Log Time),
    NOT the file mtime (files exported together share an mtime),
  * the lap table (lap number + start/end offsets in ms),
  * the GPS Speed trace sliced per lap (for speed-based lap classification),
  * a handful of telemetry channels summarised per lap (for driver
    fingerprinting).

Absolute lap start time = session start datetime + lap start offset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from libxrk import aim_xrk

# Channels summarised per lap and used as driver-fingerprint inputs. Only
# channels actually present in a file are used.
FINGERPRINT_CHANNELS = (
    "RPM",
    "TPS",
    "BrakePressure",
    "AccelerometerX",
    "AccelerometerY",
    "GPS Speed",
)

_RACE_RE = re.compile(r"(Race\s*\d+)", re.IGNORECASE)

# Metadata key that holds the free-text note. Race Studio's "Comment" field is
# exposed by libxrk as "Long Comment"; we fall back to any key mentioning
# note/comment in case a different logger uses another tag.
_PRIMARY_NOTE_KEY = "Long Comment"


def _read_note(metadata: dict) -> str:
    value = str(metadata.get(_PRIMARY_NOTE_KEY, "") or "").strip()
    if value:
        return value
    for key, val in metadata.items():
        if key == _PRIMARY_NOTE_KEY:
            continue
        if any(tag in key.lower() for tag in ("comment", "note")):
            text = str(val or "").strip()
            if text:
                return text
    return ""


@dataclass
class Lap:
    """A single lap within a session."""

    num: int
    start_ms: int
    end_ms: int
    lap_time_s: float
    abs_start: datetime | None
    # Speed statistics (m/s) computed from the GPS Speed trace for this lap.
    speed_max: float = float("nan")
    speed_mean: float = float("nan")
    speed_min: float = float("nan")
    # Fraction of samples spent below the session "crawl" speed threshold.
    low_speed_frac: float = float("nan")
    # Longest continuous stretch (seconds) spent below the crawl threshold.
    max_low_speed_run_s: float = float("nan")
    n_speed_samples: int = 0
    # Lap classification, filled in later by lap_analysis.
    lap_type: str = "unknown"
    # True when this lap was synthesised from telemetry that continues past
    # libxrk's lap table (a dropped pit-in lap with no start/finish crossing).
    synthetic: bool = False


@dataclass
class Session:
    """One XRK file = one logging session."""

    path: Path
    file_name: str
    race: str
    start_dt: datetime | None
    venue: str
    vehicle: str
    # Driver name read from the file's note/comment metadata, if present.
    note: str = ""
    laps: list[Lap] = field(default_factory=list)
    # Per-lap summary stats for fingerprint channels: {channel: {stat: value}}.
    channel_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def stint_id(self) -> str:
        return self.file_name


def _parse_race_label(name: str) -> str:
    """Extract a race/event label from an AiM export file name.

    AiM names look like ``{Event}_E36_{Venue}_...``. The event prefix
    (everything before ``_E36_``) uniquely identifies a race weekend
    session — e.g. ``Palmer NEGT Race 1 June 2026`` vs
    ``Summit Point AER Race 1`` both contain "Race 1" but are different
    events. Falls back to ``Race N`` when the ``_E36_`` marker is absent.
    """
    stem = Path(name).stem
    if "_E36_" in stem:
        return stem.split("_E36_", 1)[0].strip()
    match = _RACE_RE.search(stem)
    if match:
        digits = re.search(r"\d+", match.group(1))
        return f"Race {digits.group(0)}" if digits else match.group(1)
    return "Unknown"


def _parse_start_dt(metadata: dict) -> datetime | None:
    log_date = str(metadata.get("Log Date", "")).strip()
    log_time = str(metadata.get("Log Time", "")).strip()
    if not log_date:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            stamp = f"{log_date} {log_time}".strip() if log_time else log_date
            return datetime.strptime(stamp, fmt)
        except ValueError:
            continue
    return None


def _speed_series(log) -> pd.DataFrame | None:
    """Return the GPS Speed trace as a DataFrame [timecodes(ms), speed(m/s)]."""
    if "GPS Speed" not in log.channels:
        return None
    df = log.channels["GPS Speed"].to_pandas()
    value_col = [c for c in df.columns if c != "timecodes"][0]
    return df.rename(columns={value_col: "speed"})[["timecodes", "speed"]]


def parse_file(path: str | Path, crawl_speed_ms: float | None = None) -> Session:
    """Parse a single XRK file into a :class:`Session`.

    ``crawl_speed_ms`` is the speed threshold (m/s) below which the car is
    considered crawling/stopped; if None it is derived per-session as a
    fraction of the session's racing speed (see below).
    """
    path = Path(path)
    log = aim_xrk(str(path))
    meta = log.metadata

    start_dt = _parse_start_dt(meta)
    session = Session(
        path=path,
        file_name=path.name,
        race=_parse_race_label(path.name),
        start_dt=start_dt,
        venue=str(meta.get("Venue", "")),
        vehicle=str(meta.get("Vehicle", "")),
        note=_read_note(meta),
    )

    laps_df = log.laps.to_pandas()
    speed_df = _speed_series(log)

    # Derive the session racing speed (robust high percentile) to set a
    # sensible crawl threshold when one is not supplied.
    if crawl_speed_ms is None and speed_df is not None and len(speed_df):
        racing_speed = float(np.nanpercentile(speed_df["speed"], 90))
        crawl_speed_ms = max(2.0, 0.15 * racing_speed)
    elif crawl_speed_ms is None:
        crawl_speed_ms = 2.0

    for row in laps_df.itertuples(index=False):
        start_ms = int(row.start_time)
        end_ms = int(row.end_time)
        lap_time_s = (end_ms - start_ms) / 1000.0
        abs_start = (
            start_dt + timedelta(milliseconds=start_ms) if start_dt else None
        )
        lap = Lap(
            num=int(row.num),
            start_ms=start_ms,
            end_ms=end_ms,
            lap_time_s=lap_time_s,
            abs_start=abs_start,
        )
        _fill_speed_stats(lap, speed_df, crawl_speed_ms)
        session.laps.append(lap)

    _append_trailing_lap(session, speed_df, crawl_speed_ms, start_dt)

    if session.laps:
        session.duration_s = sum(l.lap_time_s for l in session.laps)

    session.channel_stats = _channel_stats(log)
    return session


# Telemetry sometimes keeps logging after libxrk's lap table ends (a dropped
# pit-in lap that never crossed start/finish). We recover that driving time as
# a single synthetic in-lap, trimmed at the first sustained stop so idle
# pit-box time is not counted.
TRAILING_MIN_DRIVE_S = 20.0  # ignore trivial coast-down after the last lap
TRAILING_STOP_S = 15.0  # sustained below-crawl stretch that marks "parked"
TRAILING_MIN_LAPS = 4  # only recover a pit-in on a real stint, not a blip


def _trailing_drive(
    speed_df: pd.DataFrame | None, last_end_ms: int, crawl_ms: float
) -> tuple[int, float] | None:
    """Driving time after the last lap-table row, trimmed at the first stop.

    Returns ``(cutoff_ms, drive_s)`` for the moving portion of telemetry past
    ``last_end_ms``, or ``None`` when there is no meaningful trailing driving.
    """
    if speed_df is None:
        return None
    seg = speed_df[speed_df["timecodes"] >= last_end_ms]
    if len(seg) == 0:
        return None
    times = seg["timecodes"].to_numpy()
    speeds = seg["speed"].to_numpy()
    if times[-1] <= last_end_ms:
        return None

    below = speeds < crawl_ms
    cutoff = int(times[-1])
    run_start_idx: int | None = None
    for i, flag in enumerate(below):
        if flag:
            if run_start_idx is None:
                run_start_idx = i
            if (times[i] - times[run_start_idx]) / 1000.0 >= TRAILING_STOP_S:
                cutoff = int(times[run_start_idx])
                break
        else:
            run_start_idx = None

    drive_s = (cutoff - last_end_ms) / 1000.0
    if drive_s < TRAILING_MIN_DRIVE_S:
        return None
    return cutoff, float(drive_s)


def _append_trailing_lap(
    session: Session,
    speed_df: pd.DataFrame | None,
    crawl_ms: float,
    start_dt: datetime | None,
) -> None:
    if len(session.laps) < TRAILING_MIN_LAPS:
        return
    last_end = session.laps[-1].end_ms
    trailing = _trailing_drive(speed_df, last_end, crawl_ms)
    if trailing is None:
        return
    cutoff_ms, drive_s = trailing
    abs_start = (
        start_dt + timedelta(milliseconds=last_end) if start_dt else None
    )
    lap = Lap(
        num=session.laps[-1].num + 1,
        start_ms=last_end,
        end_ms=cutoff_ms,
        lap_time_s=drive_s,
        abs_start=abs_start,
        synthetic=True,
    )
    _fill_speed_stats(lap, speed_df, crawl_ms)
    session.laps.append(lap)


def _fill_speed_stats(lap: Lap, speed_df: pd.DataFrame | None, crawl_ms: float) -> None:
    if speed_df is None:
        return
    seg = speed_df[
        (speed_df["timecodes"] >= lap.start_ms) & (speed_df["timecodes"] < lap.end_ms)
    ]
    if seg.empty:
        return
    speeds = seg["speed"].to_numpy()
    times = seg["timecodes"].to_numpy()
    lap.speed_max = float(np.nanmax(speeds))
    lap.speed_mean = float(np.nanmean(speeds))
    lap.speed_min = float(np.nanmin(speeds))
    lap.n_speed_samples = int(len(speeds))

    below = speeds < crawl_ms
    lap.low_speed_frac = float(np.mean(below))
    lap.max_low_speed_run_s = _longest_run_seconds(below, times)


def _longest_run_seconds(mask: np.ndarray, times_ms: np.ndarray) -> float:
    """Longest continuous time (s) where ``mask`` is True."""
    if not mask.any():
        return 0.0
    best = 0.0
    run_start = None
    for i, flag in enumerate(mask):
        if flag and run_start is None:
            run_start = times_ms[i]
        elif not flag and run_start is not None:
            best = max(best, (times_ms[i] - run_start) / 1000.0)
            run_start = None
    if run_start is not None:
        best = max(best, (times_ms[-1] - run_start) / 1000.0)
    return float(best)


def _channel_stats(log) -> dict[str, dict[str, float]]:
    """Summary stats per fingerprint channel across the whole session."""
    stats: dict[str, dict[str, float]] = {}
    for name in FINGERPRINT_CHANNELS:
        if name not in log.channels:
            continue
        df = log.channels[name].to_pandas()
        value_col = [c for c in df.columns if c != "timecodes"][0]
        values = df[value_col].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
    return stats


def find_xrk_files(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).expanduser().glob("*.xrk"))

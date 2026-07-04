"""Assign a driver to each session (each XRK file = one driver's stint).

Driver names are not stored in the XRK files, so the primary workflow is to
tag each session yourself, aided by a list of sessions showing when each one
started and its best lap time:

  * ``session_summaries`` / ``write_session_list`` produce that list.
  * ``tag_sessions_interactively`` prompts you per session (showing race,
    start time, and best lap) to enter a driver name.
  * Alternatively a ``driver_assignments.json`` template ({file: name}) can be
    hand-edited and reused with ``--reuse-assignments``.

Optional helpers for pre-filling names you have not tagged yet:

  * ``--profile`` - after you tag a few stints, build a driving-style profile
    per driver and suggest names for the rest (review before accepting).
  * ``--suggest-clusters`` - blind cluster labels (Driver A/B/...) when you
    have not tagged anything yet.
"""

from __future__ import annotations

import csv
import os
import shlex
import string
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lap_analysis import HOT
from xrk_parser import Session

UNASSIGNED = "Unassigned"

# Max standardized fingerprint distance to accept a profile match.
DEFAULT_PROFILE_DISTANCE = 2.5

# ---------------------------------------------------------------------------
# Per-session summary + tagging (primary workflow)
# ---------------------------------------------------------------------------


def _best_hot(session: Session) -> float:
    hot = [l.lap_time_s for l in session.laps if l.lap_type == HOT]
    if hot:
        return float(min(hot))
    complete = [l.lap_time_s for l in session.laps if l.lap_time_s > 0]
    return float(min(complete)) if complete else float("nan")


def _n_hot(session: Session) -> int:
    return sum(1 for l in session.laps if l.lap_type == HOT)


def _fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "n/a"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:06.3f}"


def _fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return ""
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def session_summaries(sessions: list[Session]) -> list[dict]:
    """One summary dict per session, sorted by race then start time."""
    rows = []
    for s in sessions:
        best = _best_hot(s)
        end_ms = max((l.end_ms for l in s.laps), default=0)
        end_dt = s.start_dt + timedelta(milliseconds=end_ms) if s.start_dt else None
        rows.append(
            {
                "file": s.file_name,
                "race": s.race,
                "start_dt": s.start_dt,
                "start": s.start_dt.strftime("%m/%d/%Y %H:%M:%S")
                if s.start_dt
                else "",
                "end": end_dt.strftime("%H:%M:%S") if end_dt else "",
                "duration": _fmt_duration(end_ms / 1000.0),
                "best_lap_s": round(best, 3) if np.isfinite(best) else "",
                "best_lap": _fmt_time(best),
                "hot_laps": _n_hot(s),
                "total_laps": len(s.laps),
                "note": s.note,
            }
        )
    rows.sort(key=lambda r: (r["race"], r["start_dt"] or datetime.min))
    return rows


def print_session_list(rows: list[dict]) -> None:
    print("\n=== Sessions (each file is one driver's stint) ===")
    print(
        f"{'#':>2}  {'race':7} {'start':19} {'end':8} {'dur':>6} {'best_lap':9} "
        f"{'hot':>4} {'laps':>4}  {'note/driver':12}  file"
    )
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>2}  {r['race']:7} {r['start']:19} {r['end']:8} {r['duration']:>6} "
            f"{r['best_lap']:9} {r['hot_laps']:>4} {r['total_laps']:>4}  "
            f"{(r.get('note') or ''):12}  {r['file']}"
        )


def write_session_list(rows: list[dict], path: Path) -> None:
    fields = [
        "file", "race", "start", "end", "duration", "best_lap", "best_lap_s",
        "hot_laps", "total_laps", "note",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    print(f"Wrote session list to {path}")


_ASSIGN_HEADER = """\
# ===== Driver assignments =====
# Fill in the driver name after '=' for each stint below, then save this file.
#   - Use the same name for every stint driven by the same person.
#   - Leave a line blank to mark that stint Unassigned (e.g. short in/out blips).
#   - You can edit freely and reorder your thinking; only the 'file = name'
#     lines are read. Lines starting with '#' are comments/context.
# When done, run:  python analyze.py --reuse-assignments
#
# Optional profiling: tag a few stints, save, then run with --profile to
# pre-fill the rest from driving-style fingerprints (review suggestions!).
# ---------------------------------------------------------------------------
"""


def write_assignments_file(
    rows: list[dict],
    prefill: dict[str, str],
    path: Path,
    profile_suggested: set[str] | None = None,
) -> None:
    """Write a single human-friendly, editable assignments file listing every
    stint (grouped by race) with its context, and a ``file = name`` line to
    fill in."""
    lines = [_ASSIGN_HEADER]
    current_race = None
    for i, r in enumerate(rows, 1):
        if r["race"] != current_race:
            current_race = r["race"]
            lines.append(f"\n# ===== {current_race} =====")
        hint = ""
        if profile_suggested and r["file"] in profile_suggested:
            hint = "  [profile suggestion - verify!]"
        lines.append(
            f"\n# [{i:>2}] {r['start']} - {r['end']} ({r['duration']})"
            f" | best {r['best_lap']} | {r['hot_laps']} hot / {r['total_laps']} laps"
            f" | note: {r.get('note') or '-'}{hint}"
        )
        lines.append(f"{r['file']} = {prefill.get(r['file'], '') or ''}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote editable driver-assignment file to {path}")


def load_assignments(path: Path) -> dict[str, str]:
    """Parse a ``file = name`` assignments file into {file_name: driver}."""
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key:
            mapping[key] = value.strip()
    return mapping


def _open_in_editor(path: Path) -> bool:
    """Best-effort: open ``path`` in $VISUAL/$EDITOR (blocking). False if none."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        return False
    try:
        subprocess.call(shlex.split(editor) + [str(path)])
        return True
    except (OSError, ValueError):
        return False


def build_prefill(
    sessions: list[Session],
    saved: dict[str, str],
    profile_suggestions: dict[str, str] | None = None,
    cluster_suggestions_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Best current guess per file: saved > note > profile > cluster."""
    profile_suggestions = profile_suggestions or {}
    cluster_suggestions_map = cluster_suggestions_map or {}
    notes = {s.file_name: s.note for s in sessions}
    return {
        s.file_name: (
            saved.get(s.file_name)
            or notes.get(s.file_name)
            or profile_suggestions.get(s.file_name)
            or cluster_suggestions_map.get(s.file_name, "")
        )
        for s in sessions
    }


def assign_drivers(
    sessions: list[Session],
    assignments_path: Path,
    sessions_csv: Path,
    reuse: bool = False,
    suggest_clusters: bool = False,
    use_profile: bool = False,
    profile_distance: float = DEFAULT_PROFILE_DISTANCE,
    n_drivers: int | None = None,
    template_only: bool = False,
) -> dict[str, str]:
    """Return a {file_name: driver_name} mapping.

    Writes the session list CSV, then an editable assignments file listing all
    stints. With ``reuse`` the saved assignments file is read directly. Without
    it, the assignments file is (re)written prefilled with notes/saved names and
    opened in your editor if $EDITOR is set; edit all stints, save, and it is
    read back. If no editor is available, edit the file and re-run with
    ``--reuse-assignments``.
    """
    rows = session_summaries(sessions)
    write_session_list(rows, sessions_csv)
    print_session_list(rows)

    files = {s.file_name for s in sessions}
    saved = load_assignments(assignments_path)
    cluster_map = (
        cluster_suggestions(sessions, n_drivers) if suggest_clusters else {}
    )
    profile_map: dict[str, str] = {}
    profile_suggested: set[str] = set()
    if use_profile:
        profile_map, profile_suggested = suggest_from_profiles(
            sessions, saved, profile_distance
        )
        if profile_map:
            print(
                f"\nProfile matching: suggested {len(profile_map)} driver name(s) "
                f"from {len({saved[f] for f in saved if saved[f]})} tagged stint(s). "
                "Review lines marked [profile suggestion] in the assignments file."
            )
        elif any(saved.values()):
            print(
                "\nProfile matching: no confident matches for untagged stints "
                f"(distance threshold {profile_distance}). Tag more stints or "
                "lower --profile-distance."
            )
        else:
            print(
                "\nProfile matching: tag at least one stint in the assignments "
                "file, save, then re-run with --profile."
            )

    prefill = build_prefill(sessions, saved, profile_map, cluster_map)

    notes = {s.file_name: s.note for s in sessions}
    n_from_notes = sum(1 for f in files if not saved.get(f) and notes.get(f))
    if n_from_notes:
        print(f"\nPrepopulated {n_from_notes} driver name(s) from file notes.")

    if reuse:
        untagged = [f for f in files if not prefill.get(f)]
        if untagged:
            print(f"\nNote: {len(untagged)} session(s) left untagged -> {UNASSIGNED}.")
        if use_profile and profile_map:
            write_assignments_file(rows, prefill, assignments_path, profile_suggested)
        return {f: (prefill.get(f) or UNASSIGNED) for f in files}

    write_assignments_file(rows, prefill, assignments_path, profile_suggested)

    if not template_only and _open_in_editor(assignments_path):
        edited = load_assignments(assignments_path)
        result = {f: (edited.get(f) or "") for f in files}
        blanks = sum(1 for f in files if not result[f])
        if blanks:
            print(f"\n{blanks} session(s) left blank -> {UNASSIGNED}.")
        return {f: (result[f] or UNASSIGNED) for f in files}

    print(
        f"\nOpen {assignments_path} in your editor, fill in a driver name after "
        "each '=', save, then run:\n  python analyze.py --reuse-assignments"
    )
    return {f: (prefill.get(f) or UNASSIGNED) for f in files}


# ---------------------------------------------------------------------------
# Optional: profile-based matching from manually tagged stints
# ---------------------------------------------------------------------------


def _feature_dataframe(sessions: list[Session]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {s.file_name: session_features(s) for s in sessions}, orient="index"
    ).reindex(columns=_FEATURE_ORDER)


def build_driver_profiles(
    sessions: list[Session], tagged: dict[str, str]
) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    """Mean standardized fingerprint per driver from tagged stints."""
    df = _feature_dataframe(sessions)
    matrix = _prepare_matrix(df)
    file_to_idx = {f: i for i, f in enumerate(df.index)}

    by_driver: dict[str, list[int]] = {}
    for fname, driver in tagged.items():
        if not driver or driver == UNASSIGNED:
            continue
        idx = file_to_idx.get(fname)
        if idx is not None:
            by_driver.setdefault(driver, []).append(idx)

    profiles = {
        driver: matrix[idxs].mean(axis=0) for driver, idxs in by_driver.items() if idxs
    }
    return profiles, df, matrix


def suggest_from_profiles(
    sessions: list[Session],
    tagged: dict[str, str],
    distance_threshold: float = DEFAULT_PROFILE_DISTANCE,
    min_hot_laps: int = 3,
) -> tuple[dict[str, str], set[str]]:
    """Suggest drivers for untagged stints by nearest tagged-driver profile.

    Returns ``({file: driver}, set of files that were profile-suggested)``.
    Only fills slots that are blank in ``tagged``; never overrides a name you
    already entered.
    """
    if not any(tagged.values()):
        return {}, set()

    profiles, df, matrix = build_driver_profiles(sessions, tagged)
    if not profiles:
        return {}, set()

    file_to_idx = {f: i for i, f in enumerate(df.index)}
    suggestions: dict[str, str] = {}
    suggested_files: set[str] = set()

    for s in sessions:
        if tagged.get(s.file_name):
            continue
        if _n_hot(s) < min_hot_laps:
            continue
        idx = file_to_idx.get(s.file_name)
        if idx is None:
            continue
        vec = matrix[idx]
        best_driver, best_dist = None, float("inf")
        for driver, prof in profiles.items():
            dist = float(np.linalg.norm(vec - prof))
            if dist < best_dist:
                best_driver, best_dist = driver, dist
        if best_driver and best_dist <= distance_threshold:
            suggestions[s.file_name] = best_driver
            suggested_files.add(s.file_name)

    return suggestions, suggested_files


# ---------------------------------------------------------------------------
# Optional: cluster sessions by driving style to suggest names
# ---------------------------------------------------------------------------

_FEATURE_ORDER = [
    "best_hot_s",
    "median_hot_s",
    "speed_p90",
    "rpm_p50",
    "rpm_p90",
    "rpm_max",
    "tps_mean",
    "tps_std",
    "brake_mean",
    "brake_p90",
    "accx_std",
    "accy_std",
]


def _median_hot(session: Session) -> float:
    hot = [l.lap_time_s for l in session.laps if l.lap_type == HOT]
    return float(np.median(hot)) if hot else float("nan")


def session_features(session: Session) -> dict[str, float]:
    cs = session.channel_stats

    def stat(channel: str, key: str) -> float:
        return cs.get(channel, {}).get(key, float("nan"))

    return {
        "best_hot_s": _best_hot(session),
        "median_hot_s": _median_hot(session),
        "speed_p90": stat("GPS Speed", "p90"),
        "rpm_p50": stat("RPM", "p50"),
        "rpm_p90": stat("RPM", "p90"),
        "rpm_max": stat("RPM", "max"),
        "tps_mean": stat("TPS", "mean"),
        "tps_std": stat("TPS", "std"),
        "brake_mean": stat("BrakePressure", "mean"),
        "brake_p90": stat("BrakePressure", "p90"),
        "accx_std": stat("AccelerometerX", "std"),
        "accy_std": stat("AccelerometerY", "std"),
    }


def _prepare_matrix(df: pd.DataFrame) -> np.ndarray:
    filled = df.copy()
    for col in filled.columns:
        median = filled[col].median()
        filled[col] = filled[col].fillna(median if np.isfinite(median) else 0.0)
    return StandardScaler().fit_transform(filled.to_numpy(dtype=float))


def _suggest_k(matrix: np.ndarray, k_min: int = 2, k_max: int = 5) -> int:
    n = matrix.shape[0]
    k_max = min(k_max, n - 1)
    if n < 3 or k_max < k_min:
        return min(3, n)
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_suggestions(
    sessions: list[Session], n_drivers: int | None, min_hot_laps: int = 3
) -> dict[str, str]:
    """Return {file_name: 'Driver A/B/...'} suggested by driving-style clusters.

    Short fragment sessions are attached to the temporally nearest real stint.
    """
    real = [s for s in sessions if _n_hot(s) >= min_hot_laps] or sessions
    fragments = [s for s in sessions if s not in real]

    df = pd.DataFrame.from_dict(
        {s.file_name: session_features(s) for s in real}, orient="index"
    ).reindex(columns=_FEATURE_ORDER)
    matrix = _prepare_matrix(df)
    n = len(real)

    k = max(1, min(n_drivers or _suggest_k(matrix), n))
    if k == 1 or n == 1:
        labels = np.zeros(n, dtype=int)
    else:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)

    best_by_file = {s.file_name: _best_hot(s) for s in sessions}
    int_groups: dict[int, list[str]] = {}
    for label, fname in zip(labels, df.index):
        int_groups.setdefault(int(label), []).append(fname)

    file_to_label = {f: lbl for lbl, fs in int_groups.items() for f in fs}
    for frag in fragments:
        pool = [s for s in real if s.race == frag.race] or real
        if frag.start_dt and any(s.start_dt for s in pool):
            anchor = min(
                (s for s in pool if s.start_dt),
                key=lambda s: abs((s.start_dt - frag.start_dt).total_seconds()),
            ).file_name
        else:
            anchor = pool[0].file_name
        int_groups.setdefault(file_to_label[anchor], []).append(frag.file_name)

    def group_best(files: list[str]) -> float:
        vals = [best_by_file[f] for f in files if np.isfinite(best_by_file[f])]
        return min(vals) if vals else float("inf")

    ordered = sorted(int_groups.values(), key=group_best)
    return {
        f: f"Driver {string.ascii_uppercase[i]}"
        for i, files in enumerate(ordered)
        for f in files
    }

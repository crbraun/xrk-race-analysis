"""Probe a single XRK file to inspect libxrk's parsed output.

Usage:
    python probe_xrk.py "/path/to/file.xrk"

Prints session metadata, the lap table (times converted to seconds), the list
of channel names, and a summary of the GPS Speed channel. Handy for confirming
the libxrk API surface and channel naming before running the full analysis.
"""

from __future__ import annotations

import sys

from libxrk import GPS_CHANNEL_NAMES, aim_xrk


def main(path: str) -> None:
    log = aim_xrk(path)

    print("=== metadata ===")
    for key, value in log.metadata.items():
        print(f"  {key!r} -> {value!r}")

    print("\n=== channels ===")
    for name in log.channels:
        print(f"  {name}")
    print("\nGPS channel names known to libxrk:", GPS_CHANNEL_NAMES)

    laps = log.laps.to_pandas()
    laps["lap_s"] = (laps["end_time"] - laps["start_time"]) / 1000.0
    print(f"\n=== laps ({len(laps)}) ===")
    print(laps.to_string())

    if "GPS Speed" in log.channels:
        speed = log.channels["GPS Speed"].to_pandas()
        col = speed.columns[-1]
        print("\n=== GPS Speed (m/s) ===")
        print(
            f"  min={speed[col].min():.2f} "
            f"median={speed[col].median():.2f} "
            f"max={speed[col].max():.2f} "
            f"samples={len(speed)} "
            f"duration_s={speed['timecodes'].max() / 1000.0:.1f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

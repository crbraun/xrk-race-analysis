# XRK Race Weekend Analyzer

Analyze a folder of AiM RaceStudio `.xrk` stint logs from an endurance race
weekend and produce a per-driver-per-day performance summary.

For each driver, for each day, the tool reports:

- **Best Time** - fastest completed lap
- **Average Time** - mean of *hot* (clean green-flag) laps only
- **Total Time in Car** - summed wall-clock stint duration
- **Total Laps** - completed laps
- **Laps Under VSC** - caution / full-course-yellow laps

## How it works

1. **Parse** every `.xrk` with [`libxrk`](https://pypi.org/project/libxrk/):
   lap start/end times, `GPS Speed`, `TPS` (throttle), `BrakePressure`, and
   session metadata (date, start time, vehicle, venue).
2. **Classify** each lap (see below) so only representative laps feed the average.
3. **Attribute** each stint to a driver. The driver name is *not* stored in the
   files, so the tool:
   - fingerprints each stint (pace + throttle/brake style) and clusters similar
     stints so you confirm a group once instead of every stint;
   - prompts you per group, showing **best time / start / duration / laps** to
     jog your memory;
   - caches the mapping in `stint_drivers.json` so re-runs never re-ask.
4. **Aggregate** to one row per driver per day and write CSVs.

## Lap classification (the averaging filter)

Only **hot** laps are averaged. Categories (all thresholds tunable):

| Category  | Meaning                                                        |
|-----------|----------------------------------------------------------------|
| `short`   | leading partial "lap 0" before the first start/finish crossing |
| `anomaly` | implausibly fast lap (double S/F trigger) - excluded           |
| `out`     | first lap of a stint (out-lap)                                 |
| `in`      | last lap of a stint (in-lap)                                   |
| `pit`     | car stopped/crawling for part of the lap                       |
| `vsc`     | circulating but much slower than green pace (caution / VSC)    |
| `warmup`  | circulating, slower than the hot threshold (traffic / warmup)  |
| `hot`     | clean green lap within `hot_factor` of the stint best          |

Defaults (in `xrkweekend/config.py`):

- `hot_factor = 1.07` - a lap counts as hot if within 7% of the stint best
- `caution_factor = 1.35` - a lap at/above 135% of the best is a caution/VSC lap
- `min_lap_fraction_of_median = 0.80` - laps faster than 80% of the median are
  treated as anomalies so one bogus-fast lap can't poison the reference

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Interactive (prompts to name each stint group), default folder:
python analyze.py

# Custom folder:
python analyze.py --input "/path/to/weekend"

# No prompts (provisional Driver A/B/... labels for a quick look):
python analyze.py --non-interactive

# Re-do all driver assignments from scratch:
python analyze.py --reassign

# Tune the hot / caution thresholds:
python analyze.py --hot-factor 1.05 --caution-factor 1.40
```

Outputs are written to `output/`:

- `driver_day_summary.csv` - the headline per-driver-per-day table
- `stints.csv` - one row per stint (start time, laps, hot/vsc counts, best, cluster)
- `laps.csv` - one row per lap with its category (for auditing / threshold tuning)

## Notes

- Assign real driver names once; they persist in an editable
  `driver_assignments.txt` file **inside your telemetry folder** (keyed by
  stint filename). On later runs against that same folder the assignments load
  and reuse automatically, so you never have to re-tag. Point elsewhere with
  `--assignments PATH` if needed.
- Fingerprint clustering only groups stints that are genuinely similar in pace
  and style; when drivers are within a second or two of each other it will fall
  back to per-stint prompts. That is expected.

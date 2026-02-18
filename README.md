# MeerKAT Scheduler

A scheduler built for the MeerKAT radio telescope. It optimizes telescope usage by greedily maximizing the length of scheduled observations within specified constraints, including visibility windows, minimum observation durations, configuration time (i.e. build + DelayCal), and day/night and sunrise/sunset constraints.

- **Greedy Scheduling Algorithm:** Maximizes observation durations of available observation time in each schedule block (SB). The obvious disadvantages of this apply.
- **Sunrise/Sunset** Computes sunrise and sunset times in Local Sidereal Time (LST).
- **Current Constraints:**
  - Minimum observation duration
  - Telescope setup time (i.e. build + delay calibration) (TBD: adjust for repeated observations in same mode)
  - Night-only observations 
  - Sunrise/sunset avoidance

## Installation

```sh
git clone https://github.com/richarms/mk_sched_opt.git
cd mk_sched_opt
uv venv .venv
source .venv/bin/activate
uv pip sync requirements.txt
```

## Dependency Workflow (uv)

Use `requirements.in` for direct dependencies only, and compile a pinned `requirements.txt` lock file.

```sh
# After changing requirements.in
uv pip compile requirements.in -o requirements.txt

# Install exact locked environment
uv pip sync requirements.txt
```

### Requirements

- Python >= 3.8
- pandas
- numpy
- astropy

## Usage:

```sh
uv run mk_sched.py \
  --max_days 100 \
  --max_no_schedule_days 2 \
  --minimum_observation_duration 0.75 \
  --setup_time 0.3 \
  --infile "data/Observations 2025 - 2025.observations.csv" \
  --outfile "schedules/MeerKAT_Schedule.csv" \
  --avoid_weds True
```

### Command-line Arguments

- `-d, --max_days`: Maximum scheduling period (default: 150 days)
- `-n, --max_no_schedule_days`: Maximum consecutive days without scheduling before stopping (default: 3)
- `-m, --minimum_observation_duration`: Minimum duration of observations in hours (default: 0.5 hours / 30 min)
- `-s, --setup_time`: Setup time required before each observation in hours (default: 0.25 hours / 15 min)
- `-i, --infile`: Input observation CSV path (default: `data/Observations 2025 - 2025.observations.csv`)
- `-o, --outfile`: Output schedule CSV path (default: `schedules/MeerKAT_Schedule.csv`)
- `-w, --avoid_weds`: Skip Wednesday 06:00-13:00 slots when `True` (default: `True`)

## Input

A CSV file containing data from approved SBs (typically downloaded from the OPT) with the following input columns:
- `id`
- `lst_start`
- `lst_start_end`
- `simulated_duration` (in seconds)
- `instrument_band`
- `night_obs`
- `avoid_sunrise_sunset`

## Output

Writes a CSV file to ```schedules/```

### Output Columns

- `Day`: Scheduling day
- `CaptureBlock_ID`: Unique ID for each observation
- `SB_ID`: Observation ID (uses "DelayCal" for delay calibration)
- `Setup_Start_LST`: Start time of the setup
- `Observation_Start_LST`: Start of actual observation
- `Observation_End_LST`: End of the observation
- `Band`: Instrument band
- `Night_only`: Indicates if night observation is required
- `Visibility_window`: Allowed LST visibility window
- `Duration_hrs`: Duration of each scheduled segment in hours

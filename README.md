# MeerKAT Scheduler

A greedy scheduler built for the MeerKAT radio telescope. It optimizes telescope usage by greedily maximizing the length of scheduled observations within specified constraints, including visibility windows, minimum observation durations, configuration time (i.e. DleayCal), and day/night and sunrise/sunset constraints.

## Notes

- **Greedy Scheduling Algorithm:** Maximizes observation durations of available observation time in each schedule block (SB).
- **Sunrise/Sunset** Computes sunrise and sunset times in Local Sidereal Time (LST) using Astropy.
- **Current Constraints:**
  - Minimum observation duration
  - Telescope setup time (i.e. Delay calibration)
  - Night-only observations 
  - Sunrise/sunset avoidance

## Installation

Clone the repository and install requirements:

```sh
git clone https://github.com/richarms/mk_sched_opt.git
cd mk_sched_opt
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- pandas
- numpy
- astropy

## Usage:

```sh
python mk_sched.py \
  --max_days 100 \
  --max_no_schedule_days 2 \
  --minimum_observation_duration 0.75 \
  --setup_time 0.3
```

### Command-line Arguments

- `--max_days`: Maximum scheduling period (default: 150 days)
- `--max_no_schedule_days`: Maximum consecutive days without scheduling before stopping (default: 3)
- `--minimum_observation_duration`: Minimum duration of observations in hours (default: 0.5 hours / 30 min)
- `--setup_time`: Setup time required before each observation in hours (default: 0.25 hours / 15 min)

## Data Input

A CSV file containing data from approved SBs with the following input columns:
- `id`
- `lst_start`
- `lst_start_end`
- `simulated_duration` (in seconds)
- `instrument_band`
- `night_obs` (Yes/No)
- `avoid_sunrise_sunset` (Yes/No)

## Output

Generates a CSV file in ```schedules/```

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

## License

MIT License

---

**Contact:** Richard Armstrong richarms@sarao.ac.za


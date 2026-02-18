import argparse
import astropy.units as u
import numpy as np
import logging
import pandas as pd
import re

from astroplan import Observer
from astropy.time import Time
from astropy.coordinates import AltAz, Angle, EarthLocation, get_sun
from astropy.utils import iers
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Allow scheduling to continue when fresh IERS-A downloads are unavailable.
iers.conf.auto_max_age = None

parser = argparse.ArgumentParser(description="MeerKAT Scheduling Script")
parser.add_argument('-d', '--max_days', type=int, default=150, help='Maximum number of scheduling days')
parser.add_argument('-n', '--max_no_schedule_days', type=int, default=3, help='Exit scheduler after this many days without any observations')
parser.add_argument('-m', '--minimum_observation_duration', type=float, default=0.5, help='Minimum observation duration in hours (default 0.5 = 30 min)')
parser.add_argument('-s', '--setup_time', type=float, default=0.25, help='Setup time in hours (default 0.25 = 15 min)')
parser.add_argument('-o', '--outfile', type=str, default='schedules/MeerKAT_Schedule.csv', help='Output filename')
parser.add_argument('-w', '--avoid_weds', type=bool, default=True)
parser.add_argument('-i', '--infile', type=str,  default='data/Observations 2025 - 2025.observations.csv')

# MeerKAT location
meerkat_location = EarthLocation(lat=-30.7130*u.deg, lon=21.4430*u.deg, height=1038*u.m)

# Convert LST strings to float (hours)
def lst_to_hours(lst_str):
    h, m = map(int, lst_str.split(':'))
    return h + m / 60

def is_nighttime(start, end, sunrise, sunset):
    if sunrise < sunset:
        return end <= sunrise or start >= sunset
    else:
        return sunrise <= start <= sunset and sunrise <= end <= sunset

def get_sunrise_sunset_lst_astroplan(obs_date):
    """Use the astroplan library for sunrise/sunset calculation"""
    observer = Observer(location=meerkat_location)
    time = Time(obs_date)
    
    # Define the horizon for sunrise/sunset
    horizon = -0.833 * u.deg
    
    # astroplan finds the next rise/set time from the given time
    sunrise_utc = observer.sun_rise_time(time, which='next', horizon=horizon)
    sunset_utc = observer.sun_set_time(time, which='next', horizon=horizon)
    
    # Convert to LST in hours
    sunrise_lst = sunrise_utc.sidereal_time('apparent', longitude=meerkat_location.lon).hour
    sunset_lst = sunset_utc.sidereal_time('apparent', longitude=meerkat_location.lon).hour

    return sunrise_lst, sunset_lst

# Calculate sunrise and sunset LST hours
def get_sunrise_sunset_lst(obs_date):
    midnight = Time(obs_date) + timedelta(days=0.5)
    delta_midnight = np.linspace(-12, 12, 1000)*u.hour
    times = midnight + delta_midnight
    frame = AltAz(obstime=times, location=meerkat_location)
    sun_altazs = get_sun(times).transform_to(frame)
    sunrise_time = times[np.argmin(np.abs(sun_altazs.alt - (-0.833*u.deg)))]
    sunset_time = times[np.argmin(np.abs(sun_altazs.alt - (-0.833*u.deg)))]
    return sunrise_time.sidereal_time('apparent', meerkat_location).hour, sunset_time.sidereal_time('apparent', meerkat_location).hour

def next_lst_zero(location: EarthLocation, from_time: Time = None) -> Time:
    # Calculate the next UTC time when local sidereal time (LST) will be 0h at the given EarthLocation.
    if from_time is None:
        from_time = Time(datetime.now(timezone.utc))

    current_lst = from_time.sidereal_time('apparent', longitude=location.lon)
    delta_lst = (Angle(24 * u.hourangle) - current_lst).wrap_at(24 * u.hourangle)

    sidereal_day = 23.9344696 * u.hour
    delta_utc = delta_lst.hour / 24.0 * sidereal_day

    return from_time + delta_utc

def _constraint_failure_reason(obs, start_time, duration, sunrise, sunset, start_datetime=None):
    end_time = (start_time + duration) % 24
    min_lst, max_lst = obs['lst_start'], obs['lst_start_end']

    if min_lst < max_lst:
        visible = min_lst <= start_time < max_lst
    else:
        visible = start_time >= min_lst or start_time < max_lst
    if not visible:
        return "lst_visibility"

    if obs['night_obs'] == 'Yes':
        if not ((8 <= start_time <= 20) and (8 <= end_time <= 20)):
            return "night_only"

    if obs['avoid_sunrise_sunset'] == 'Yes':
        if (start_time < sunrise < end_time) or (start_time < sunset < end_time):
            return "sunrise_sunset"

    if args.avoid_weds and start_datetime is not None:
        if start_datetime.weekday() == 2 and 6 <= start_datetime.hour < 13:
            return "avoid_weds"

    return None

# Helper to check visibility constraints
def fits_constraints(obs, start_time, duration, sunrise, sunset, start_datetime=None):
    return _constraint_failure_reason(obs, start_time, duration, sunrise, sunset, start_datetime) is None

def schedule_observations(unscheduled_df, max_days, min_obs_duration, setup_time):
    unscheduled = unscheduled_df.copy()
    if 'split_count' not in unscheduled.columns:
        unscheduled['split_count'] = 0
    scheduled_observations = []
    script_start_datetime = next_lst_zero(location=meerkat_location).to_datetime()
    
    unscheduled_LST_hours = np.zeros(24)
    no_schedule_days = 0
    days_processed = 0
    total_scheduled_duration = 0.0
    no_candidate_reasons = {"min_duration": 0, "lst_visibility": 0, "night_only": 0, "sunrise_sunset": 0, "avoid_weds": 0}
    
    for day in range(1, max_days + 1):
        days_processed = day
        daily_schedule, daily_scheduled_duration, daily_reason_counts = schedule_day(
            unscheduled,
            day,
            script_start_datetime,
            setup_time,
            min_obs_duration,
            unscheduled_LST_hours,
            include_reason_counts=True
        )
        total_scheduled_duration += daily_scheduled_duration
        for reason, count in daily_reason_counts.items():
            no_candidate_reasons[reason] += count
        
        if daily_scheduled_duration == 0:
            no_schedule_days += 1
            reason_summary = ", ".join(f"{k}={v}" for k, v in daily_reason_counts.items() if v > 0) or "none"
            logging.info(
                "Day %d: 0 scheduled hours (no candidates). Rejection counts: %s",
                day,
                reason_summary,
            )
        else:
            no_schedule_days = 0
        
        if no_schedule_days >= args.max_no_schedule_days:
            logging.info(f"Exiting after {no_schedule_days} consecutive days without scheduling.")
            break
        
        # Remove fully scheduled observations
        fully_scheduled_ids = {entry['ID'] for entry in daily_schedule if entry['ID'] != 'DelayCal'}
        unscheduled = unscheduled[~unscheduled['id'].isin(fully_scheduled_ids)]
        
        scheduled_observations.extend(daily_schedule)
        
        if args.progress_every > 0 and day % args.progress_every == 0:
            logging.info(
                "Progress day %d/%d: scheduled %.2f hours today, %.2f cumulative, %d remaining observations",
                day,
                max_days,
                daily_scheduled_duration,
                total_scheduled_duration,
                len(unscheduled),
            )

    # report_unscheduled_observations(unscheduled)
    report_unscheduled_lst_hours(unscheduled_LST_hours)
    final_log_level = logging.WARNING if args.quiet else logging.INFO
    logging.log(
        final_log_level,
        "Run summary: days_processed=%d total_scheduled_hours=%.2f scheduled_entries=%d remaining_observations=%d no_candidate_counts=%s",
        days_processed,
        total_scheduled_duration,
        len(scheduled_observations),
        len(unscheduled),
        ", ".join(f"{k}:{v}" for k, v in no_candidate_reasons.items()),
    )
    
    return scheduled_observations

def schedule_day(unscheduled, day, script_start_datetime, setup_time, min_obs_duration, unscheduled_LST_hours, include_reason_counts=False):
    if 'split_count' not in unscheduled.columns:
        unscheduled['split_count'] = 0

    current_LST = 0.0
    daily_schedule = []
    daily_scheduled_duration = 0
    daily_time_remaining = 24
    scheduled_today = set()
    prev_mode = None
    no_candidate_reasons = {"min_duration": 0, "lst_visibility": 0, "night_only": 0, "sunrise_sunset": 0, "avoid_weds": 0}

    sunrise, sunset = get_sunrise_sunset_lst_astroplan(script_start_datetime + timedelta(days=day - 1))

    while daily_time_remaining > setup_time and not unscheduled.empty:
        candidates, reason_counts = get_schedulable_candidates(
            unscheduled,
            current_LST,
            daily_time_remaining,
            setup_time,
            min_obs_duration,
            sunrise,
            sunset,
            script_start_datetime,
            day,
            return_reasons=True,
        )

        if not candidates:
            for reason, count in reason_counts.items():
                no_candidate_reasons[reason] += count
            track_unscheduled_hour(unscheduled_LST_hours, current_LST)
            current_LST = (current_LST + 0.5) % 24
            daily_time_remaining -= 0.5
            continue

        obs_start_dt = script_start_datetime + timedelta(days=(day - 1), hours=(current_LST + setup_time))
        idx, duration_to_schedule = select_best_candidate(candidates, unscheduled, prev_mode, obs_start_dt)
        obs = unscheduled.loc[idx]

        append_observation_to_schedule(
            daily_schedule, obs, day, current_LST, setup_time, duration_to_schedule, script_start_datetime
        )

        remaining_before = unscheduled.loc[idx, 'simulated_duration']
        update_observation_duration(unscheduled, idx, duration_to_schedule)

        if duration_to_schedule < remaining_before:
            unscheduled.at[idx, 'split_count'] += 1

        prev_mode = parse_observation_mode(obs)

        current_LST = (current_LST + setup_time + duration_to_schedule) % 24
        daily_time_remaining -= (setup_time + duration_to_schedule)
        daily_scheduled_duration += duration_to_schedule

        if unscheduled.at[idx, 'simulated_duration'] <= 0:
            scheduled_today.add(idx)

    unscheduled = unscheduled.drop(list(scheduled_today)).copy()

    if include_reason_counts:
        return daily_schedule, daily_scheduled_duration, no_candidate_reasons

    return daily_schedule, daily_scheduled_duration

# Constraint checking logic
def get_schedulable_candidates(unscheduled, current_LST, daily_time_remaining, setup_time, min_obs_duration, sunrise, sunset, script_start_datetime, day, return_reasons=False):
    candidates = []
    reason_counts = {"min_duration": 0, "lst_visibility": 0, "night_only": 0, "sunrise_sunset": 0, "avoid_weds": 0}
    captureblock_datetime = script_start_datetime + timedelta(days=(day - 1), hours=current_LST)  # retained for future scheduling metadata
    for idx, obs in unscheduled.iterrows():
        available_duration = daily_time_remaining - setup_time
        duration_possible = min(obs['simulated_duration'], available_duration)
        if duration_possible < min_obs_duration:
            reason_counts["min_duration"] += 1
            continue
        start_datetime = script_start_datetime + timedelta(days=(day - 1), hours=(current_LST + setup_time))
        if return_reasons:
            failure_reason = _constraint_failure_reason(
                obs, (current_LST + setup_time) % 24, duration_possible, sunrise, sunset, start_datetime
            )
            if failure_reason is None:
                candidates.append((idx, duration_possible))
            else:
                reason_counts[failure_reason] += 1
        else:
            if fits_constraints(obs, (current_LST + setup_time) % 24, duration_possible, sunrise, sunset, start_datetime):
                candidates.append((idx, duration_possible))

    if return_reasons:
        return candidates, reason_counts
    return candidates

def parse_observation_mode(obs):
    product = str(obs.get('product', '')).lower()
    band = str(obs.get('instrument_band', '')).lower()
    if '32k' in product:
        channels = '32k'
    elif '4k' in product:
        channels = '4k'
    else:
        channels = ''
    if 'narrow' in product or re.search(r'(^|[^a-z])n([^a-z]|$)', product):
        width = 'narrow'
    elif 'wide' in product or re.search(r'(^|[^a-z])w([^a-z]|$)', product):
        width = 'wide'
    else:
        width = ''
    return band, channels, width


def calculate_observation_score(obs, duration, split_count, prev_mode, current_datetime):
    score = 0.0

    future_splits = split_count + (duration < obs['simulated_duration'])
    score -= future_splits

    constraints = 0
    if obs.get('night_obs') == 'Yes':
        constraints += 1
    if obs.get('avoid_sunrise_sunset') == 'Yes':
        constraints += 1
    score += constraints

    current_mode = parse_observation_mode(obs)
    if prev_mode is not None and current_mode == prev_mode:
        score += 1

    if (
        'cadence_days' in obs
        and 'last_observed' in obs
        and not pd.isna(obs['cadence_days'])
        and not pd.isna(obs['last_observed'])
        and obs['cadence_days'] > 0
    ):
        current_mjd = Time(current_datetime).mjd
        next_due = obs['last_observed'] + obs['cadence_days']
        diff = abs(current_mjd - next_due)
        if diff < obs['cadence_days']:
            score += (obs['cadence_days'] - diff) / obs['cadence_days']

    return score


def select_best_candidate(candidates, unscheduled, prev_mode, current_datetime):
    def candidate_key(item):
        idx, duration = item
        obs = unscheduled.loc[idx]
        split_count = obs.get('split_count', 0)
        score = calculate_observation_score(obs, duration, split_count, prev_mode, current_datetime)
        return (score, duration)

    return max(candidates, key=candidate_key)

def append_observation_to_schedule(schedule, obs, day, current_LST, setup_time, duration, script_start_datetime):
    captureblock_datetime = script_start_datetime + timedelta(days=(day - 1), hours=current_LST)
    
    schedule.append({
        'Day': day,
        'ID': 'DelayCal',
        'Proposal ID': obs['proposal_id'],
        'Description': 'Build and DelayCal',
        'UTC': captureblock_datetime.isoformat(),
        'Observation_Start_LST': current_LST,
        'Observation_End_LST': (current_LST + setup_time) % 24,
        'Band': obs['instrument_band'],
        'Night_only': 'n/a',
        'Visibility_window': '',
        'Duration_hrs': setup_time
    })

    schedule.append({
        'Day': day,
        'ID': obs['id'],
        'Proposal ID': obs['proposal_id'],
        'Description': obs['description'],
        'UTC': (captureblock_datetime + timedelta(hours=setup_time)).isoformat(),
        'Observation_Start_LST': (current_LST + setup_time) % 24,
        'Observation_End_LST': (current_LST + setup_time + duration) % 24,
        'Band': obs['instrument_band'],
        'Night_only': obs['night_obs'],
        'Visibility_window': f"{obs['lst_start']:.2f}-{obs['lst_start_end']:.2f}",
        'Duration_hrs': duration
    })

def update_observation_duration(unscheduled, idx, duration):
    unscheduled.at[idx, 'simulated_duration'] -= duration

def track_unscheduled_hour(unscheduled_LST_hours, current_LST):
    unscheduled_hour = int(current_LST) % 24
    unscheduled_LST_hours[unscheduled_hour] += 0.5

def report_unscheduled_observations(unscheduled):
    if not unscheduled.empty:
        logging.info("Unscheduled Observations Remaining:")
        for _, obs in unscheduled.iterrows():
            logging.info(f"SB {obs['id']} has {obs['simulated_duration']:.2f} hrs unscheduled.")

def report_unscheduled_lst_hours(unscheduled_LST_hours):
    logging.info('Total unscheduled hours by LST (0-23 order):')
    indexed_values = {}
    for index, value in enumerate(unscheduled_LST_hours):
        logging.info('LST %d: total %.1f', index, float(value))
        indexed_values[index] = float(value)

    sorted_dict =  {k: v for k, v in sorted(indexed_values.items(), key=lambda item: item[1], reverse=True)}
    logging.info('Least LST pressure: %s', list(sorted_dict.keys()))


def configure_logging(verbose=False, quiet=False):
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.getLogger().setLevel(level)


if __name__ == '__main__':
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Load approved SBs
    df = pd.read_csv(args.infile)

    df['lst_start'] = df['lst_start'].apply(lst_to_hours)
    df['lst_start_end'] = df['lst_start_end'].apply(lst_to_hours)

    # Convert simulated_duration from seconds to hours
    df['simulated_duration'] = df['simulated_duration'] / 3600

    schedule = schedule_observations(df, args.max_days, args.minimum_observation_duration, args.setup_time)
    schedule_df = pd.DataFrame(schedule)
    schedule_df.to_csv(args.outfile, index=False)
    final_log_level = logging.WARNING if args.quiet else logging.INFO
    logging.log(final_log_level, 'Schedule created successfully: %s', args.outfile)

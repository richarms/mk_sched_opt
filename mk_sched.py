import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
from astropy.time import Time
from astropy.coordinates import AltAz, Angle, EarthLocation, get_sun
import astropy.units as u

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="MeerKAT Scheduling Script")
parser.add_argument('--max_days', type=int, default=150, help='Maximum number of scheduling days')
parser.add_argument('--max_no_schedule_days', type=int, default=3, help='Exit scheduler after this many days without any observations')
parser.add_argument('--minimum_observation_duration', type=float, default=0.5, help='Minimum observation duration in hours (default 0.5 = 30 min)')
parser.add_argument('--setup_time', type=float, default=0.25, help='Setup time in hours (default 0.25 = 15 min)')
parser.add_argument('--outfile', type=str, default='schedules/MeerKAT_Schedule.csv', help='Output filename')
parser.add_argument('--avoid_weds', type=bool, default=True)
args = parser.parse_args()

# Convert LST strings to float (hours)
def lst_to_hours(lst_str):
    h, m = map(int, lst_str.split(':'))
    return h + m / 60

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
        from_time = Time(datetime.utcnow())

    current_lst = from_time.sidereal_time('apparent', longitude=location.lon)
    delta_lst = (Angle(24 * u.hourangle) - current_lst).wrap_at(24 * u.hourangle)

    sidereal_day = 23.9344696 * u.hour
    delta_utc = delta_lst.hour / 24.0 * sidereal_day

    return from_time + delta_utc

# Helper to check visibility constraints
def fits_constraints(obs, start_time, duration, sunrise, sunset):
    end_time = (start_time + duration) % 24
    min_lst, max_lst = obs['lst_start'], obs['lst_start_end']
    if min_lst < max_lst:
        visible = min_lst <= start_time < max_lst
    else:
        visible = start_time >= min_lst or start_time < max_lst

    if not visible:
        return False

    if obs['night_obs'] == 'Yes':
        if not ((8 <= start_time <= 20) and (8 <= end_time <= 20)):
            return False

    if obs['avoid_sunrise_sunset'] == 'Yes':
        if (start_time < sunrise < end_time) or (start_time < sunset < end_time):
            return False
    
    if args.avoid_weds:
        pass
        # if start_time.weekday() == 2 and 6 <= start_time.hour < 13:
        #     return False

    return True

def schedule_observations(unscheduled_df, max_days, min_obs_duration, setup_time):
    unscheduled = unscheduled_df.copy()
    scheduled_observations = []
    script_start_datetime = next_lst_zero(location=meerkat_location).to_datetime()
    
    unscheduled_LST_hours = np.zeros(24)
    no_schedule_days = 0
    
    for day in range(1, max_days + 1):
        daily_schedule, daily_scheduled_duration = schedule_day(
            unscheduled,
            day,
            script_start_datetime,
            setup_time,
            min_obs_duration,
            unscheduled_LST_hours
        )
        
        if daily_scheduled_duration == 0:
            no_schedule_days += 1
        else:
            no_schedule_days = 0
        
        if no_schedule_days >= args.max_no_schedule_days:
            logging.info(f"Exiting after {no_schedule_days} consecutive days without scheduling.")
            break
        
        # Remove fully scheduled observations
        fully_scheduled_ids = {entry['ID'] for entry in daily_schedule if entry['ID'] != 'DelayCal'}
        unscheduled = unscheduled[~unscheduled['id'].isin(fully_scheduled_ids)]
        
        scheduled_observations.extend(daily_schedule)
        
        if day % 20 == 0:
            print(f"Scheduled day {day}")

    # report_unscheduled_observations(unscheduled)
    report_unscheduled_lst_hours(unscheduled_LST_hours)
    
    return scheduled_observations

def schedule_day(unscheduled, day, script_start_datetime, setup_time, min_obs_duration, unscheduled_LST_hours):
    current_LST = 0.0
    daily_schedule = []
    daily_scheduled_duration = 0
    daily_time_remaining = 24
    scheduled_today = set()
    
    sunrise, sunset = get_sunrise_sunset_lst(script_start_datetime + timedelta(days=day - 1))
    
    while daily_time_remaining > setup_time and not unscheduled.empty:
        candidates = get_schedulable_candidates(
            unscheduled,
            current_LST,
            daily_time_remaining,
            setup_time,
            min_obs_duration,
            sunrise,
            sunset,
            script_start_datetime,
            day
        )

        if not candidates:
            track_unscheduled_hour(unscheduled_LST_hours, current_LST)
            current_LST = (current_LST + 0.5) % 24
            daily_time_remaining -= 0.5
            continue
        
        idx, duration_to_schedule = select_best_candidate(candidates)
        obs = unscheduled.loc[idx]
        
        append_observation_to_schedule(
            daily_schedule, obs, day, current_LST, setup_time, duration_to_schedule, script_start_datetime
        )

        update_observation_duration(unscheduled, idx, duration_to_schedule)
        
        current_LST = (current_LST + setup_time + duration_to_schedule) % 24
        daily_time_remaining -= (setup_time + duration_to_schedule)
        daily_scheduled_duration += duration_to_schedule
        
        if unscheduled.at[idx, 'simulated_duration'] <= 0:
            scheduled_today.add(idx)
    
    unscheduled = unscheduled.drop(list(scheduled_today)).copy()
    
    return daily_schedule, daily_scheduled_duration

# Constraint checking logic
def get_schedulable_candidates(unscheduled, current_LST, daily_time_remaining, setup_time, min_obs_duration, sunrise, sunset, script_start_datetime, day):
    candidates = []
    captureblock_datetime = script_start_datetime + timedelta(days=(day - 1), hours=current_LST)
    for idx, obs in unscheduled.iterrows():
        available_duration = daily_time_remaining - setup_time
        duration_possible = min(obs['simulated_duration'], available_duration)
        if duration_possible < min_obs_duration:
            continue
        if fits_constraints(obs, (current_LST + setup_time) % 24, duration_possible, sunrise, sunset):
            candidates.append((idx, duration_possible))
    return candidates

def select_best_candidate(candidates):
    # Prioritize candidate with maximum telescope utilization
    return max(candidates, key=lambda x: x[1])

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
    # logging.info('Total unscheduled hours by LST:')
    # for index, value in enumerate(unscheduled_LST_hours):
    #     print(f'LST: {index}, unscheduled hours: {float(value)}')
    
    # Report unscheduled hours explicitly
    print(f'Total number of unscheduled hours in LST 0-23 order: ')
    indexed_values = {}
    # Create a dictionary with sorted values from a unscheduled list as keys, and original indices as values.
    for index, value in enumerate(unscheduled_LST_hours):
        print(f'LST: {index}, total: {float(value)}')
        indexed_values[index] = float(value)

    sorted_dict =  {k: v for k, v in sorted(indexed_values.items(), key=lambda item: item[1], reverse=True)}
    print(f'Least LST pressure: {sorted_dict.keys()}')





if __name__ == '__main__':
    # MeerKAT location
    meerkat_location = EarthLocation(lat=-30.7130*u.deg, lon=21.4430*u.deg, height=1038*u.m)

    # Load approved SBs
    df = pd.read_csv('data/Observations 2025 - 2025.observations.csv')

    df['lst_start'] = df['lst_start'].apply(lst_to_hours)
    df['lst_start_end'] = df['lst_start_end'].apply(lst_to_hours)

    # Convert simulated_duration from seconds to hours
    df['simulated_duration'] = df['simulated_duration'] / 3600

    schedule = schedule_observations(df, args.max_days, args.minimum_observation_duration, args.setup_time)
    schedule_df = pd.DataFrame(schedule)
    schedule_df.to_csv(args.outfile, index=False)
    print(f'Schedule created successfully: {args.outfile}')
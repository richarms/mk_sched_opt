import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_sun, AltAz
import astropy.units as u
import pprint

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MeerKAT Scheduling Script")
parser.add_argument('--max_days', type=int, default=150, help='Maximum number of scheduling days')
parser.add_argument('--max_no_schedule_days', type=int, default=3, help='Exit scheduler after this many days without any observations')
parser.add_argument('--minimum_observation_duration', type=float, default=0.5, help='Minimum observation duration in hours (default 0.5 = 30 min)')
parser.add_argument('--setup_time', type=float, default=0.25, help='Setup time in hours (default 0.25 = 15 min)')
args = parser.parse_args()

# MeerKAT location
meerkat_location = EarthLocation(lat=-30.7130*u.deg, lon=21.4430*u.deg, height=1038*u.m)

# Load approved observations
df = pd.read_csv('data/Observations 2025 - 2025.observations.csv')

# Convert LST strings to float (hours)
def lst_to_hours(lst_str):
    h, m = map(int, lst_str.split(':'))
    return h + m / 60

df['lst_start'] = df['lst_start'].apply(lst_to_hours)
df['lst_start_end'] = df['lst_start_end'].apply(lst_to_hours)

# Convert simulated_duration from seconds to hours
df['simulated_duration'] = df['simulated_duration'] / 3600

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

    return True

# Schedule observations greedily maximizing observation length
def schedule_observations(unscheduled, max_days, min_obs_duration, setup_time):
    unscheduled = unscheduled.copy()
    schedule = []
    day = 1
    current_LST = 0.0
    script_start_datetime = datetime.utcnow()
    no_schedule_days = 0

    unscheduled_LST_hours = np.zeros(24)  # Track unscheduled hours explicitly

    while not unscheduled.empty and day <= max_days:
        daily_schedule = []
        scheduled_today = set()
        daily_time_remaining = 24
        daily_scheduled_duration = 0

        sunrise, sunset = get_sunrise_sunset_lst(script_start_datetime + timedelta(days=day - 1))

        while daily_time_remaining > setup_time and not unscheduled.empty:
            obs_durations = []

            for idx, obs in unscheduled.iterrows():
                available_duration = daily_time_remaining - setup_time
                duration_to_schedule = min(obs['simulated_duration'], available_duration)

                if duration_to_schedule < min_obs_duration:
                    continue

                if fits_constraints(obs, (current_LST + setup_time) % 24, duration_to_schedule, sunrise, sunset):
                    obs_durations.append((idx, duration_to_schedule))

            if not obs_durations:
                unscheduled_hour = int(current_LST) % 24
                unscheduled_LST_hours[unscheduled_hour] += 0.5
                current_LST = (current_LST + 0.5) % 24
                daily_time_remaining -= 0.5
                continue

            idx, duration_to_schedule = max(obs_durations, key=lambda x: x[1])
            obs = unscheduled.loc[idx]

            captureblock_datetime = script_start_datetime + timedelta(days=(day - 1), hours=current_LST)
            captureblock_id = captureblock_datetime.strftime("%Y%m%d%H%M%S")

            daily_schedule.append({
                'Day': day,
                'CaptureBlock_ID': captureblock_id,
                'SB_ID': 'DelayCal',
                'Setup_Start_LST': current_LST,
                'Observation_Start_LST': current_LST,
                'Observation_End_LST': (current_LST + setup_time) % 24,
                'Band': obs['instrument_band'],
                'Night_only': obs['night_obs'],
                'Visibility_window': '',
                'Duration_hrs': setup_time
            })

            daily_schedule.append({
                'Day': day,
                'CaptureBlock_ID': captureblock_id,
                'SB_ID': obs['id'],
                'Setup_Start_LST': current_LST,
                'Observation_Start_LST': (current_LST + setup_time) % 24,
                'Observation_End_LST': (current_LST + setup_time + duration_to_schedule) % 24,
                'Band': obs['instrument_band'],
                'Night_only': obs['night_obs'],
                'Visibility_window': f"{obs['lst_start']:.2f}-{obs['lst_start_end']:.2f}",
                'Duration_hrs': duration_to_schedule
            })

            current_LST = (current_LST + setup_time + duration_to_schedule) % 24
            daily_time_remaining -= (setup_time + duration_to_schedule)
            daily_scheduled_duration += duration_to_schedule

            scheduled_today.add(idx)

        if daily_scheduled_duration == 0:
            no_schedule_days += 1
        else:
            no_schedule_days = 0

        if no_schedule_days >= args.max_no_schedule_days:
            logging.info(f"Exiting after {no_schedule_days} consecutive days without scheduling.")
            break

        unscheduled = unscheduled.drop(list(scheduled_today))
        schedule.extend(daily_schedule)
        day += 1
        if day % 10 == 0:
            logging.info(f'Scheduling day: {day}')

    if not unscheduled.empty:
        logging.info("Unscheduled or unschedulable observations:")
        logging.info(unscheduled['id'].to_list())

    # Report unscheduled hours explicitly
    # logging.info("Unscheduled LST Hours Summary (Total hours unscheduled at each LST hour):")
    # Create a dictionary with sorted values from a unscheduled list as keys, and original indices as values.
    indexed_values = {}
    print(f'Total number of unscheduled hours in LST 0-23 order: ')
    for index, value in enumerate(unscheduled_LST_hours):
        print(f'LST: {index}, total: {float(value)}')
        indexed_values[index] = float(value)

    sorted_dict =  {k: v for k, v in sorted(indexed_values.items(), key=lambda item: item[1], reverse=True)}
    print(f'Least LST pressure: {sorted_dict.keys()}')

    # for hour in range(24):
    #     logging.info(f"LST Hour {hour:02d}:00 - {hour+1:02d}:00 => {unscheduled_LST_hours[hour]:.2f} hours unscheduled")

    return schedule

schedule = schedule_observations(df, args.max_days, args.minimum_observation_duration, args.setup_time)
schedule_df = pd.DataFrame(schedule)
schedule_df.to_csv('schedules/MeerKAT_Greedy_Schedule.csv', index=False)
logging.info("Schedule created successfully: schedules/MeerKAT_Greedy_Schedule.csv")


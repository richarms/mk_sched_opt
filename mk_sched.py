import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

schedule = []
setup_time_hours = 15 / 60  # 15-minute setup time
minimum_observation_duration_hours = 30 / 60  # 30-minute minimum duration

# Helper to check visibility constraints
def fits_constraints(obs, start_time, duration):
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
        sunrise, sunset = 6, 18
        if (start_time < sunrise < end_time) or (start_time < sunset < end_time):
            return False

    return True

# Schedule observations greedily maximizing observation length
def schedule_observations(unscheduled, max_days=150):
    unscheduled = unscheduled.copy()
    day = 1
    current_LST = 0.0
    script_start_datetime = datetime.utcnow()
    no_schedule_days = 0

    while not unscheduled.empty and day <= max_days:
        daily_schedule = []
        scheduled_today = set()
        daily_time_remaining = 24
        daily_scheduled_duration = 0

        while daily_time_remaining > setup_time_hours and not unscheduled.empty:
            obs_durations = []

            for idx, obs in unscheduled.iterrows():
                available_duration = daily_time_remaining - setup_time_hours
                duration_to_schedule = min(obs['simulated_duration'], available_duration)

                if duration_to_schedule < minimum_observation_duration_hours:
                    continue

                if fits_constraints(obs, (current_LST + setup_time_hours) % 24, duration_to_schedule):
                    obs_durations.append((idx, duration_to_schedule))

            if not obs_durations:
                current_LST = (current_LST + 0.5) % 24
                daily_time_remaining -= 0.5
                continue

            idx, duration_to_schedule = max(obs_durations, key=lambda x: x[1])
            obs = unscheduled.loc[idx]

            captureblock_datetime = script_start_datetime + timedelta(days=(day - 1), hours=current_LST)
            captureblock_id = captureblock_datetime.strftime("%Y%m%d%H%M%S")

            # Add DelayCal entry explicitly
            daily_schedule.append({
                'Day': day,
                'CaptureBlock_ID': captureblock_id,
                'SB_ID': 'DelayCal',
                'Setup_Start_LST': current_LST,
                'Observation_Start_LST': current_LST,
                'Observation_End_LST': (current_LST + setup_time_hours) % 24,
                'Band': obs['instrument_band'],
                'Night_only': obs['night_obs'],
                'Visibility_window': '',
                'Duration_hrs': setup_time_hours
            })

            # Add main observation entry
            daily_schedule.append({
                'Day': day,
                'CaptureBlock_ID': captureblock_id,
                'SB_ID': obs['id'],
                'Setup_Start_LST': current_LST,
                'Observation_Start_LST': (current_LST + setup_time_hours) % 24,
                'Observation_End_LST': (current_LST + setup_time_hours + duration_to_schedule) % 24,
                'Band': obs['instrument_band'],
                'Night_only': obs['night_obs'],
                'Visibility_window': f"{obs['lst_start']:.2f}-{obs['lst_start_end']:.2f}",
                'Duration_hrs': duration_to_schedule
            })

            current_LST = (current_LST + setup_time_hours + duration_to_schedule) % 24
            daily_time_remaining -= (setup_time_hours + duration_to_schedule)
            daily_scheduled_duration += duration_to_schedule

            if duration_to_schedule < obs['simulated_duration']:
                unscheduled.at[idx, 'simulated_duration'] -= duration_to_schedule
                unscheduled.at[idx, 'lst_start'] = current_LST
            else:
                scheduled_today.add(idx)

        if daily_scheduled_duration == 0:
            no_schedule_days += 1
        else:
            no_schedule_days = 0

        if no_schedule_days >= 3:
            print(f"Exiting after {no_schedule_days} consecutive days without scheduling.")
            break

        unscheduled = unscheduled.drop(list(scheduled_today))
        schedule.extend(daily_schedule)
        day += 1
        if day % 10 == 0:
          print(f'Scheduling day: {day}')

    if not unscheduled.empty:
        print("Unscheduled or unschedulable observations:")
        print(unscheduled['id'].to_list())

unscheduled = df.copy()
schedule_observations(unscheduled)

# Convert schedule to DataFrame
schedule_df = pd.DataFrame(schedule)

# Output schedule CSV
schedule_df.to_csv('schedules/MeerKAT_Greedy_Schedule.csv', index=False)

print("Greedy schedule created successfully: MeerKAT_Greedy_Schedule.csv")


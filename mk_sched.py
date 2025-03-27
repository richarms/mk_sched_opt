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

# Sort observations by earliest possible start
df = df.sort_values(by='lst_start').reset_index(drop=True)

schedule = []
setup_time_hours = 15 / 60  # 15-minute setup time

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

# Schedule observations robustly with LST-day wrap-around and capture block IDs
def schedule_observations(unscheduled, max_days=150):
    unscheduled = unscheduled.copy()
    day = 1
    current_LST = 0.0
    script_start_datetime = datetime.utcnow()

    while not unscheduled.empty and day <= max_days:
        daily_schedule = []
        scheduled_today = set()
        if day % 10 == 0:
            print(f"Scheduling day {day}")
        daily_time_remaining = 24

        while daily_time_remaining > 0 and not unscheduled.empty:
            candidates = unscheduled[unscheduled.apply(
                lambda obs: fits_constraints(obs, (current_LST + setup_time_hours) % 24, obs['simulated_duration']), axis=1)]

            if candidates.empty:
                current_LST = (current_LST + 0.1) % 24
                daily_time_remaining -= 0.1
                continue

            obs = candidates.iloc[0]
            duration_to_schedule = min(obs['simulated_duration'], daily_time_remaining - setup_time_hours)

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
                'Visibility_window': ''
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
                'Visibility_window': f"{obs['lst_start']:.2f}-{obs['lst_start_end']:.2f}"
            })

            current_LST = (current_LST + setup_time_hours + duration_to_schedule) % 24
            daily_time_remaining -= (setup_time_hours + duration_to_schedule)

            if duration_to_schedule < obs['simulated_duration']:
                unscheduled.at[obs.name, 'simulated_duration'] -= duration_to_schedule
                unscheduled.at[obs.name, 'lst_start'] = current_LST
            else:
                scheduled_today.add(obs.name)

        unscheduled = unscheduled.drop(list(scheduled_today))
        schedule.extend(daily_schedule)
        day += 1

unscheduled = df.copy()
schedule_observations(unscheduled)

# Convert schedule to DataFrame
schedule_df = pd.DataFrame(schedule)

# Identify least-occupied LST ranges across all days
occupied_times = np.zeros(24)
for _, row in schedule_df.iterrows():
    start_hour = int(row['Observation_Start_LST'])
    end_hour = int(np.ceil(row['Observation_End_LST']))
    for hour in range(start_hour, end_hour):
        occupied_times[hour % 24] += 1

least_occupied = np.argsort(occupied_times)[:12]  # Get 12 least-occupied hours
print(f"Least-occupied LST hours (ordered): {least_occupied}")

# Output schedule CSV
schedule_df.to_csv('schedules/MeerKAT_LST_Wraparound_Schedule.csv', index=False)

print("Schedule with explicit DelayCal entries, setup time, LST wrap-around, capture block IDs, and least-occupied LST report created successfully: MeerKAT_LST_Wraparound_Schedule.csv")


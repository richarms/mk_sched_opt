import pandas as pd
import numpy as np

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
unschedulable = []

# Helper to check visibility constraints
def fits_constraints(obs, start_time, duration):
    end_time = start_time + duration
    if end_time > 24:
        return False
    if start_time < obs['lst_start'] or start_time >= obs['lst_start_end']:
        return False
    if obs['night_obs'] == 'Yes' and not (8 <= start_time and end_time <= 20):
        return False
    if obs['avoid_sunrise_sunset'] == 'Yes' and ((start_time <= 6 <= end_time) or (start_time <= 18 <= end_time)):
        return False
    return True

# Schedule observations robustly with partial scheduling allowed
def schedule_observations(unscheduled, max_days=90, max_attempts=3):
    unscheduled = unscheduled.copy()
    unscheduled['attempts'] = 0
    day = 1

    while not unscheduled.empty and day <= max_days:
        current_LST = 0.0
        daily_schedule = []
        progress_made = False
        scheduled_today = set()

        print(f"Scheduling day {day}")

        while current_LST < 24:
            candidates = unscheduled[(unscheduled['lst_start_end'] > current_LST) & (unscheduled['lst_start'] <= current_LST)]

            if candidates.empty:
                future_candidates = unscheduled[unscheduled['lst_start'] > current_LST]
                if future_candidates.empty:
                    break
                next_start = future_candidates['lst_start'].min()
                daily_schedule.append({
                    'Day': day,
                    'Observation_ID': '(Gap)',
                    'Start_LST': current_LST,
                    'End_LST': next_start,
                    'Band': '',
                    'Night_only': '',
                    'Visibility_window': ''
                })
                current_LST = next_start
                continue

            obs = candidates.iloc[0]
            available_window = obs['lst_start_end'] - current_LST
            duration_to_schedule = min(obs['simulated_duration'], available_window)

            if fits_constraints(obs, current_LST, duration_to_schedule):
                daily_schedule.append({
                    'Day': day,
                    'Observation_ID': obs['id'],
                    'Start_LST': current_LST,
                    'End_LST': current_LST + duration_to_schedule,
                    'Band': obs['instrument_band'],
                    'Night_only': obs['night_obs'],
                    'Visibility_window': f"{obs['lst_start']:.2f}-{obs['lst_start_end']:.2f}"
                })
                current_LST += duration_to_schedule
                progress_made = True

                if duration_to_schedule < obs['simulated_duration']:
                    remaining_duration = obs['simulated_duration'] - duration_to_schedule
                    unscheduled.at[obs.name, 'simulated_duration'] = remaining_duration
                    unscheduled.at[obs.name, 'lst_start'] = max(obs['lst_start'], current_LST)
                else:
                    scheduled_today.add(obs.name)
                    unscheduled.at[obs.name, 'attempts'] = 0
            else:
                unscheduled.at[obs.name, 'attempts'] += 1
                if unscheduled.at[obs.name, 'attempts'] >= max_attempts:
                    unschedulable.append(obs)
                    scheduled_today.add(obs.name)
                current_LST = obs['lst_start_end']  # Explicitly jump past problematic observation

        unscheduled = unscheduled.drop(list(scheduled_today))
        schedule.extend(daily_schedule)

        if not progress_made:
            print("No progress made today, stopping scheduling.")
            break

        day += 1

    if unschedulable:
        print("The following observations could not be scheduled:")
        for obs in unschedulable:
            print(obs[['id', 'lst_start', 'lst_start_end', 'simulated_duration']])

unscheduled = df.copy()
schedule_observations(unscheduled)

# Convert schedule to DataFrame
schedule_df = pd.DataFrame(schedule)

# Identify least-occupied LST ranges across all days
occupied_times = np.zeros(24)
for _, row in schedule_df.iterrows():
    occupied_times[int(row['Start_LST']):int(np.ceil(row['End_LST']))] += 1

least_occupied = np.argsort(occupied_times)[:6]
print(f"Least-occupied LST hours: {least_occupied}")

# Output schedule CSV
schedule_df.to_csv('MeerKAT_MultiDay_Schedule.csv', index=False)

print("Multi-day schedule created successfully: MeerKAT_MultiDay_Schedule.csv")


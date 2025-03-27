import pandas as pd
import numpy as np

# Load approved observations
df = pd.read_csv('/mnt/data/Observations 2025 - 2025.observations.csv')

# Convert LST strings to float (hours)
def lst_to_hours(lst_str):
    h, m = map(int, lst_str.split(':'))
    return h + m/60

df['Min_LST'] = df['Min_LST'].apply(lst_to_hours)
df['Max_LST'] = df['Max_LST'].apply(lst_to_hours)

# Sort observations by earliest possible start
df = df.sort_values(by='Min_LST').reset_index(drop=True)

schedule = []
current_LST = 0.0
unscheduled = df.copy()

# Helper to handle LST wrap-around
def wrap_around(lst_time):
    return lst_time % 24

# Helper to check visibility constraints
def fits_constraints(obs, start_time):
    end_time = wrap_around(start_time + obs['Duration_hours'])
    min_lst = obs['Min_LST']
    max_lst = obs['Max_LST']

    crosses_boundary = min_lst > max_lst
    if crosses_boundary:
        visible = start_time >= min_lst or start_time <= max_lst
    else:
        visible = min_lst <= start_time <= max_lst

    if not visible:
        return False

    if obs['Night_only'] == 'Yes':
        if not (8 <= start_time <= 20 and 8 <= end_time <= 20):
            return False

    if obs['Avoid_Sunrise_Sunset'] == 'Yes':
        if (start_time <= 6 <= end_time) or (start_time <= 18 <= end_time):
            return False

    return True

# Schedule observations
def schedule_observations(unscheduled):
    global current_LST
    while not unscheduled.empty:
        candidates = unscheduled[unscheduled.apply(lambda obs: fits_constraints(obs, current_LST), axis=1)]
        if not candidates.empty:
            obs = candidates.iloc[0]
            start_time = current_LST
            end_time = wrap_around(start_time + obs['Duration_hours'])
            schedule.append({
                'Observation_ID': obs['Observation_ID'],
                'Start_LST': start_time,
                'End_LST': end_time,
                'Band': obs['Band'],
                'Night_only': obs['Night_only'],
                'Visibility_window': f"{obs['Min_LST']:.2f}-{obs['Max_LST']:.2f}",
                'Gap_after_LST': None
            })
            current_LST = end_time
            unscheduled = unscheduled.drop(obs.name)
        else:
            next_start = unscheduled['Min_LST'].min()
            schedule.append({
                'Observation_ID': '(Gap)',
                'Start_LST': current_LST,
                'End_LST': next_start,
                'Band': '',
                'Night_only': '',
                'Visibility_window': '',
                'Gap_after_LST': ''
            })
            current_LST = next_start

schedule_observations(unscheduled)

# Convert schedule to DataFrame
schedule_df = pd.DataFrame(schedule)

# Identify gaps explicitly
schedule_df['Gap_after_LST'] = schedule_df['End_LST'].shift(-1) - schedule_df['End_LST']
schedule_df['Gap_after_LST'] = schedule_df['Gap_after_LST'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) and x > 0 else '')

# Output schedule CSV
schedule_df.to_csv('MeerKAT_Schedule.csv', index=False)

print("Schedule created successfully: MeerKAT_Schedule.csv")


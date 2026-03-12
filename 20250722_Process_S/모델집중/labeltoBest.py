import os
import pandas as pd
import numpy as np
import random

mouse_path = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S\HDAC4SA\C4SAABF20101'
group_name = "HDAC4SA"
mouse_id = "C4SAABF20101"
states = ['W', 'NR', 'R'] # 수면 단계 정의

# ---------------------------------------------------
def add_day_night_label(df):
    df['Epoch_in_cycle'] = df['Epoch#'] % 4320  # 하루 주기 - 파일 내에 에폭이 4320보다 큰 경우도 존재하기에 추가함. 
    df['TimeOfDay'] = df['Epoch_in_cycle'].apply(lambda x: 'Day' if x < 2160 else 'Night')
    return df

# def detect_microsleep(stage_series): # 마이크로슬립 탐지 함수 -> 생각과 다르게 작동 -> 생략 
#     microsleep = [False] * len(stage_series)
#     stage = stage_series.values
#     for i in range(1, len(stage)-2): #
#         if stage[i-1] == 'W' and stage[i+1] == 'W' and stage[i] in ('NR', 'R'): 
#             microsleep[i] = True
#         elif stage[i-1] == 'W' and stage[i] in ('NR', 'R') and stage[i+1] in ('NR', 'R') and stage[i+2] == 'W':
#             microsleep[i] = True
#             microsleep[i+1] = True
#     return microsleep

def compute_transition_matrix(df_subset): # 전이 행렬 계산 함수
    transition_counts = {s1: {s2: 0 for s2 in states} for s1 in states}
    total_counts = {s1: 0 for s1 in states}
    prev_stage = df_subset.iloc[0]['Stage']
    for stage in df_subset['Stage'].iloc[1:]:
        if prev_stage in states and stage in states:
            transition_counts[prev_stage][stage] += 1
            total_counts[prev_stage] += 1
        prev_stage = stage
    rows = []
    for s1 in states:
        total = total_counts[s1]
        row = {s2: transition_counts[s1][s2] / total if total > 0 else 0 for s2 in states}
        rows.append(pd.Series(row, name=s1))
    return pd.DataFrame(rows)

def simulate_sleep(transitions, steps=5000, initial_state=None):
    current = initial_state if initial_state else random.choice(states)  # 랜덤 초기 상태
    sequence = [current]
    for _ in range(steps - 1):
        probs = [transitions[current][s] for s in states]
        current = random.choices(states, weights=probs)[0]
        sequence.append(current)
    return sequence

def compute_empirical_matrix(seq): # 시퀀스에서 전이 행렬 계산 함수
    counts = {s1: {s2: 0 for s2 in states} for s1 in states}
    totals = {s1: 0 for s1 in states}
    for a, b in zip(seq[:-1], seq[1:]):
        if a in states and b in states:
            counts[a][b] += 1
            totals[a] += 1
    return {
        s1: {s2: counts[s1][s2] / totals[s1] if totals[s1] > 0 else 0 for s2 in states}
        for s1 in states
    }

def loss(sim, true): 
    return np.mean([(sim[s1][s2] - true[s1][s2]) ** 2 for s1 in states for s2 in states])

def random_transition_matrix():
    # 쥐의 수면은 상태 유지보다는 전이가 많은 경향
    # GPT의 제안 → Dirichlet의 각 α값을 낮춰서 더 분산된 전이 확률 생성
    alpha = [0.7, 0.7, 0.7]  # 각 상태에서 전이가 더 고르게 발생하도록 하기 위함
    return {
        s1: {s2: round(p, 4) for s2, p in zip(states, np.random.dirichlet(alpha))}
        for s1 in states
    }


# ---------------------------------------------------
# 라벨링된 데이터프레임 
all_data = []
for file in os.listdir(mouse_path):
    fpath = os.path.join(mouse_path, file)
    if not file.endswith(('.csv', '.xls', '.xlsx')):
        continue
    try:
        df = pd.read_csv(fpath) if file.endswith('.csv') else pd.read_excel(fpath)
        if not {'Epoch#', 'Stage'}.issubset(df.columns):
            continue
        df['Group'] = group_name
        df['Mouse_ID'] = mouse_id
        df = add_day_night_label(df)
        # df['MicroSleep'] = detect_microsleep(df['Stage'])
        all_data.append(df)
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

if not all_data:
    raise RuntimeError("No valid data found.")

full_df = pd.concat(all_data, ignore_index=True)

# 저장
save_path = os.path.join(mouse_path, f"{mouse_id}_with_labels.csv")
full_df.to_csv(save_path, index=False)

# ---------------------------------------------------
# S전이 행렬 계산 및 저장
print("Calculating transition matrices...")
day_df = full_df[full_df['TimeOfDay'] == 'Day']
night_df = full_df[full_df['TimeOfDay'] == 'Night']

day_matrix = compute_transition_matrix(day_df)
night_matrix = compute_transition_matrix(night_df)

day_matrix.to_csv(os.path.join(mouse_path, f"{mouse_id}_day_transition_matrix.csv"))
night_matrix.to_csv(os.path.join(mouse_path, f"{mouse_id}_night_transition_matrix.csv"))

true_matrix_day = day_matrix.to_dict()
true_matrix_night = night_matrix.to_dict()

# ---------------------------------------------------
# Parameter Recovery (낮/밤 전이 확률 추정)
best_loss = float('inf')
best_params = {}
print("Searching for best parameters...")
for _ in range(500):
    candidate_day = random_transition_matrix()
    candidate_night = random_transition_matrix()

    sim_day = simulate_sleep(candidate_day)
    sim_night = simulate_sleep(candidate_night)

    sim_day_mat = compute_empirical_matrix(sim_day)
    sim_night_mat = compute_empirical_matrix(sim_night)

    total_loss = loss(sim_day_mat, true_matrix_day) + loss(sim_night_mat, true_matrix_night)

    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {'Day': candidate_day, 'Night': candidate_night}

# ---------------------------------------------------
# 결과 출력
print(" Best Parameters for Day:")
for state, transitions in best_params['Day'].items():
    print(f"{state} → {transitions}")

print("\n Best Parameters for Night:")
for state, transitions in best_params['Night'].items():
    print(f"{state} → {transitions}")

print(f"\nMinimum total loss (MSE): {best_loss}")

param_rows = []

for phase in ['Day', 'Night']:
    for from_state in states:
        for to_state in states:
            param_rows.append({
                'TimeOfDay': phase,
                'From': from_state,
                'To': to_state,
                'Probability': best_params[phase][from_state][to_state]
            })

param_df = pd.DataFrame(param_rows)

# 저장 경로
param_save_path = os.path.join(mouse_path, f"{mouse_id}_best_parameters.csv")
param_df.to_csv(param_save_path, index=False)

print(f" best parameter has been saved: {param_save_path}")
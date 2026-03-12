import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 경로
mouse_path = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S\HDAC4SA\C4SAABF20101'
mouse_id = "C4SAABF20101"
states = ['W', 'NR', 'R']

# ---------------------------------------------------
# 파일 불러오기
label_df = pd.read_csv(os.path.join(mouse_path, f"{mouse_id}_with_labels.csv"))
param_df = pd.read_csv(os.path.join(mouse_path, f"{mouse_id}_best_parameters.csv"))

# ---------------------------------------------------
# 전이 파라미터 딕셔너리 형태로 변환
def param_to_dict(df):
    result = {'Day': {}, 'Night': {}}
    for tod in ['Day', 'Night']:
        sub = df[df['TimeOfDay'] == tod]
        for state in states:
            row = sub[sub['From'] == state].sort_values('To')
            result[tod][state] = row['Probability'].tolist()
    return result

trans_dict = param_to_dict(param_df)

# ---------------------------------------------------
# 시뮬레이션 함수
def simulate(transitions, steps=2160, initial_state='W'):
    current = initial_state
    sequence = [current]
    for _ in range(steps - 1):
        probs = transitions[current]
        current = random.choices(states, weights=probs)[0]
        sequence.append(current)
    return sequence

sim_day = simulate(trans_dict['Day'])
sim_night = simulate(trans_dict['Night'])

# ---------------------------------------------------
# 비율 계산 함수
def stage_distribution(seq):
    counts = pd.Series(seq).value_counts(normalize=True)
    return [counts.get(s, 0) for s in states]

true_day = stage_distribution(label_df[label_df['TimeOfDay'] == 'Day']['Stage'])
true_night = stage_distribution(label_df[label_df['TimeOfDay'] == 'Night']['Stage'])

sim_day = stage_distribution(sim_day)
sim_night = stage_distribution(sim_night)

# ---------------------------------------------------
# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
x = np.arange(len(states))
width = 0.35

axes[0].bar(x - width/2, true_day, width, label='Actual')
axes[0].bar(x + width/2, sim_day, width, label='Simulated')
axes[0].set_title('Day')
axes[0].set_xticks(x)
axes[0].set_xticklabels(states)
axes[0].set_ylabel('Proportion')
axes[0].legend()

axes[1].bar(x - width/2, true_night, width, label='Actual')
axes[1].bar(x + width/2, sim_night, width, label='Simulated')
axes[1].set_title('Night')
axes[1].set_xticks(x)
axes[1].set_xticklabels(states)
axes[1].legend()

plt.suptitle(f'Sleep State Comparison for {mouse_id}')
plt.tight_layout()
plt.show()

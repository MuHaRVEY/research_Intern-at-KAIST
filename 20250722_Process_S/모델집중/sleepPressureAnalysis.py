import pandas as pd
import numpy as np
import random
import os

# 시뮬레이션 설정
states = ['W', 'NR', 'R']
epochs_per_day = 4320
epoch_per_hour = 180

# 경로 지정
mouse_id = "C4SAABF20101"
mouse_path = rf'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S\HDAC4SA\{mouse_id}'
param_file = os.path.join(mouse_path, f"{mouse_id}_best_parameters.csv")

# best parameter 파일 불러오기
param_df = pd.read_csv(param_file)

# 낮/밤 파라미터 딕셔너리화
def build_transition_dict(param_df, time_of_day):
    subset = param_df[param_df['TimeOfDay'] == time_of_day]
    result = {s: {} for s in states}
    for _, row in subset.iterrows():
        result[row['From']][row['To']] = row['Probability']
    return result

day_param = build_transition_dict(param_df, 'Day')
night_param = build_transition_dict(param_df, 'Night')

# 낮밤 기반 시뮬레이션 (Light ON at 9:00, 9AM~9PM = Day)
def simulate_day_night_sequence(day_param, night_param, steps=4320):
    sequence = []
    current = random.choice(states)
    for epoch in range(steps):
        hour = (epoch // epoch_per_hour)  # 0~23
        time_of_day = 'Day' if 9 <= hour < 21 else 'Night'
        probs = [day_param if time_of_day == 'Day' else night_param][0][current]
        current = random.choices(states, weights=[probs[s] for s in states])[0]
        sequence.append(current)
    return sequence

# 시퀀스 생성
simulated_sequence = simulate_day_night_sequence(day_param, night_param)

# 시간대별 상태 비율 계산 (1시간 단위)
summary = []
for hour in range(24):
    start = hour * epoch_per_hour
    end = (hour + 1) * epoch_per_hour
    chunk = simulated_sequence[start:end]
    count = {s: chunk.count(s) / len(chunk) for s in states}
    count['Hour'] = hour
    summary.append(count)

summary_df = pd.DataFrame(summary)
summary_df = summary_df[['Hour', 'W', 'NR', 'R']]  # 순서 조정

# 저장
output_path = os.path.join(mouse_path, f"{mouse_id}_simulated_sleep_distribution.csv")
summary_df.to_csv(output_path, index=False)
print(f"Simulated sleep distribution saved to: {output_path}")

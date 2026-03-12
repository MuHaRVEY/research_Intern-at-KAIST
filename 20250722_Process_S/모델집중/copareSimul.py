import pandas as pd
import matplotlib.pyplot as plt
import os

# 기본 설정
mouse_id = "C4SAABF20101"
mouse_path = rf'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S\HDAC4SA\{mouse_id}'

# 파일 경로
real_file = os.path.join(mouse_path, f"{mouse_id}_with_labels.csv")
simul_file = os.path.join(mouse_path, f"{mouse_id}_simulated_sleep_distribution.csv")

# 데이터 불러오기
real_df = pd.read_csv(real_file)
simul_df = pd.read_csv(simul_file)

# 실제 수면 데이터에서 시간(Hour) 추가
real_df['Hour'] = (real_df['Epoch#'] % 4320) // 180  # 180 epoch = 1 hour

# 실제 수면 데이터 시간대별 비율 계산
real_counts = (
    real_df.groupby('Hour')['Stage']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)
real_counts = real_counts[['Hour', 'W', 'NR', 'R']]  # 순서 정렬

# 시각화
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for idx, state in enumerate(['W', 'NR', 'R']):
    axs[idx].plot(real_counts['Hour'], real_counts[state], label='Actual', marker='o')
    axs[idx].plot(simul_df['Hour'], simul_df[state], label='Simulated', marker='x')
    axs[idx].set_ylabel(f'{state} proportion')
    axs[idx].legend()
    axs[idx].set_ylim(0, 1)

axs[2].set_xlabel('Hour of Day')
plt.suptitle(f"Hourly Sleep State Proportions (Real vs Simulated) - {mouse_id}")
plt.tight_layout()
plt.show()

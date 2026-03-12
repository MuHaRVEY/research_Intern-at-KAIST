import os
import pandas as pd
import matplotlib.pyplot as plt

# 분석 대상 루트 폴더 - 일단 local로 가져와서 사용
root_dir = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S'
output_dir = os.path.join(root_dir, "_individual_plots")
os.makedirs(output_dir, exist_ok=True)

# EEG 밴드 정의
eeg_bands = {
    'Delta': list(range(1, 5)),    # 1–4 Hz
    'Theta': list(range(5, 9)),    # 5–8 Hz
    'Alpha': list(range(8, 13)),   # 8–12 Hz
    'Beta':  list(range(13, 31))   # 13–30 Hz
}

# 개체 단위 처리
def process_individual(mouse_id_path, group_name):
    all_df = []
    # 개체 폴더 내 모든 파일 처리
    for fname in os.listdir(mouse_id_path):
        fpath = os.path.join(mouse_id_path, fname)
        if not fname.endswith(('.xls', '.xlsx', '.csv')):
            continue

        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(fpath)
            elif fname.endswith('.xls'):
                df = pd.read_excel(fpath, engine='xlrd')
            else:
                df = pd.read_excel(fpath)

            required_cols = {'Epoch#', 'Stage', 'EMG Integ'}
            required_cols.update({f"{hz}Hz" for hz in range(1, 31)})
            if not required_cols.issubset(df.columns):
                print(f"⚠️ {fname} missing columns. Skipping.")
                continue

            df['Time (hr)'] = df['Epoch#'] * 20 / 3600  # 20초 간격 → 시간
            df['Stage_num'] = df['Stage'].map({'NR': 0, 'R': 1, 'W': 2})

            for band, freqs in eeg_bands.items():
                hz_cols = [f"{hz}Hz" for hz in freqs]
                df[f"{band} Power"] = df[hz_cols].sum(axis=1)

            all_df.append(df)

        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

    # 모든 파일 합치기
    if not all_df:
        print(f"No usable files in {mouse_id_path}")
        return
    df_all = pd.concat(all_df, ignore_index=True)

    # Plot 생성
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"{group_name} - {os.path.basename(mouse_id_path)}")

    # 수면 단계 플롯 (산점도)
    axs[0].scatter(df_all['Time (hr)'], df_all['Stage_num'], c=df_all['Stage_num'], cmap='tab10', s=10)
    axs[0].set_yticks([0, 1, 2])
    axs[0].set_yticklabels(['NR', 'R', 'WAKE'])
    axs[0].set_ylabel('Sleep Stage')

     # EMG 값 시계열 플롯
    axs[1].plot(df_all['Time (hr)'], df_all['EMG Integ'], color='orange')
    axs[1].set_ylabel('EMG Integ')

    # EEG 밴드별 Power 시계열 플롯
    axs[2].plot(df_all['Time (hr)'], df_all['Delta Power'], label='Delta', color='green')
    axs[2].plot(df_all['Time (hr)'], df_all['Theta Power'], label='Theta', color='blue')
    axs[2].plot(df_all['Time (hr)'], df_all['Alpha Power'], label='Alpha', color='purple')
    axs[2].plot(df_all['Time (hr)'], df_all['Beta Power'], label='Beta', color='red')
    axs[2].set_ylabel('EEG Power')
    axs[2].legend()

    #Beta/Delta 비율 시계열 플롯  
    beta_delta_ratio = df_all['Beta Power'] / df_all['Delta Power'].replace(0, pd.NA)
    axs[3].plot(df_all['Time (hr)'], beta_delta_ratio, label='Beta/Delta', color='black')
    axs[3].set_ylabel('Beta/Delta Ratio')
    axs[3].set_xlabel('Time (hr)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 저장 경로 구성
    save_folder = os.path.join(output_dir, group_name)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{os.path.basename(mouse_id_path)}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# 바깥 for문 : 그룹 순회
for group in os.listdir(root_dir):
    group_path = os.path.join(root_dir, group)
    if not os.path.isdir(group_path) or not group.startswith("HDAC"): # HDAC로 시작하는 폴더만 처리
        continue
# 안쪽 for문 : 마우스 ID 순회
    for mouse_id in os.listdir(group_path):
        mouse_path = os.path.join(group_path, mouse_id)
        if os.path.isdir(mouse_path):
            process_individual(mouse_path, group) #각 마우스 ID에 대해 개별 처리 -> 전달

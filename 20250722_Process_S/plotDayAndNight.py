import os
import pandas as pd
import matplotlib.pyplot as plt

# 분석 대상 루트 폴더
root_dir = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S'
output_dir = os.path.join(root_dir, "_individual_daynight_plots")
os.makedirs(output_dir, exist_ok=True)

# EEG 밴드 정의
eeg_bands = {
    'Delta': list(range(1, 5)),
    'Theta': list(range(5, 9)),
    'Alpha': list(range(8, 13)),
    'Beta':  list(range(13, 31))
}

# 낮/밤 구분 함수 (9:00 기준으로 12시간 주기)
def add_day_night_label(df):
    # 한 사이클 = 4320 epoch (12시간 주기, 10초 epoch 기준)
    df['Epoch_in_cycle'] = df['Epoch#'] % 4320
    
    # ZT0이 9시이므로, ZT0~ZT12 = 낮(0~2160), ZT12~ZT24 = 밤(2160~4320)
    df['Day_Night'] = df['Epoch_in_cycle'].apply(lambda x: 'Light' if x < 2160 else 'Dark')
    return df



# # 낮/밤 구분 함수 00시부터의 12시간을 기준으로
# def add_day_night_label(df):
#     df['Epoch_in_cycle'] = df['Epoch#'] % 4320
#     df['Day_Night'] = df['Epoch_in_cycle'].apply(lambda x: 'Day' if x < 2160 else 'Night')
#     return df

# 개체 단위 처리 함수
def process_individual(mouse_id_path, group_name):
    all_df = []

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
                print(f"Missing columns in {fname}. Skipping.")
                continue

            df['Time (hr)'] = df['Epoch#'] * 20 / 3600
            df['Stage_num'] = df['Stage'].map({'NR': 0, 'R': 1, 'W': 2})

            for band, freqs in eeg_bands.items():
                hz_cols = [f"{hz}Hz" for hz in freqs]
                df[f"{band} Power"] = df[hz_cols].sum(axis=1)

            all_df.append(df)

        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

    if not all_df:
        print(f"No usable files in {mouse_id_path}")
        return

    # 전체 데이터 합치기 및 낮밤 구분
    df_all = pd.concat(all_df, ignore_index=True)
    df_all = add_day_night_label(df_all)

    for phase in ['Day', 'Night']:
        df_phase = df_all[df_all['Day_Night'] == phase]
        if df_phase.empty:
            continue

        fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"{group_name} - {os.path.basename(mouse_id_path)} - {phase}")

        axs[0].scatter(df_phase['Time (hr)'], df_phase['Stage_num'], c=df_phase['Stage_num'], cmap='tab10', s=10)
        axs[0].set_yticks([0, 1, 2])
        axs[0].set_yticklabels(['NR', 'R', 'WAKE'])
        axs[0].set_ylabel('Sleep Stage')

        axs[1].plot(df_phase['Time (hr)'], df_phase['EMG Integ'], color='orange')
        axs[1].set_ylabel('EMG Integ')

        axs[2].plot(df_phase['Time (hr)'], df_phase['Delta Power'], label='Delta', color='green')
        axs[2].plot(df_phase['Time (hr)'], df_phase['Theta Power'], label='Theta', color='blue')
        axs[2].plot(df_phase['Time (hr)'], df_phase['Alpha Power'], label='Alpha', color='purple')
        axs[2].plot(df_phase['Time (hr)'], df_phase['Beta Power'], label='Beta', color='red')
        axs[2].set_ylabel('EEG Power')
        axs[2].legend()

        beta_delta_ratio = df_phase['Beta Power'] / df_phase['Delta Power'].replace(0, pd.NA)
        axs[3].plot(df_phase['Time (hr)'], beta_delta_ratio, color='black')
        axs[3].set_ylabel('Beta/Delta Ratio')
        axs[3].set_xlabel('Time (hr)')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_folder = os.path.join(output_dir, group_name)
        os.makedirs(save_folder, exist_ok=True)
        save_name = f"{os.path.basename(mouse_id_path)}_{phase}.png"
        save_path = os.path.join(save_folder, save_name)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

# 그룹 및 개체 순회
for group in os.listdir(root_dir):
    group_path = os.path.join(root_dir, group)
    if not os.path.isdir(group_path) or not group.startswith("HDAC"):
        continue

    for mouse_id in os.listdir(group_path):
        mouse_path = os.path.join(group_path, mouse_id)
        if os.path.isdir(mouse_path):
            process_individual(mouse_path, group)

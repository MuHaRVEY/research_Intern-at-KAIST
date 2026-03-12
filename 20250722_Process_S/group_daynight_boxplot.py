import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 분석 대상 루트 폴더
root_dir = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S'
output_dir = os.path.join(root_dir, "_group_daynight_boxplot")
os.makedirs(output_dir, exist_ok=True)

# EEG 밴드 정의
eeg_bands = {
    'Delta': list(range(1, 5)),
    'Theta': list(range(5, 9)),
    'Alpha': list(range(8, 13)),
    'Beta':  list(range(13, 31))
}

# 낮/밤 구분 함수 (Light ON: 9am)
def add_day_night_label(df):
    df['Epoch_in_cycle'] = df['Epoch#'] % 4320  # 12시간 주기 (10초 epoch)
    df['Day_Night'] = df['Epoch_in_cycle'].apply(lambda x: 'On' if x < 2160 else 'Off')
    return df

# 그룹별 데이터 수집
def collect_group_data(group_path):
    all_df = []
    for mouse_id in os.listdir(group_path):
        mouse_path = os.path.join(group_path, mouse_id)
        if not os.path.isdir(mouse_path):
            continue

        for fname in os.listdir(mouse_path):
            fpath = os.path.join(mouse_path, fname)
            if not fname.endswith(('.csv', '.xls', '.xlsx')):
                continue

            try:
                if fname.endswith('.csv'):
                    df = pd.read_csv(fpath)
                else:
                    df = pd.read_excel(fpath)

                required_cols = {'Epoch#', 'Stage', 'EMG Integ'}
                required_cols.update({f"{hz}Hz" for hz in range(1, 31)})
                if not required_cols.issubset(df.columns):
                    continue

                df['Mouse_ID'] = mouse_id
                df = add_day_night_label(df)

                # EEG 밴드 계산
                for band, freqs in eeg_bands.items():
                    df[f"{band} Power"] = df[[f"{hz}Hz" for hz in freqs]].sum(axis=1)

                df['Beta/Delta'] = df['Beta Power'] / df['Delta Power'].replace(0, pd.NA)
                all_df.append(df)

            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue

    if all_df:
        return pd.concat(all_df, ignore_index=True)
    else:
        return None

# Boxplot 시각화
def plot_group_boxplots(group_name, df_group):
    metrics = ['EMG Integ', 'Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power', 'Beta/Delta']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_group, x='Day_Night', y=metric)
        plt.title(f"{group_name} - {metric} by Light/Dark")
        plt.xlabel('Phase')
        plt.ylabel(metric)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{group_name}_{metric.replace('/', '_')}_boxplot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

# 전체 그룹 순회
for group in os.listdir(root_dir):
    group_path = os.path.join(root_dir, group)
    if not os.path.isdir(group_path) or not group.startswith("HDAC"):
        continue

    df_group = collect_group_data(group_path)
    if df_group is not None:
        plot_group_boxplots(group, df_group)

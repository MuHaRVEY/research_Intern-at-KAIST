import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 분석 대상 루트 폴더
root_dir = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S'
output_dir = os.path.join(root_dir, "_boxplots")
os.makedirs(output_dir, exist_ok=True)

# EEG 밴드 정의
eeg_bands = {
    'Delta': list(range(1, 5)),
    'Theta': list(range(5, 9)),
    'Alpha': list(range(8, 13)),
    'Beta':  list(range(13, 31))
}

# 낮/밤 구분 함수 (ZT0 = 9AM 기준)
def add_day_night_label(df):
    df['Epoch_in_cycle'] = df['Epoch#'] % 4320
    df['Day_Night'] = df['Epoch_in_cycle'].apply(lambda x: 'Day' if x < 2160 else 'Night')
    return df

# EEG Power 계산 함수
def compute_eeg_power(df):
    for band, freqs in eeg_bands.items():
        hz_cols = [f"{hz}Hz" for hz in freqs]
        df[f"{band} Power"] = df[hz_cols].sum(axis=1)
    return df

# 마우스 개체 단위 처리
def process_and_plot_box(mouse_id_path, group_name):
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
                continue

            df = compute_eeg_power(df)
            df = add_day_night_label(df)
            all_df.append(df)

        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

    if not all_df:
        return

    df_all = pd.concat(all_df, ignore_index=True)
    mouse_id = os.path.basename(mouse_id_path)

    # 시각화: 낮/밤별 boxplot
    variables = ['Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power', 'EMG Integ']
    for var in variables:
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=df_all, x='Day_Night', y=var, palette="Set2")
        plt.title(f"{mouse_id} - {var} by Day/Night")
        plt.ylabel(var)
        plt.xlabel("Day/Night")
        save_folder = os.path.join(output_dir, group_name)
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{mouse_id}_{var.replace(' ', '_')}_boxplot.png"))
        plt.close()
        print(f"Saved: {mouse_id}_{var}_boxplot")

# 그룹 순회
for group in os.listdir(root_dir):
    group_path = os.path.join(root_dir, group)
    if not os.path.isdir(group_path) or not group.startswith("HDAC"):
        continue

    for mouse_id in os.listdir(group_path):
        mouse_path = os.path.join(group_path, mouse_id)
        if os.path.isdir(mouse_path):
            process_and_plot_box(mouse_path, group)

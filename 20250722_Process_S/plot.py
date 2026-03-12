import os
import pandas as pd
import matplotlib.pyplot as plt

# 분석 대상 루트 폴더
root_dir = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S'

# 저장용 폴더
output_dir = os.path.join(root_dir, "_plots")
os.makedirs(output_dir, exist_ok=True)

# EEG 밴드 정의 raw가 아닌 band별
eeg_bands = {
    'Delta': list(range(1, 5)),    # 1–4 Hz
    'Theta': list(range(5, 9)),    # 5–8 Hz
    'Alpha': list(range(8, 13)),   # 8–12 Hz
    'Beta':  list(range(13, 31))   # 13–30 Hz
}

# 개별 파일 처리 함수
def process_file(filepath):
    print(f" File Processing: {filepath}")
    try:
        # 파일 읽기, 오류 대비하여 확장자 구분해놓기 -제공받은 파일 xls 
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xls'):
            df = pd.read_excel(filepath, engine='xlrd') 
        else:
            df = pd.read_excel(filepath, engine='openpyxl')


        # 필수 컬럼 확인하기
        required_cols = {'Epoch#', 'Stage', 'EMG Integ'}
        required_cols.update({f"{hz}Hz" for hz in range(1, 31)})
        if not required_cols.issubset(df.columns):
            print("⚠️ Missing required columns. Skipping.")
            return

        # 전처리
        df['Time (hr)'] = df['Epoch#'] * 20 / 3600 # 1에폭 = 20초, 1시간 = 3600초이므로 - 실제 시간 단위 변환
        stage_map = {'NR': 0, 'R': 1, 'WAKE': 2} # 수면 단계 매핑 plot을 위한 숫자
        df['Stage_num'] = df['Stage'].map(stage_map) 

        # EEG band별 power 계산 , 뇌파는 주파수별보다 대역별이 일반적이라고 함.
        for band, freqs in eeg_bands.items():
            hz_cols = [f"{hz}Hz" for hz in freqs]
            df[f"{band} Power"] = df[hz_cols].sum(axis=1)

        # 그래프 그리기
        fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        axs[0].scatter(df['Time (hr)'], df['Stage_num'], c=df['Stage_num'], cmap='tab10', s=10)
        axs[0].set_yticks([0, 1, 2])
        axs[0].set_yticklabels(['NR', 'R', 'WAKE'])
        axs[0].set_ylabel('Sleep Stage')
        axs[0].set_title(f"Sleep Stage: {os.path.basename(filepath)}")

        axs[1].plot(df['Time (hr)'], df['EMG Integ'], color='orange')
        axs[1].set_ylabel('EMG Integ')

        axs[2].plot(df['Time (hr)'], df['Delta Power'], label='Delta', color='green')
        axs[2].plot(df['Time (hr)'], df['Theta Power'], label='Theta', color='blue')
        axs[2].plot(df['Time (hr)'], df['Alpha Power'], label='Alpha', color='purple')
        axs[2].plot(df['Time (hr)'], df['Beta Power'], label='Beta', color='red')
        axs[2].set_ylabel('EEG Power')
        axs[2].legend()

        beta_delta_ratio = df['Beta Power'] / df['Delta Power'].replace(0, pd.NA)
        axs[3].plot(df['Time (hr)'], beta_delta_ratio, label='Beta/Delta', color='black')

        axs[3].set_ylabel('Beta/Delta Ratio')
        axs[3].set_xlabel('Time (hr)')

        plt.tight_layout()

        # 저장 경로 구성
        relative_path = os.path.relpath(filepath, root_dir)
        relative_folder = os.path.dirname(relative_path)
        plot_subdir = os.path.join(output_dir, relative_folder)
        os.makedirs(plot_subdir, exist_ok=True)

        save_path = os.path.join(plot_subdir, os.path.splitext(os.path.basename(filepath))[0] + '.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# 전체 파일 재귀적으로 처리
for folderpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith(('.xls', '.xlsx', '.csv')):
            full_path = os.path.join(folderpath, fname)
            print(f"Found file: {full_path}")  # 코드 실행 여부 확인
            process_file(full_path)

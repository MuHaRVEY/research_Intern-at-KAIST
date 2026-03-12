import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 분석 대상 루트 폴더
root_dir = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S'

# 그룹별 데이터를 수집
all_group_data = []

# EEG 스펙트럼 열 자동 생성
hz_cols = [f"{hz}Hz" for hz in range(1, 31)]

# 대상 그룹만 순회
for group in os.listdir(root_dir):
    group_path = os.path.join(root_dir, group)
    if not os.path.isdir(group_path) or not group.startswith("HDAC"):
        continue

    for mouse_id in os.listdir(group_path):
        mouse_path = os.path.join(group_path, mouse_id)
        if not os.path.isdir(mouse_path):
            continue

        for fname in os.listdir(mouse_path):
            fpath = os.path.join(mouse_path, fname)
            if not fname.endswith(('.xls', '.xlsx', '.csv')):
                continue

            try:
                # 파일 불러오기
                if fname.endswith('.csv'):
                    df = pd.read_csv(fpath)
                elif fname.endswith('.xls'):
                    df = pd.read_excel(fpath, engine='xlrd')
                else:
                    df = pd.read_excel(fpath)

                # 필수 컬럼 확인
                if not {'Epoch#', 'Stage'}.issubset(df.columns):
                    continue

                # 그룹 정보 컬럼 추가
                df['Group'] = group
                df['MouseID'] = mouse_id

                # 필요한 열만 유지
                df = df[['Group', 'MouseID', 'Stage']]
                all_group_data.append(df)

            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue

# 모든 데이터 통합
if not all_group_data:
    raise ValueError("No valid data found.")

df_all = pd.concat(all_group_data, ignore_index=True)

# Stage 순서 지정
df_all['Stage'] = pd.Categorical(df_all['Stage'], categories=['W', 'NR', 'R'], ordered=True)

# 스테이지 비율 계산
stage_counts = df_all.groupby(['Group', 'Stage']).size().reset_index(name='Count')
total_counts = df_all.groupby('Group').size().reset_index(name='Total')
stage_proportions = pd.merge(stage_counts, total_counts, on='Group')
stage_proportions['Proportion'] = stage_proportions['Count'] / stage_proportions['Total']

# 시각화
plt.figure(figsize=(8, 6))
sns.barplot(data=stage_proportions, x='Group', y='Proportion', hue='Stage', hue_order=['W', 'NR', 'R'])
plt.title('Proportion of Sleep Stages by Group')
plt.ylabel('Proportion')
plt.xlabel('Group')
plt.legend(title='Sleep Stage')
plt.tight_layout()
plt.savefig(os.path.join(root_dir, "group_stage_proportion.png"))
plt.show()



import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 간단한 낮/밤 구분 함수 (12시간씩)
def get_day_night_label_simple(epoch_num):
    epoch_in_day = epoch_num % 4320  # 20초 * 4320 = 24시간
    return 'Day' if epoch_in_day < 2160 else 'Night'

root_dir = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S'

group_stage_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
group_total_counts = defaultdict(lambda: defaultdict(int))
valid_stages = {'NR', 'R', 'W'}

# 파일 읽기 및 낮/밤에 따른 수면 단계 카운트
for group_name in os.listdir(root_dir):
    group_path = os.path.join(root_dir, group_name)
    if not os.path.isdir(group_path) or not group_name.startswith("HDAC"):
        continue

    for mouse_id in os.listdir(group_path):
        mouse_path = os.path.join(group_path, mouse_id)
        if not os.path.isdir(mouse_path):
            continue

        for file_name in os.listdir(mouse_path):
            if not file_name.endswith((".xls", ".xlsx", ".csv")):
                continue
            file_path = os.path.join(mouse_path, file_name)

            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                if "Stage" not in df.columns or "Epoch#" not in df.columns:
                    continue

                df = df[df['Stage'].isin(valid_stages)].copy()
                df['DayNight'] = df['Epoch#'].apply(get_day_night_label_simple)

                for dn in ['Day', 'Night']:
                    subset = df[df['DayNight'] == dn]
                    for stage in valid_stages:
                        group_stage_counts[group_name][dn][stage] += (subset['Stage'] == stage).sum()
                    group_total_counts[group_name][dn] += len(subset)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue

# 시각화용 데이터 정리
plot_data = []
for group, dn_dict in group_stage_counts.items():
    for dn, stage_dict in dn_dict.items():
        total = group_total_counts[group][dn]
        for stage, count in stage_dict.items():
            proportion = count / total if total > 0 else 0
            group_dn = f"{group}_{dn}"
            plot_data.append({
                'Group': group,
                'DayNight': dn,
                'Group_DayNight': group_dn,
                'Stage': stage,
                'Proportion': proportion
            })

df_plot = pd.DataFrame(plot_data)

# 바플롯 시각화: 낮/밤에 따라 그룹 구분
plt.figure(figsize=(14, 6))
sns.barplot(data=df_plot, x="Group_DayNight", y="Proportion", hue="Stage", ci=None,
            palette="Set2", hue_order=['W', 'NR', 'R'], dodge=True)
plt.title("Proportion of Sleep Stages by Group and Day/Night (12h Split)")
plt.ylabel("Proportion")
plt.xlabel("Group (Day/Night)")
plt.xticks(rotation=45)
plt.legend(title="Sleep Stage")
sns.despine()
plt.tight_layout()
plt.show()

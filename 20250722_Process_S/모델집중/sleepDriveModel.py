from math import e
import os
from unittest import result
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import optuna
from sympy import plot


mouse_path = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S\HDAC4SA\C4SAABF20101'
mouse_id = 'C4SAABF20101'
labeled_file_path = os.path.join(mouse_path, f"{mouse_id}_with_labels.csv")

states = ['W', 'NR', 'R']
scale_day = 5
scale_night = 8

# 수면 드라이브 모델 likelihood 계산
# def simulate_3state_drive_likelihood_v2(param_vector, df, scale_day=scale_day, scale_night=scale_night):
#     inc_W_day, dec_NR_day, dec_R_day, th_WNR_day, th_NRR_day, th_RW_day = param_vector[:6]
#     inc_W_night, dec_NR_night, dec_R_night, th_WNR_night, th_NRR_night, th_RW_night = param_vector[6:]
#     dt = 20 / 3600  # epoch 길이 (시간 단위)
#     drive = 0.4
#     log_likelihood = 0.0
#     state_prev = df.iloc[0]['Stage']

#     def sigmoid(x, scale): return 1 / (1 + np.exp(-x * scale))

#     for i in range(1, len(df)):
#         tod = df.iloc[i]['TimeOfDay']
#         state_true = df.iloc[i]['Stage']
#         if tod == 'Day':
#             inc_W, dec_NR, dec_R = inc_W_day * dt, dec_NR_day * dt, dec_R_day * dt
#             th_WNR, th_NRR, th_RW, scale = th_WNR_day, th_NRR_day, th_RW_day, scale_day
#         else:
#             inc_W, dec_NR, dec_R = inc_W_night * dt, dec_NR_night * dt, dec_R_night * dt
#             th_WNR, th_NRR, th_RW, scale = th_WNR_night, th_NRR_night, th_RW_night, scale_night

#         if state_prev == 'W':
#             drive = min(drive + inc_W, 1.0)
#         elif state_prev == 'NR':
#             drive = max(drive - dec_NR, 0.0)
#         elif state_prev == 'R':
#             drive = max(drive - dec_R, 0.0)

#         if state_prev == 'W':
#             p = sigmoid(drive - th_WNR, scale)
#             predicted = 'NR' if np.random.rand() < p else 'W'
#         elif state_prev == 'NR':
#             p = sigmoid(drive - th_NRR, scale)
#             predicted = 'R' if np.random.rand() < p else 'NR'
#         elif state_prev == 'R':
#             p = sigmoid(drive - th_RW, scale)
#             predicted = 'W' if np.random.rand() < p else 'R'

#         log_likelihood += np.log(0.9 if predicted == state_true else 0.1)
#         state_prev = predicted

#     return -log_likelihood

# 모델 피팅 함수
# def fit_3state_drive_model(df):
#     init_params = [0.03, 0.04, 0.03, 0.3, 0.4, 0.6, 0.02, 0.05, 0.04, 0.4, 0.5, 0.7]
#     bounds = [
#         (0.005, 0.06), (0.01, 0.12), (0.01, 0.12),  # Day
#         (0.2, 0.6), (0.3, 0.7), (0.4, 0.8),
#         (0.005, 0.06), (0.01, 0.12), (0.01, 0.12),  # Night
#         (0.1, 0.5), (0.2, 0.6), (0.3, 0.7)
#     ]
#     result = minimize(simulate_3state_drive_likelihood_v2, init_params, args=(df,), method='L-BFGS-B', bounds=bounds)

#     if not result.success:
#         raise RuntimeError("Optimization failed: " + result.message)
#     p = result.x
#     return {
#         'Day': {
#             'inc_W': round(p[0], 4), 'dec_NR': round(p[1], 4), 'dec_R': round(p[2], 4),
#             'θ_WNR': round(p[3], 4), 'θ_NRR': round(p[4], 4), 'θ_RW': round(p[5], 4), 'scale': scale_day
#         },
#         'Night': {
#             'inc_W': round(p[6], 4), 'dec_NR': round(p[7], 4), 'dec_R': round(p[8], 4),
#             'θ_WNR': round(p[9], 4), 'θ_NRR': round(p[10], 4), 'θ_RW': round(p[11], 4), 'scale': scale_night
#         },
#         'log_likelihood': round(-result.fun, 4)
#     }


# 시뮬레이션 함수
def simulate_drive_sequence(df, params):
    def sigmoid(x, scale): return 1 / (1 + np.exp(-x * scale))
    drive = 0.4
    drive_seq, predicted_seq = [], []
    state_prev = df.iloc[0]['Stage']

    for i in range(len(df)):
        tod = df.iloc[i]['TimeOfDay']
        pset = params['Day'] if tod == 'Day' else params['Night']

        inc_W = pset['inc_W']
        dec_NR = pset['dec_NR']
        dec_R = pset['dec_R']
        th_WNR, th_NRR, th_RW, scale = pset['θ_WNR'], pset['θ_NRR'], pset['θ_RW'], pset['scale']

        if state_prev == 'W':
            drive = min(drive + inc_W, 1.0)
        elif state_prev == 'NR':
            drive = max(drive - dec_NR, 0.0)
        elif state_prev == 'R':
            drive = max(drive - dec_R, 0.0)

        if state_prev == 'W': #다음 상태 전이 확률 계산 
            p = sigmoid(drive - th_WNR, scale)
            predicted = 'NR' if np.random.rand() < p else 'W'
        elif state_prev == 'NR':
            p = sigmoid(drive - th_NRR, scale)
            predicted = 'R' if np.random.rand() < p else 'NR'
        elif state_prev == 'R':
            p = sigmoid(drive - th_RW, scale)
            predicted = 'W' if np.random.rand() < p else 'R'
        else:
            predicted = state_prev

        drive_seq.append(drive)
        predicted_seq.append(predicted)
        state_prev = predicted

    df = df.copy()
    df['SleepDrive'] = drive_seq
    df['PredictedStage'] = predicted_seq
    return df

# Optuna를 이용한 자동 최적화 함수
def run_optuna_param_search(df, n_trials=100, n_repeat=5):
    def objective(trial):
        np.random.seed(42)  # 재현성을 위해 시드 설정
        params = [
            trial.suggest_float("inc_W_day", 0.002, 0.07), # 맨 뒤의 숫자 2개는 day와 night의 scale을 의미
            trial.suggest_float("dec_NR_day", 0.005, 0.15), # 
            trial.suggest_float("dec_R_day", 0.005, 0.15),
            trial.suggest_float("th_WNR_day", 0.1, 0.7),
            trial.suggest_float("th_NRR_day", 0.1, 0.8),
            trial.suggest_float("th_RW_day", 0.1, 0.9),
            trial.suggest_float("scale_day", 2.0, 6.0),

            trial.suggest_float("inc_W_night", 0.002, 0.07),
            trial.suggest_float("dec_NR_night", 0.005, 0.15),
            trial.suggest_float("dec_R_night", 0.005, 0.15),
            trial.suggest_float("th_WNR_night", 0.05, 0.6),
            trial.suggest_float("th_NRR_night", 0.05, 0.7),
            trial.suggest_float("th_RW_night", 0.05, 0.9),
            trial.suggest_float("scale_night", 2.0, 6.0),
        ]
        # 평가 반복 횟수
        n_repeat = 10
        macro_f1_list = []
        rem_f1_list = []

        # 파라미터 dict 구성
        fitted_params = {
            'Day': {
                'inc_W': params[0], 'dec_NR': params[1], 'dec_R': params[2],
                'θ_WNR': params[3], 'θ_NRR': params[4], 'θ_RW': params[5], 'scale': scale_day
            },
            'Night': {
                'inc_W': params[6], 'dec_NR': params[7], 'dec_R': params[8],
                'θ_WNR': params[9], 'θ_NRR': params[10], 'θ_RW': params[11], 'scale': scale_night
            }
        }
        f1_scores = []
        try:
            for _ in range(n_repeat):
                df_sim = simulate_drive_sequence(df, fitted_params)
                y_true = df['Stage']
                y_pred = df_sim['PredictedStage']
                report = classification_report(y_true, y_pred, labels=states, output_dict=True)
                macro_f1_list.append(report['macro avg']['f1-score'])
                rem_f1_list.append(report['R']['f1-score'])

            # 평균 계산
            macro_f1 = np.mean(macro_f1_list)
            rem_f1 = np.mean(rem_f1_list)
            rem_weight = 2.0  # REM stage를 더 중요하게 여기는 가중치

            # 가중 평균으로 반환
            return macro_f1 + rem_weight * rem_f1
        except:
            return 0.0

    # 최적화 시작
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n Best trial:")
    print(f"  Macro F1-score: {study.best_value:.4f}")
    print("  Parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params

# 실행
df = pd.read_csv(labeled_file_path)[['Epoch#', 'Stage', 'TimeOfDay']]
df.columns = ['Epoch', 'Stage', 'TimeOfDay']
# fit_result = fit_drive_model(df)
# fit_result = fit_3state_drive_model_v2(df)


# # result_path = os.path.join(mouse_path, f"{mouse_id}_fitted_drive_parameters.json")
# result_path = os.path.join(mouse_path, f"{mouse_id}_fitted_3state_drive_parameters.json")
# with open(result_path, 'w') as f:
#     json.dump(fit_result, f, indent=2)

# print("Fitted parameters:")
# print(json.dumps(fit_result, indent=2))
# print(f"Saved to: {result_path}")



#--- plot ----
def plot_sleep_drive(df):
    color_map = {'W': 'orange', 'NR': 'blue', 'R': 'green'}

    plt.figure(figsize=(18, 6))

    # Plot sleep drive
    plt.subplot(2, 1, 1)
    plt.plot(df['Epoch'], df['SleepDrive'], label='Sleep Drive', color='black')
    for i in range(len(df)):
        if df['TimeOfDay'].iloc[i] == 'Night':
            plt.axvspan(df['Epoch'].iloc[i], df['Epoch'].iloc[i]+1, color='gray', alpha=0.1)
        elif df['TimeOfDay'].iloc[i] == 'Day':
            plt.axvspan(df['Epoch'].iloc[i], df['Epoch'].iloc[i]+1, color='yellow', alpha=0.1)
    plt.ylabel('Sleep Drive')
    plt.title('Sleep Drive Over Time (Day/Night shaded)')
    plt.grid(True)

    # Plot predicted vs actual stage
    plt.subplot(2, 1, 2)
    plt.plot(df['Epoch'], [states.index(s) for s in df['Stage']], label='Actual', alpha=0.5)
    plt.plot(df['Epoch'], [states.index(s) for s in df['PredictedStage']], label='Predicted', linestyle='--')
    plt.yticks([0, 1, 2], ['W', 'NR', 'R'])
    plt.xlabel('Epoch')
    plt.ylabel('Stage')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 파일 불러오기
df = pd.read_csv(f"{mouse_path}/{mouse_id}_with_labels.csv")
df = df.rename(columns={'Epoch#': 'Epoch'})

# # Sleep drive 시퀀스 생성
# fitted_params = fit_result
# # df = simulate_drive_sequence(df, fitted_params)
# df = simulate_drive_sequence_soft_v2(df, fitted_params)
# # df = simulate_drive_sequence_soft(df, fitted_params, scale_day=5, scale_night=10)

# plot_sleep_drive(df)

# 평가를 평균을 통해 모델 성능을 확인
def evaluate_model_multiple(df, params, scale=10, n_repeat=20):
    from sklearn.metrics import classification_report, confusion_matrix

    all_accuracy = []
    all_reports = []
    cm_sum = np.zeros((3, 3), dtype=int)
    labels = ['W', 'NR', 'R']

    for i in range(n_repeat):
        df_sim = simulate_drive_sequence(df, params)
        # df_sim = simulate_drive_sequence_soft(df, params, scale_day=5, scale_night=10)
        y_true = df['Stage']
        y_pred = df_sim['PredictedStage']

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        all_accuracy.append(acc)

        # Classification report (macro avg 등)
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        all_reports.append(report)

        # Confusion Matrix 누적
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum += cm

    # 평균 출력
    mean_acc = np.mean(all_accuracy)
    std_acc = np.std(all_accuracy)
    mean_macro_f1 = np.mean([r['macro avg']['f1-score'] for r in all_reports])

    print(f"\n[Repeated Evaluation x{n_repeat}]")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean Macro F1-score: {mean_macro_f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=3))

    # 평균 confusion matrix 시각화
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_sum // n_repeat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Mean Confusion Matrix (x{n_repeat})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


# === Optuna로 최적 파라미터 탐색 ===
best_params_flat = run_optuna_param_search(df, n_trials=100)

# === best_params를 기존 모델 형식으로 변환 ===
fitted_params = {
    'Day': {
        'inc_W': best_params_flat['inc_W_day'],
        'dec_NR': best_params_flat['dec_NR_day'],
        'dec_R': best_params_flat['dec_R_day'],
        'θ_WNR': best_params_flat['th_WNR_day'],
        'θ_NRR': best_params_flat['th_NRR_day'],
        'θ_RW': best_params_flat['th_RW_day'],
        'scale': scale_day
    },
    'Night': {
        'inc_W': best_params_flat['inc_W_night'],
        'dec_NR': best_params_flat['dec_NR_night'],
        'dec_R': best_params_flat['dec_R_night'],
        'θ_WNR': best_params_flat['th_WNR_night'],
        'θ_NRR': best_params_flat['th_NRR_night'],
        'θ_RW': best_params_flat['th_RW_night'],
        'scale': scale_night
    }
}

# 시뮬레이션 실행
# df_simulated = simulate_drive_sequence_soft_v2(df, fitted_params)

# === Sleep drive 시퀀스 생성 및 저장 ===
df = simulate_drive_sequence(df, fitted_params)

# 시각화
# plot_sleep_drive(df_simulated)
plot_sleep_drive(df)
                 
evaluate_model_multiple(df, fitted_params) 

# 추가로 저장
optuna_param_path = os.path.join(mouse_path, f"{mouse_id}_optuna_best_params.json")
with open(optuna_param_path, 'w') as f:
    json.dump(fitted_params, f, indent=2)
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import optuna

mouse_path = r'C:\Users\rkddn\OneDrive\바탕 화면\20250722_Process_S\HDAC4SA\C4SAABF20101'
mouse_id = 'C4SAABF20101'
labeled_file_path = os.path.join(mouse_path, f"{mouse_id}_with_labels.csv")

states = ['W', 'NR', 'R']
# state_idx = {'W': 0, 'NR': 1, 'R': 2}
scale_day = 5
scale_night = 8 
#현재는 밤을 더 크게 설정하여 drive의 변화가 더 민감하게 반영되도록 해보았음.
# 한쪽 스케일이 조금 더 크다고 해서 별다른 의미가 결과가 나타나지 않는듯 함.

# goal 낮/밤에 서로 다른 파라미터를 사용하는 모델을 만들어야 한다.
# 쥐는 microsleep을 한다.
# 이 때 잠의 빈도가 낮/밤에 따라 바뀌는 부분을 별도의 파라미터 또는 hyperparameter로 넣어야 한다.

# 현재 모델은 낮/밤에 따라 파라미터가 다르지 않음
# 따라서, 낮/밤에 따라 파라미터를 다르게 설정할 수 있도록 수정해야 한다.

# 구현 방향이 잘못된듯하다.
# 현재는 단순히 drive를 증가시키고, 감소시키는 방식으로 구현되어 있음
# 어떻게 해야할까????

#----------------------------------------------------------


# 시그모이드 함수 
def sigmoid(x, scale):
    return 1 / (1 + np.exp(-x * scale)) #x가 0보다 크면 1에 가까워지고, 작으면 0에 가까워지는 함수
#즉 하단의 sigmoid(drive - pset['θ_WNR'], scale) 는 drive가 pset['θ_WNR']보다 크면 1에 가까워지고: 전이 확률 증가, 작으면 0에 가까워져 전이 확률 감소

# 시뮬레이션 
def simulate_drive_sequence_sigmoid_full(df, params):
    drive = 0.4 #Sleep drive 초기값을 0.4로 해놓음 모델의 한계가 분명하다면 "이 또한 튜닝으로 바꾸어야 할 것."
    drive_seq, predicted_seq = [], [] # 각 epoch에서 Sleep drive와 예측된 상태를 저장할 리스트
    state_prev = df.iloc[0]['Stage'] # 첫 번째 epoch, 실제 수면 상태로 초기화

    for i in range(len(df)): #에폭 반복하며 수행
        tod = df.iloc[i]['TimeOfDay'] 
        pset = params['Day'] if tod == 'Day' else params['Night'] #알맞은 파라미터 셋 로딩
        scale = pset['scale'] 

        inc_W, dec_NR, dec_R = pset['inc_W'], pset['dec_NR'], pset['dec_R'] # 각 상태에 따른 졸림 증가/감소 값
        if state_prev == 'W': #wake에는 drive 증가 : 졸림 누적
            drive = min(drive + inc_W, 1.0)
        elif state_prev == 'NR': #non-REM에는 drive 감소 : 졸림 해소
            drive = max(drive - dec_NR, 0.0)
        elif state_prev == 'R': #REM에는 drive 감소 : 졸림 해소
            drive = max(drive - dec_R, 0.0) #min,max를 사용하여 drive가 0.0 ~ 1.0 사이에 있도록 제한

        if state_prev == 'W': #다음 상태 전이 확률 계산
            p_NR = sigmoid(drive - pset['θ_WNR'], scale)
            p_R = sigmoid(drive - pset['θ_WR'], scale)
            p_W = max(1.0 - p_NR - p_R, 0.0)
            probs = [p_W, p_NR, p_R] # 다음 상태로 전이될 확률을 담고 있는 리스트
        elif state_prev == 'NR':
            p_W = sigmoid(drive - pset['θ_NRW'], scale)
            p_R = sigmoid(drive - pset['θ_NRR'], scale)
            p_NR = max(1.0 - p_W - p_R, 0.0)
            probs = [p_W, p_NR, p_R]
        elif state_prev == 'R':
            p_W = sigmoid(drive - pset['θ_RW'], scale)
            p_NR = sigmoid(drive - pset['θ_RNR'], scale)
            p_R = max(1.0 - p_W - p_NR, 0.0)
            probs = [p_W, p_NR, p_R] 

        probs = np.clip(probs, 0.001, 1.0) # 확률이 0이 되는 것을 방지하기 위해 최소값을 0.001로 설정
        probs /= np.sum(probs) # 1이 되도록 확률을 정규화
        next_state = np.random.choice(states, p=probs) # 다음 상태 샘플링

        drive_seq.append(drive)
        predicted_seq.append(next_state)
        state_prev = next_state

    df = df.copy()
    df['SleepDrive'] = drive_seq
    df['PredictedStage'] = predicted_seq
    return df

# === Optuna 목적 함수 ===
def run_optuna_sigmoid_full(df, n_trials=100, n_repeat=10):#trial마다 10번 평가하여 평균
    def objective(trial):
        np.random.seed(42)
        params = { #지정한 범위에서 파라미터 값 제안, scale는 고정
            'Day': {
                'inc_W': trial.suggest_float('inc_W_day', 0.01, 0.07),
                'dec_NR': trial.suggest_float('dec_NR_day', 0.01, 0.15),
                'dec_R': trial.suggest_float('dec_R_day', 0.01, 0.15),
                'θ_WNR': trial.suggest_float('θ_WNR_day', 0.2, 0.8),
                'θ_WR': trial.suggest_float('θ_WR_day', 0.2, 0.8),
                'θ_NRW': trial.suggest_float('θ_NRW_day', 0.2, 0.8),
                'θ_NRR': trial.suggest_float('θ_NRR_day', 0.2, 0.8),
                'θ_RW': trial.suggest_float('θ_RW_day', 0.2, 0.8),
                'θ_RNR': trial.suggest_float('θ_RNR_day', 0.2, 0.8),
                'scale': scale_day
            },
            'Night': {
                'inc_W': trial.suggest_float('inc_W_night', 0.01, 0.07),
                'dec_NR': trial.suggest_float('dec_NR_night', 0.01, 0.15),
                'dec_R': trial.suggest_float('dec_R_night', 0.01, 0.15),
                'θ_WNR': trial.suggest_float('θ_WNR_night', 0.2, 0.8),
                'θ_WR': trial.suggest_float('θ_WR_night', 0.2, 0.8),
                'θ_NRW': trial.suggest_float('θ_NRW_night', 0.2, 0.8),
                'θ_NRR': trial.suggest_float('θ_NRR_night', 0.2, 0.8),
                'θ_RW': trial.suggest_float('θ_RW_night', 0.2, 0.8),
                'θ_RNR': trial.suggest_float('θ_RNR_night', 0.2, 0.8),
                'scale': scale_night
            }
        }
        #평균 Macro F1와 REM F1을 합산하여 최적화 목표로 설정
        macro_f1_list = []
        rem_f1_list = []

        try:
            for _ in range(n_repeat):
                df_sim = simulate_drive_sequence_sigmoid_full(df, params)
                y_true, y_pred = df['Stage'], df_sim['PredictedStage'] #실제값과 예측값 비교용
                report = classification_report(y_true, y_pred, labels=states, output_dict=True) #딕셔너리 형태로 평가 지표들을 추출
                macro_f1_list.append(report['macro avg']['f1-score']) # w, nr, r의 평균 성능 list에 저장
                rem_f1_list.append(report['R']['f1-score']) #  r의 성능 list에 별도로 저장 -> 개수가 적어 가중치가 낮기 때문에 별도로 평가
            return np.mean(macro_f1_list) + 2.0 * np.mean(rem_f1_list)
        except:
            return 0.0

    study = optuna.create_study(direction='maximize') #return값인 f1-score를 최대화하는 방향으로 설정
    study.optimize(objective, n_trials=n_trials) #최적화 수행
    print("\n[Best Trial Result]")
    print(f"Macro F1: {study.best_value:.4f}") # 최적화된 Macro F1 출력
    print(json.dumps(study.best_params, indent=2)) # 최적 파라미터 출력
    return study.best_params

# === 평가 함수 재사용 ===
def evaluate_model(df, params, n_repeat=20):
    accs, reports = [], []
    cm_sum = np.zeros((3, 3), dtype=int)
    for _ in range(n_repeat):
        df_sim = simulate_drive_sequence_sigmoid_full(df, params)
        y_true = df['Stage']
        y_pred = df_sim['PredictedStage']
        accs.append(accuracy_score(y_true, y_pred))
        reports.append(classification_report(y_true, y_pred, labels=states, output_dict=True))
        cm_sum += confusion_matrix(y_true, y_pred, labels=states)

    print(f"\n[Evaluation x{n_repeat}]")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1: {np.mean([r['macro avg']['f1-score'] for r in reports]):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=states, digits=3))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_sum // n_repeat, annot=True, fmt='d', cmap='Blues', xticklabels=states, yticklabels=states)
    plt.title(f'Mean Confusion Matrix (x{n_repeat})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_sleep_drive(df):
   
    plt.figure(figsize=(18, 6))
    # Plot sleep drive// drive가 밤과 낮에 따라 어떻게 변화하는지 시각화
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

    # Plot predicted vs actual stage// 실제 데이터와 예측된 상태 비교
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

# 합쳐 놓은 통합 파일 불러오기
df = pd.read_csv(f"{mouse_path}/{mouse_id}_with_labels.csv")
df = df.rename(columns={'Epoch#': 'Epoch'})

# Optuna 탐색 
best_params = run_optuna_sigmoid_full(df, n_trials=100)


# 변환 및 시뮬레이션
fitted_params = { # 파라미터를 Day/Night로 구분하여 저장 
    'Day': {k.replace('_day', ''): v for k, v in best_params.items() if '_day' in k} | {'scale': scale_day},
    'Night': {k.replace('_night', ''): v for k, v in best_params.items() if '_night' in k} | {'scale': scale_night}
}
df_sim = simulate_drive_sequence_sigmoid_full(df, fitted_params)

# 시각화, 평가
plot_sleep_drive(df_sim)
evaluate_model(df, fitted_params)

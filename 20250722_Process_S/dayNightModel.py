import numpy as np
from sympy import im
import sleep_model_helper as smh


def is_daytime(hour):
    return 9 <= hour < 21  # 09:00–21:00을 'Day'로 간주

def split_day_night_indices(tvec):
    day_indices = []
    night_indices = []
    for i, t in enumerate(tvec):
        hour = t % 24
        if is_daytime(hour):
            day_indices.append(i)
        else:
            night_indices.append(i)
    return day_indices, night_indices

def ss2p_sleep_model_daynight(tvec, paramset_day, paramset_night, cphase):
    # Circadian component
    C = smh.circadian_rhythm(tvec, cphase)

    # 결과 배열 초기화
    s_p = np.zeros_like(tvec)
    w_p = np.zeros_like(tvec)
    V = np.zeros_like(tvec)

    # 낮/밤 인덱스 구분
    day_indices, night_indices = split_day_night_indices(tvec)

    # 낮 파라미터 적용
    if day_indices:
        t_day = tvec[day_indices]
        C_day = C[day_indices]
        V_day = paramset_day[0] * np.cos(t_day * 2 * np.pi / 24 + paramset_day[1])
        V_C_day = V_day - C_day
        s_p_day, w_p_day = calc_sleep_trans(V_C_day, *paramset_day[2:])
        s_p[day_indices] = s_p_day
        w_p[day_indices] = w_p_day
        V[day_indices] = V_day

    # 밤 파라미터 적용
    if night_indices:
        t_night = tvec[night_indices]
        C_night = C[night_indices]
        V_night = paramset_night[0] * np.cos(t_night * 2 * np.pi / 24 + paramset_night[1])
        V_C_night = V_night - C_night
        s_p_night, w_p_night = calc_sleep_trans(V_C_night, *paramset_night[2:])
        s_p[night_indices] = s_p_night
        w_p[night_indices] = w_p_night
        V[night_indices] = V_night

    return s_p, w_p, C, V

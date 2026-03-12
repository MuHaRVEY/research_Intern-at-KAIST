# -*- coding: utf-8 -*-
import os
import json
from glob import glob
from datetime import timedelta
from re import sub
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
import pytz

r"""
목표: Convert Fitbit raw files to 5-min records in some sheets (걸음수, 운동시간 - 가능한가?, 심박수, 수면시간)
전달해주신 raw 데이터들을 활용하여 기존 형식으로 변환하는 코드 만들기, matching the "somday-record / somday-fitbit-record"
시트별 style: [비식별키, 기록일자, 기록시간, 값에 해당하는 열들~~~~,...,...].

-> 제공해주신 파일에--비식별키가 raw file에 없음
일단 기존 샘플 파일에서 비식별키를 읽어오도록 함 (일단 DEFAULT로 둠.) : 나중에 여쭤보고 어떻게 처리할지 질문드리기
-> 굳이 필요 없으므로, 그냥 DEFAULT_KEY로 고정

- 제 환경에서 폴더 구조입니다 - 
C:\Users\rkddn\convertToRecord\
 ├─ goal_samples_file\           <얘는 딱히 필요하지 않음.>
 │    ├─ SOM_1_0000_somday-fitbit-records.xlsx   
 │    └─ SOM_1_0000_somday-records.xlsx          
 ├─ fibit_raw_file\
 │    ├─ drive-download-*.zip   (optional; any number of zip files)
 │    └─ or extracted files like heart_rate_*.csv, steps_*.csv, sleep-*.json
 └─ converted_records\          (결과물 저장 파일)

Notes:
- Timezone: Asia/Seoul 변환? -> 그냥 원래값 그대로 진행하기
- Steps: 5-min sum from steps_*.csv - 걸음수는 5분 동안의 값 합산
- Heart rate: 5-min mean from heart_rate_*.csv - 심박수는 5분 평균
- Sleep time: 5분 동안으로 각각의 값들 (light, deep, rem) 저장
- Output per dataset: one Excel file with three sheets: 걸음수, 심박수, 수면시간 - 현재까지 3개 SHEET 가능?
-> 운동시간도 가능한가??
"""
# ====== 경로 설정 ======
ROOT        = r"C:\Users\rkddn\convertToRecord"
RAW_DIR     = os.path.join(ROOT, "fibit_raw_file")          # raw 파일 폴더
OUT_DIR     = os.path.join(ROOT, "converted_records")        # 결과 저장 폴더
os.makedirs(OUT_DIR, exist_ok=True)

LOCAL_TZ   = pytz.timezone("Asia/Seoul")
DEFAULT_KEY = "DEFAULT"
#"SOM_1_0000"                                   # 샘플을 못 읽으면 이 키 사용

# ====== 유틸 ======
def log(msg): print(msg, flush=True)
def to_utc(ts):  return pd.to_datetime(ts, utc=True, errors="coerce")      # tz-aware UTC
def to_kst_str(ts):
    if not ts: return None
    dt = dateparser.parse(ts)
    if dt.tzinfo is None:
        dt = LOCAL_TZ.localize(dt)
    else:
        dt = dt.astimezone(LOCAL_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ====== 로더 ======
def load_raw_from_dir(in_dir: str):
    # steps
    steps_frames = []
    for fp in sorted(glob(os.path.join(in_dir, "steps_*.csv"))):
        df = pd.read_csv(fp)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if {"timestamp","steps"}.issubset(df.columns):
            steps_frames.append(df[["timestamp","steps"]])
    steps_raw = pd.concat(steps_frames, ignore_index=True) if steps_frames \
                else pd.DataFrame(columns=["timestamp","steps"])

    # heart rate
    hr_frames = []
    for fp in sorted(glob(os.path.join(in_dir, "heart_rate_*.csv"))):
        df = pd.read_csv(fp)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if {"timestamp","beats_per_minute"}.issubset(df.columns):
            hr_frames.append(df[["timestamp","beats_per_minute"]])
    hr_raw = pd.concat(hr_frames, ignore_index=True) if hr_frames \
             else pd.DataFrame(columns=["timestamp","beats_per_minute"])

    # sleep
    sleep_json = ""
    cands = sorted(glob(os.path.join(in_dir, "sleep-*.json")))
    if cands: sleep_json = cands[0]

    # activity_level
    act_csv = ""
    for pat in ["activity_level*.csv", "activity-level*.csv", "activities_level*.csv"]:
        cand = sorted(glob(os.path.join(in_dir, pat)))
        if cand:
            act_csv = cand[0]; break

    return steps_raw, hr_raw, sleep_json, act_csv

# ====== 5분 리샘플 ======
def resample_5min_steps(df_steps: pd.DataFrame) -> pd.DataFrame:
    if df_steps.empty:
        return pd.DataFrame(columns=["timestamp","value"])
    idx = to_utc(df_steps["timestamp"])
    s   = pd.Series(df_steps["steps"].values, index=idx)
    # UTC 5분, 좌측 경계/라벨 통일
    out = s.resample("5min", label="left", closed="left").sum().reset_index()
    out.columns = ["timestamp","value"]
    return out

def resample_5min_hr(df_hr: pd.DataFrame) -> pd.DataFrame:
    if df_hr.empty:
        return pd.DataFrame(columns=["timestamp","value"])
    idx = to_utc(df_hr["timestamp"])
    s   = pd.Series(df_hr["beats_per_minute"].values, index=idx)
    out = s.resample("5min", label="left", closed="left").mean().round(1).reset_index()
    out.columns = ["timestamp","value"]
    return out

# ====== 운동시간(분) : activity_level 기반 ======
def active_minutes_5min_from_activity_level(csv_path: str, steps_grid_index: pd.Index) -> pd.DataFrame:
    """
    activity_level CSV → 5분 운동시간(분)
    - UTC 5분, label/closed='left' (걸음수와 동일)
    - 활동 초 = (level != 'sedentary') 동안의 지속시간(초)
      · duration/seconds 열 있으면 사용, 없으면 next_ts - ts (마지막 표본은 60초로 가정)
    - steps_grid_index 로 reindex → 타임스탬프 집합 완전 일치 + 빈 구간은 0
    """
    if not csv_path or not os.path.exists(csv_path):
        # activity 파일이 없다면 걸음수 그리드에 0으로 채워 리턴
        return pd.DataFrame({"timestamp": steps_grid_index, "value": pd.Series(0, index=steps_grid_index, dtype="Int64")})

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    ts_col  = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
    lvl_col = next((c for c in ["activity_level","level","intensity","activity"] if c in df.columns), None)
    dur_col = next((c for c in ["duration","duration_s","seconds","sec"] if c in df.columns), None)
    if ts_col is None or lvl_col is None:
        return pd.DataFrame({"timestamp": steps_grid_index, "value": pd.Series(0, index=steps_grid_index, dtype="Int64")})

    ts  = to_utc(df[ts_col])
    lvl = df[lvl_col].astype(str).str.strip().str.lower()

    if dur_col is not None:
        dur_s = pd.to_numeric(df[dur_col], errors="coerce").fillna(0).astype(float)
    else:
        dt_next = ts.shift(-1)
        dur_s = (dt_next - ts).dt.total_seconds().fillna(60).clip(lower=0, upper=60)

    active_sec = np.where(~lvl.isin(["sedentary"]), dur_s, 0.0)
    s = pd.Series(active_sec, index=ts)

    sec_5  = s.resample("5min", label="left", closed="left").sum(min_count=1)
    sec_5  = sec_5.reindex(steps_grid_index, fill_value=0)      # ← 걸음수 그리드로 강제
    mins_5 = (sec_5 // 60).astype("Int64")

    out = mins_5.reset_index()
    out.columns = ["timestamp","value"]
    return out

# ====== 수면 요약 (KST, summary 우선) ======
def build_sleep_summary_table(sleep_json_path: str, subject_key: str) -> pd.DataFrame:
    cols = ["비식별키","기록일자","기록시간","총 수면시간 (분)","깬 시간 (분)","램수면 (분)","얕은 수면 (분)","깊은 수면 (분)","깬 횟수"]
    if not sleep_json_path:
        return pd.DataFrame(columns=cols)

    with open(sleep_json_path, "r", encoding="utf-8") as f:
        sleeps = json.load(f)

    rows = []
    for ent in sleeps if isinstance(sleeps, list) else []:
        date_of_sleep = ent.get("dateOfSleep")
        end_str = to_kst_str(ent.get("endTime"))

        levels  = (ent.get("levels") or {})
        summary = (levels.get("summary") or {})
        data    = levels.get("data") or []
        short   = levels.get("shortData") or []
        typ     = (ent.get("type") or "").lower()

        if summary:
            if typ == "stages":
                mins_light = int((summary.get("light") or {}).get("minutes", 0))
                mins_deep  = int((summary.get("deep")  or {}).get("minutes", 0))
                mins_rem   = int((summary.get("rem")   or {}).get("minutes", 0))
                mins_wake  = int((summary.get("wake")  or {}).get("minutes", 0))
                mins_total = mins_light + mins_deep + mins_rem
                wake_count = int((summary.get("wake")  or {}).get("count", 0))
            else:  # classic
                mins_asleep   = int((summary.get("asleep")   or {}).get("minutes", 0))
                mins_awake    = int((summary.get("awake")    or {}).get("minutes", 0))
                mins_restless = int((summary.get("restless") or {}).get("minutes", 0))
                mins_total = mins_asleep
                mins_wake  = mins_awake + mins_restless
                mins_light = mins_deep = mins_rem = 0
                wake_count = int((summary.get("awake") or {}).get("count", 0)) + int((summary.get("restless") or {}).get("count", 0))
        else:
            sec = {"wake":0,"awake":0,"restless":0,"light":0,"deep":0,"rem":0,"asleep":0}
            wake_count = 0
            def acc(seglist):
                nonlocal wake_count
                for seg in seglist:
                    lvl = str(seg.get("level","")).lower()
                    s = int(seg.get("seconds",0))
                    if lvl in sec: sec[lvl] += s
                    if lvl in {"wake","awake"}: wake_count += 1
            acc(data); acc(short)

            mins_light = sec["light"]//60
            mins_deep  = sec["deep"]//60
            mins_rem   = sec["rem"]//60
            if (mins_light + mins_deep + mins_rem) > 0:
                mins_total = mins_light + mins_deep + mins_rem
                mins_wake  = (sec["wake"] + sec["awake"] + sec["restless"])//60
            else:
                mins_total = sec["asleep"]//60
                mins_wake  = (sec["awake"] + sec["restless"])//60

        rows.append({
            "비식별키": subject_key,
            "기록일자": date_of_sleep,
            "기록시간": end_str,
            "총 수면시간 (분)": mins_total,
            "깬 시간 (분)": mins_wake,
            "램수면 (분)": mins_rem,
            "얕은 수면 (분)": mins_light,
            "깊은 수면 (분)": mins_deep,
            "깬 횟수": wake_count
        })

    df = pd.DataFrame(rows, columns=cols)
    df["_dt"] = pd.to_datetime(df["기록시간"], errors="coerce")
    df = df.sort_values(["기록일자","_dt"], ascending=False).drop(columns=["_dt"]).reset_index(drop=True)
    return df

# ====== 표 형태 변환 ======
def to_sheet(df: pd.DataFrame, subject_key: str, value_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["비식별키","기록일자","기록시간", value_name])
    ts = to_utc(df["timestamp"])
    out = pd.DataFrame({
        "비식별키": subject_key,
        "기록일자": ts.dt.strftime("%Y-%m-%d"),
        "기록시간": ts.dt.strftime("%H:%M:%S"),
        value_name: df["value"]
    })
    # 최신이 위로
    return out.sort_values(["기록일자","기록시간"], ascending=False).reset_index(drop=True)

# ====== 엑셀 서식(헤더 일반체 + 자동 너비) ======
def autosize_and_unbold(writer, sheet_name: str, df: pd.DataFrame):
    ws = writer.sheets[sheet_name]
    wb = writer.book
    header_fmt = wb.add_format({"bold": False})
    ws.set_row(0, None, header_fmt)
    for i, col in enumerate(df.columns):
        series = df[col].astype(str)
        width = max([len(str(col))] + series.map(len).tolist()) + 2
        ws.set_column(i, i, width)

# ====== 파이프라인 ======
def process_dataset(dataset_root: str, subject_key: str, out_path: str):
    log(f"[INFO] Processing: {dataset_root}")
    steps_raw, hr_raw, sleep_json, act_csv = load_raw_from_dir(dataset_root)

    # 5분 시리즈 (UTC, 좌측 라벨/경계 통일)
    steps_5 = resample_5min_steps(steps_raw)
    hr_5    = resample_5min_hr(hr_raw)

    # 걸음수의 5분 격자를 '정답 그리드'로 사용 → 운동시간은 반드시 동일 타임스탬프
    steps_grid = to_utc(steps_5["timestamp"])
    active_5   = active_minutes_5min_from_activity_level(act_csv, steps_grid_index=steps_grid)

    # 시트 변환
    sheet_steps   = to_sheet(steps_5,  subject_key, "걸음수")
    sheet_hr      = to_sheet(hr_5,     subject_key, "심박수")
    sheet_active  = to_sheet(active_5, subject_key, "운동시간 (분)")
    sheet_sleep   = build_sleep_summary_table(sleep_json, subject_key)  # 수면 요약(KST)

    # 빈값 보호
    if not sheet_active.empty:
        sheet_active["운동시간 (분)"] = sheet_active["운동시간 (분)"].fillna(0).astype("Int64")

    # 타임스탬프 집합 검증 로그
    st_ts = set(to_utc(steps_5["timestamp"]).dropna())
    ac_ts = set(to_utc(active_5["timestamp"]).dropna())
    if st_ts == ac_ts:
        log("[CHECK] 운동시간과 걸음수의 5분 타임스탬프가 완전히 일치합니다.")
    else:
        log(f"[CHECK] 타임스탬프 불일치 → steps-only:{len(st_ts-ac_ts)}, active-only:{len(ac_ts-st_ts)}")

    # 저장
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        sheet_steps.to_excel( writer, index=False, sheet_name="Fitbit-걸음수")
        sheet_hr.to_excel(    writer, index=False, sheet_name="Fitbit-심박수")
        sheet_active.to_excel(writer, index=False, sheet_name="Fitbit-운동시간")
        sheet_sleep.to_excel( writer, index=False, sheet_name="Fitbit-수면시간")

        autosize_and_unbold(writer, "Fitbit-걸음수",  sheet_steps)
        autosize_and_unbold(writer, "Fitbit-심박수",  sheet_hr)
        autosize_and_unbold(writer, "Fitbit-운동시간", sheet_active)
        autosize_and_unbold(writer, "Fitbit-수면시간", sheet_sleep)

    log(f"[OK] Saved: {out_path}")

def main():
    subject_key = DEFAULT_KEY
    log(f"[KEY] 비식별키 = {subject_key}")
    out_file = os.path.join(OUT_DIR, "somday-fitbit-records-5min-tabs.xlsx")
    process_dataset(RAW_DIR, subject_key, out_file)
    log("[DONE] Raw directory processed.")

if __name__ == "__main__":
    main()
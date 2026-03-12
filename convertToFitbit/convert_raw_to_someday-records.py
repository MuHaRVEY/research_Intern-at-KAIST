# -*- coding: utf-8 -*-
import os, json
from glob import glob
import pandas as pd
import numpy as np
from dateutil import parser as dateparser

# =SUMPRODUCT( (LEFT($A$2:$A$100000,10)="2025-05-08") * --$B$2:$B$100000 )
# 액셀에서 총합이 맞는 값인지 확인용 함수 

# ===== 경로 =====
ROOT        = r"C:\Users\rkddn\convertToRecord"
RAW_DIR     = os.path.join(ROOT, "fibit_raw_file")       # raw 파일 폴더 (steps_*.csv, activity_level_*.csv …)
OUT_DIR     = os.path.join(ROOT, "converted_records")    # 결과 저장 폴더
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_KEY = "DEFAULT"
#"SOM_1_0000"

def log(m): print(m, flush=True)

# ---- RAW 로딩 ----
def load_steps_raw():
    frames = []
    for fp in sorted(glob(os.path.join(RAW_DIR, "steps_*.csv"))):
        df = pd.read_csv(fp)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if {"timestamp","steps"}.issubset(df.columns):
            frames.append(df[["timestamp","steps"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp","steps"])

def find_activity_level_csv():
    for pat in ["activity_level*.csv", "activity-level*.csv", "activities_level*.csv"]:
        c = sorted(glob(os.path.join(RAW_DIR, pat)))
        if c: return c[0]
    return ""

def load_heart_raw():
    # heart_rate_*.csv에서 timestamp만 모은다 (값은 일단 쓰지 않음 - 총 심박수 값이 뭔지 아직 모르겠으므로).
    frames = []
    for fp in sorted(glob(os.path.join(RAW_DIR, "heart_rate_*.csv"))):
        df = pd.read_csv(fp)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # timestamp 또는 time 열이 있는지만 확인
        ts_col = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
        if ts_col is not None:
            frames.append(df[[ts_col]].rename(columns={ts_col: "timestamp"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp"])


# ---- 하루 단위 합계 ----
def steps_daily_total(df_steps: pd.DataFrame, key: str) -> pd.DataFrame:
    if df_steps.empty:
        return pd.DataFrame(columns=["비식별키","기록일자","총 걸음수"])
    ts = pd.to_datetime(df_steps["timestamp"], errors="coerce")
    df = pd.DataFrame({
        "date": ts.dt.date,
        "steps": pd.to_numeric(df_steps["steps"], errors="coerce").fillna(0)
    })
    daily = df.groupby("date", dropna=True)["steps"].sum().reset_index()
    daily["비식별키"] = key
    daily["기록일자"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    daily["총 걸음수"] = daily["steps"].astype("Int64")
    return daily[["비식별키","기록일자","총 걸음수"]].sort_values("기록일자", ascending=False).reset_index(drop=True)

def active_daily_total(act_csv: str, key: str) -> pd.DataFrame:
    if not act_csv or not os.path.exists(act_csv):
        return pd.DataFrame(columns=["비식별키","기록일자","총 운동시간 (분)"])
    df = pd.read_csv(act_csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    ts_col  = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
    lvl_col = next((c for c in ["activity_level","level","intensity","activity"] if c in df.columns), None)
    dur_col = next((c for c in ["duration","duration_s","seconds","sec"] if c in df.columns), None)
    if ts_col is None or lvl_col is None:
        return pd.DataFrame(columns=["비식별키","기록일자","총 운동시간 (분)"])

    ts  = pd.to_datetime(df[ts_col], errors="coerce")
    lvl = df[lvl_col].astype(str).str.strip().str.lower()

    if dur_col is not None:
        dur_s = pd.to_numeric(df[dur_col], errors="coerce").fillna(0).astype(float)
    else:
        dt_next = ts.shift(-1)
        dur_s = (dt_next - ts).dt.total_seconds().fillna(60).clip(lower=0)

    active_sec = np.where(~lvl.isin(["sedentary"]), dur_s, 0.0)
    tmp = pd.DataFrame({"date": ts.dt.date, "sec": active_sec})
    daily = tmp.groupby("date", dropna=True)["sec"].sum().reset_index()
    daily["비식별키"] = key
    daily["기록일자"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    daily["총 운동시간 (분)"] = (daily["sec"] // 60).astype("Int64")
    return daily[["비식별키","기록일자","총 운동시간 (분)"]].sort_values("기록일자", ascending=False).reset_index(drop=True)

def heart_daily_placeholder(df_hr: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    hr 데이터에서 날짜만 추출하여 일단 생성
    '총 심박수' 값은 일단 공란으로
    """
    cols = ["비식별키","기록일자","총 심박수"]
    if df_hr.empty:
        return pd.DataFrame(columns=cols)

    ts = pd.to_datetime(df_hr["timestamp"], errors="coerce")
    dates = pd.Series(ts.dt.date.dropna().unique())
    if dates.empty:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame({
        "비식별키": key,
        "기록일자": pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
        "총 심박수": ""  # 이건 일단 공란
    })
    # 최신일이 위로
    out = out.sort_values("기록일자", ascending=False).reset_index(drop=True)
    return out

#---- 수면 시간 sheet 용 ----
def _fmt_ymdhm(ts: str) -> str:
    """타임존 변환 없이 문자열→datetime 파싱 후 'YYYY-MM-DD HH:MM'로만 포맷."""
    if not ts:
        return ""
    try:
        dt = dateparser.parse(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)[:16]  # 실패 시 최대한 비슷하게 잘라서 반환

def find_sleep_jsons(raw_dir: str):
    return sorted(glob(os.path.join(raw_dir, "sleep-*.json")))

def build_sleep_sheet_from_json(sleep_json_path: str, subject_key: str) -> pd.DataFrame:
    """
    JSON의 있는 값만 사용해 수면시간 시트 생성.
    - 기록일자: endTime(YYYY-MM-DD HH:MM)
    - 수면시작/종료: startTime/endTime(동일 포맷)
    - 깬횟수: levels.summary 우선, 없으면 data/shortData에서 wake/awake 카운트
    - 수면품질: JSON에 없으므로 빈 문자열
    - 총 수면시간(시간): minutesAsleep/60 (없으면 light+deep+rem 초 합/3600), 소수 1자리
    - 데이터 타입: "Fitbit records"
    """
    cols = ["비식별키","기록일자","수면시작시간","수면종료시간","깬횟수","수면품질","총 수면시간 (시간)","데이터 타입"]
    if not sleep_json_path or not os.path.exists(sleep_json_path):
        return pd.DataFrame(columns=cols)

    with open(sleep_json_path, "r", encoding="utf-8") as f:
        sleeps = json.load(f)

    rows = []
    for ent in sleeps if isinstance(sleeps, list) else []:
        start_iso = ent.get("startTime") or ""
        end_iso   = ent.get("endTime") or ""

        # 깬 횟수
        wake_count = 0
        levels  = ent.get("levels") or {}
        summary = levels.get("summary") or {}

        if summary:
            # stages 포맷: wake.count
            if "wake" in summary and isinstance(summary["wake"], dict):
                wake_count = int(summary["wake"].get("count", 0) or 0)
            # classic 포맷: awake.count + (restless.count available면 더함)
            else:
                awake_cnt    = int((summary.get("awake")    or {}).get("count", 0) or 0)
                restless_cnt = int((summary.get("restless") or {}).get("count", 0) or 0)
                wake_count = awake_cnt + restless_cnt
        else:
            # summary가 없으면 data/shortData에서 level이 wake/awake 인 구간 수를 카운트
            def count_wakes(seglist):
                if not seglist: return 0
                lv = pd.Series([str(s.get("level","")).lower() for s in seglist])
                return int(((lv == "wake") | (lv == "awake")).sum())
            wake_count = count_wakes(levels.get("data")) + count_wakes(levels.get("shortData"))

        # 총 수면시간(시간)
        total_hours = None
        mins_asleep = ent.get("minutesAsleep")
        if mins_asleep is not None and pd.notna(mins_asleep):
            total_hours = round(float(mins_asleep)/60.0, 1)
        else:
            # stages 초 기반 (light/deep/rem)
            secs = {"light":0, "deep":0, "rem":0}
            for seglist in [levels.get("data"), levels.get("shortData")]:
                if not seglist: continue
                for seg in seglist:
                    lvl = str(seg.get("level","")).lower()
                    s   = int(seg.get("seconds", 0) or 0)
                    if lvl in secs: secs[lvl] += s
            total_hours = round((secs["light"]+secs["deep"]+secs["rem"])/3600.0, 1)

        rows.append({
            "비식별키": subject_key,
            "기록일자": _fmt_ymdhm(end_iso) if end_iso else _fmt_ymdhm(start_iso),
            "수면시작시간": _fmt_ymdhm(start_iso),
            "수면종료시간": _fmt_ymdhm(end_iso),
            "깬횟수": int(wake_count),
            "수면품질": "",  # JSON 원천 없음 → 빈칸
            "총 수면시간 (시간)": total_hours if total_hours is not None else "",
            "데이터 타입": "Fitbit records",
        })

    df = pd.DataFrame(rows, columns=cols)
    # 최신 기록일이 위로
    if not df.empty:
        df["_sort"] = pd.to_datetime(df["기록일자"], errors="coerce")
        df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
    return df

# ---- 엑셀 서식(헤더 일반체 + 자동 너비) ----
def autosize_and_unbold(writer, sheet_name: str, df: pd.DataFrame):
    ws = writer.sheets[sheet_name]
    wb = writer.book
    header_fmt = wb.add_format({"bold": False})
    ws.set_row(0, None, header_fmt)
    for i, col in enumerate(df.columns):
        width = max([len(str(col))] + df[col].astype(str).map(len).tolist()) + 2
        ws.set_column(i, i, width)

# ---- 메인 ----
def main():
    key = DEFAULT_KEY
    log(f"[KEY] 비식별키 = {key}")

    steps_raw = load_steps_raw()
    act_csv   = find_activity_level_csv()

    day_steps  = steps_daily_total(steps_raw, key)
    day_active = active_daily_total(act_csv, key)
    # 심박수 시트 (총 심박수는 공란)
    heart_raw = load_heart_raw()
    day_heart = heart_daily_placeholder(heart_raw, key)


    #sleep sheet
    # 찾기
    sleep_files = find_sleep_jsons(RAW_DIR)
    sleep_sheet = pd.DataFrame(columns=["비식별키","기록일자","수면시작시간","수면종료시간","깬횟수","수면품질","총 수면시간 (시간)","데이터 타입"])
    if sleep_files:
        # 여러 파일이 있으면 모두 합치고 정렬
        parts = [build_sleep_sheet_from_json(fp, DEFAULT_KEY) for fp in sleep_files]
        sleep_sheet = pd.concat(parts, ignore_index=True) if parts else sleep_sheet

    out_path = os.path.join(OUT_DIR, "someday-records_from_raw.xlsx")
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        day_steps.to_excel(w,  index=False, sheet_name="Fitbit-걸음수")
        day_active.to_excel(w, index=False, sheet_name="Fitbit-운동시간")
        day_heart.to_excel(w, index=False, sheet_name="Fitbit-심박수")
        sleep_sheet.to_excel(w, index=False, sheet_name="수면시간")
        autosize_and_unbold(w, "Fitbit-걸음수", day_steps)
        autosize_and_unbold(w, "Fitbit-운동시간", day_active)
        autosize_and_unbold(w, "Fitbit-심박수", day_heart)
        autosize_and_unbold(w, "수면시간", sleep_sheet)
    log(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()

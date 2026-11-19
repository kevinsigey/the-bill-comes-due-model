"""
Earnings event study, adapted 1:1 from FIN 559 Lecture 16 / Homework 9
(dividend-announcement event study) -> earnings announcements.

Class method preserved:
  AR(day)   = stock daily return - market daily return (SPY as VWRETD proxy)
  CAR       = sum of AR over an event window (here: reaction day, +1..+3, +1..+5)
  AAR/CAAR  = average AR by event day, cumulated
  Tests     = one-sample t-test of mean CAR vs 0 (scipy.stats.ttest_1samp)
Event indicator swapped: div_chg (+1/-1) -> EPS surprise sign (beat/miss),
with surprise magnitude = (actual - estimate) / |estimate|.

Data: yfinance (prices + reported EPS vs estimate), pulled Mon Oct 26, 2026 after close.
Sample: last 12 reported quarters per ticker (~3 years).
Reaction day: first trading day whose CLOSE reflects the print
  (AMC prints -> next trading day; BMO prints -> same trading day).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from scipy import stats

TICKERS = ["GOOGL","MSFT","AMZN","AAPL","PLTR","XOM","CVX","META",
           "EQT","INTC","AMD","SWK","PBF","LLY"]
N_Q = 12

spy = yf.Ticker("SPY").history(period="4y", auto_adjust=False)
spy_ret = spy["Close"].pct_change()
spy_ret.index = spy_ret.index.tz_localize(None).normalize()

rows = []
for tk in TICKERS:
    t = yf.Ticker(tk)
    try:
        ed = t.get_earnings_dates(limit=N_Q + 4)
    except Exception as e:
        print(tk, "earnings dates failed:", e); continue
    if ed is None or len(ed) == 0: continue
    ed.index = pd.to_datetime(ed.index).tz_localize(None)
    ed = ed[ed["Reported EPS"].notna()].head(N_Q)
    px = t.history(period="4y", auto_adjust=False)
    if len(px) == 0: continue
    px.index = px.index.tz_localize(None).normalize()
    ret = px["Close"].pct_change()
    days = px.index
    for when, r in ed.iterrows():
        est, act = r["EPS Estimate"], r["Reported EPS"]
        if est is None or act is None or est == 0: continue
        surp = (act - est) / abs(est)
        ann_date = when.normalize()
        after = days[days > ann_date]
        same  = days[days == ann_date]
        # BMO (before ~9:30 ET) -> reaction is ann_date itself; AMC -> next session
        if when.hour < 9 or (when.hour == 9 and when.minute < 30):
            cand = same if len(same) else after
        else:
            cand = after
        if len(cand) == 0: continue
        ridx = days.get_loc(cand[0])
        if ridx < 6 or ridx + 6 >= len(days): continue
        def ar(i):  # abnormal return on trading-day index i (class formula)
            s, m = ret.iloc[i], spy_ret.get(days[i], np.nan)
            return np.nan if pd.isna(s) or pd.isna(m) else s - m
        win = {d: ar(ridx + d) for d in range(-5, 6)}
        if any(pd.isna(v) for v in win.values()): continue
        rows.append(dict(ticker=tk, report_date=str(ann_date.date()),
                         q_period=f"{'Q3' if ann_date.month in (10,11) else 'other'}",
                         fall_print=(ann_date.month in (10, 11)),
                         eps_est=est, eps_act=act, surprise_pct=round(100 * surp, 1),
                         beat=surp > 0.01, miss=surp < -0.01,
                         pre_car_m5_m1=round(100 * sum(win[d] for d in range(-5, 0)), 2),
                         react_ar=round(100 * win[0], 2),
                         car_p1_p3=round(100 * sum(win[d] for d in range(1, 4)), 2),
                         car_p1_p5=round(100 * sum(win[d] for d in range(1, 6)), 2),
                         car_m1_p1=round(100 * sum(win[d] for d in range(-1, 2)), 2),
                         car_m5_p5=round(100 * sum(win[d] for d in range(-5, 6)), 2)))

ev = pd.DataFrame(rows).sort_values(["ticker", "report_date"])
ev.to_csv("earnings_events.csv", index=False)
print(f"events built: {len(ev)} across {ev.ticker.nunique()} tickers\n")

def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3: return (np.nan, np.nan, len(x))
    t, p = stats.ttest_1samp(x, 0)
    return (round(float(t), 2), round(float(p), 3), len(x))

print("=== PER-TICKER (all quarters in sample) ===")
print(f"{'tk':6}{'n':>3}{'beat%':>6}{'avgSurp%':>9}{'reactAR%':>9}{'t':>6}{'p':>6}"
      f"{'|react|%':>9}{'CAR+1..3':>9}{'CAR+1..5':>9}{'preCAR%':>8}")
per = []
for tk, g in ev.groupby("ticker"):
    t, p, n = tstat(g.react_ar)
    per.append(dict(ticker=tk, n=n, beat_rate=round(100 * g.beat.mean()),
                    avg_surprise=round(g.surprise_pct.mean(), 1), mean_react_ar=round(g.react_ar.mean(), 2),
                    react_t=t, react_p=p, mean_abs_react=round(g.react_ar.abs().mean(), 2),
                    drift_1_3=round(g.car_p1_p3.mean(), 2), drift_1_5=round(g.car_p1_p5.mean(), 2),
                    pre_car=round(g.pre_car_m5_m1.mean(), 2)))
    print(f"{tk:6}{n:>3}{100*g.beat.mean():>5.0f}%{g.surprise_pct.mean():>9.1f}"
          f"{g.react_ar.mean():>9.2f}{t:>6}{p:>6}{g.react_ar.abs().mean():>9.2f}"
          f"{g.car_p1_p3.mean():>9.2f}{g.car_p1_p5.mean():>9.2f}{g.pre_car_m5_m1.mean():>8.2f}")
pd.DataFrame(per).to_csv("per_ticker_summary.csv", index=False)

print("\n=== BEAT vs MISS (pooled, the class's div_chg split) ===")
for label, g in [("BEAT (>+1%)", ev[ev.beat]), ("INLINE (±1%)", ev[~ev.beat & ~ev.miss]), ("MISS (<-1%)", ev[ev.miss])]:
    if len(g) < 3: continue
    t, p, n = tstat(g.react_ar); t3, p3, _ = tstat(g.car_p1_p5)
    hit = 100 * ((g.react_ar > 0) == g.beat).mean() if label != "INLINE (±1%)" else np.nan
    print(f"{label:14} n={n:3}  reactAR {g.react_ar.mean():+6.2f}% (t={t}, p={p})  "
          f"|react| {g.react_ar.abs().mean():.2f}%  drift+1..5 {g.car_p1_p5.mean():+6.2f}% (t={t3}, p={p3})"
          + (f"  direction-hit {hit:.0f}%" if not np.isnan(hit) else ""))

print("\n=== FALL (Q3-season) PRINTS ONLY — 'this period of the year' ===")
f = ev[ev.fall_print]
for tk, g in f.groupby("ticker"):
    if len(g) < 2: continue
    print(f"{tk:6} n={len(g)}  reactARs: {list(g.react_ar)}  mean {g.react_ar.mean():+.2f}%  "
          f"drift+1..5 mean {g.car_p1_p5.mean():+.2f}%  surprises: {list(g.surprise_pct)}")

print("\n=== AAR / CAAR by event day, BEATS only (class Homework 9 Q6-7) ===")
# rebuild day-by-day AAR for beats from stored windows is not possible post-hoc;
# use react + drift blocks as the CAAR spine:
b = ev[ev.beat]
print(f"beats n={len(b)}  pre[-5,-1] {b.pre_car_m5_m1.mean():+.2f}%  day0 {b.react_ar.mean():+.2f}%  "
      f"[+1,+3] {b.car_p1_p3.mean():+.2f}%  [+1,+5] {b.car_p1_p5.mean():+.2f}%  "
      f"CAAR[-5,+5] {b.car_m5_p5.mean():+.2f}%")

print("\n=== LAST 4 PRINTS PER KEY NAME (recency check) ===")
for tk in ["GOOGL", "MSFT", "AMZN", "META", "AAPL", "PLTR", "XOM", "EQT", "SWK"]:
    g = ev[ev.ticker == tk].tail(4)
    print(tk, " | ".join(f"{r.report_date}: surp {r.surprise_pct:+.1f}% -> AR {r.react_ar:+.2f}%, drift5 {r.car_p1_p5:+.2f}%"
                          for r in g.itertuples()))

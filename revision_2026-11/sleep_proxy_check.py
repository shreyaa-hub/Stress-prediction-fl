"""Sleep-proxy validation (run on the raw StudentLife folder).
For every Sleep-EMA answer ("How many hours did you sleep last night?"), computes the
longest dark-environment, locked, and dark-or-locked interval between 20:00 the previous
day and 12:00 on the response day (America/New_York), and the total dark time in that window.
Usage: python sleep_proxy_check.py DATASET_DIR OUT_CSV ; then analyse with r10d_sleep_proxy_testsize_mlp.py
"""
import os, sys, json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
B, OUT = sys.argv[1], sys.argv[2]; TZ = ZoneInfo('America/New_York')
def li(p):
    if not os.path.exists(p): return np.empty((0, 2), np.int64)
    d = pd.read_csv(p); d.columns = [c.strip().lower() for c in d.columns]
    s = [c for c in d.columns if 'start' in c][0]; e = [c for c in d.columns if 'end' in c][0]
    a = d[[s, e]].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(np.int64); a = a[a[:, 1] > a[:, 0]]
    return merge(a)
def merge(a):
    if len(a) == 0: return a
    a = a[np.argsort(a[:, 0])]; o = [list(a[0])]
    for s, e in a[1:]:
        if s <= o[-1][1]: o[-1][1] = max(o[-1][1], e)
        else: o.append([s, e])
    return np.array(o, np.int64)
def longest(iv, a, b):
    if len(iv) == 0: return 0.0
    m = (iv[:, 1] > a) & (iv[:, 0] < b)
    return (np.clip(np.minimum(iv[m, 1], b) - np.maximum(iv[m, 0], a), 0, None).max() / 3600) if m.any() else 0.0
rows = []
for fn in sorted(os.listdir(f'{B}/EMA/response/Sleep')):
    u = fn[6:-5]; dark = li(f'{B}/sensing/dark/dark_{u}.csv'); lock = li(f'{B}/sensing/phonelock/phonelock_{u}.csv')
    both = merge(np.vstack([v for v in [dark, lock] if len(v)])) if (len(dark) or len(lock)) else np.empty((0, 2), np.int64)
    for r in json.load(open(f'{B}/EMA/response/Sleep/{fn}')):
        try: k = int(r.get('hour')); t = int(r['resp_time'])
        except Exception: continue
        if not 1 <= k <= 19: continue
        hrs = 2.5 if k == 1 else 3 + 0.5 * (k - 1)      # [1]<3, [2]3.5 ... [19]12 hours
        lt = datetime.fromtimestamp(t, TZ)
        n0 = datetime(lt.year, lt.month, lt.day, tzinfo=TZ) - timedelta(hours=4); n1 = n0 + timedelta(hours=16)
        n0, n1 = int(n0.timestamp()), int(n1.timestamp())
        rows.append(dict(uid=u, t=t, hour=lt.hour, ema=hrs, night_dark=longest(dark, n0, n1), night_lock=longest(lock, n0, n1),
                         night_union=longest(both, n0, n1),
                         night_dark_total=(np.clip(np.minimum(dark[:, 1], n1) - np.maximum(dark[:, 0], n0), 0, None).sum() / 3600 if len(dark) else 0)))
d = pd.DataFrame(rows).drop_duplicates(['uid', 't']); d = d[d.hour >= 6]
d.to_csv(OUT, index=False); print(len(d), 'responses written to', OUT)

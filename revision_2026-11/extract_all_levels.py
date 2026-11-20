"""Re-extract the eight features for ALL parsed Stress-EMA responses (levels 1-5), with the
same 24-h causal window, interval merging and inference filtering as notebook cells 6-7.
Usage: python extract_all_levels.py DATASET_DIR OUT_CSV
The Scheme-C rows reproduce results/features.csv exactly (checked: max abs difference 0)."""
import os, sys, json, numpy as np, pandas as pd
BASE, OUT = sys.argv[1], sys.argv[2]; WINDOW = 24 * 3600
def load_intervals(path):
    if not os.path.exists(path): return np.empty((0, 2), dtype=np.int64)
    df = pd.read_csv(path); df.columns = [c.strip().lower() for c in df.columns]
    s = [c for c in df.columns if 'start' in c][0]; e = [c for c in df.columns if 'end' in c][0]
    a = df[[s, e]].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(np.int64)
    return a[a[:, 1] > a[:, 0]]
def merge(iv):
    if len(iv) == 0: return iv
    iv = iv[np.argsort(iv[:, 0])]; out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return np.array(out, dtype=np.int64)
def load_points(path):
    if not os.path.exists(path): return np.empty(0, np.int64), np.empty(0, np.int64)
    df = pd.read_csv(path); df.columns = [c.strip().lower() for c in df.columns]
    t = [c for c in df.columns if 'time' in c][0]; v = [c for c in df.columns if 'inference' in c][0]
    df = df[[t, v]].apply(pd.to_numeric, errors='coerce').dropna()
    ts, vs = df[t].to_numpy(np.int64), df[v].to_numpy(np.int64); o = np.argsort(ts); return ts[o], vs[o]
def ov(iv, w0, w1):
    if len(iv) == 0: return np.zeros(0)
    return np.clip(np.minimum(iv[:, 1], w1) - np.maximum(iv[:, 0], w0), 0, None)
rows = []
for fn in sorted(os.listdir(f'{BASE}/EMA/response/Stress')):
    if not fn.endswith('.json'): continue
    uid = fn.replace('Stress_', '').replace('.json', ''); resp = []
    for r in json.load(open(f'{BASE}/EMA/response/Stress/{fn}')):
        lv = r.get('level')
        if lv is None: continue
        try: resp.append((int(r['resp_time']), int(lv)))
        except Exception: pass
    dark = merge(load_intervals(f'{BASE}/sensing/dark/dark_{uid}.csv'))
    lock = merge(load_intervals(f'{BASE}/sensing/phonelock/phonelock_{uid}.csv'))
    conv = merge(load_intervals(f'{BASE}/sensing/conversation/conversation_{uid}.csv'))
    a_ts, a_v = load_points(f'{BASE}/sensing/activity/activity_{uid}.csv')
    u_ts, u_v = load_points(f'{BASE}/sensing/audio/audio_{uid}.csv')
    for w1, lv in resp:
        w0 = w1 - WINDOW; d, l, c = ov(dark, w0, w1), ov(lock, w0, w1), ov(conv, w0, w1)
        i0, i1 = np.searchsorted(a_ts, [w0, w1]); act = a_v[i0:i1]; act = act[act != 3]
        j0, j1 = np.searchsorted(u_ts, [w0, w1]); aud = u_v[j0:j1]; aud = aud[aud != 3]
        rows.append({'uid': uid, 'resp_time': w1, 'level': lv,
            'sleep_hours': d.max() / 3600 if len(d) else 0.0, 'dark_hours': d.sum() / 3600, 'idle_hours': l.sum() / 3600,
            'idle_episodes': int(((lock[:, 1] >= w0) & (lock[:, 1] < w1)).sum()) if len(lock) else 0,
            'conv_count': int((c > 0).sum()), 'conv_hours': c.sum() / 3600,
            'activity_n': len(act), 'activity_level': float(np.isin(act, [1, 2]).mean()) if len(act) else np.nan,
            'audio_n': len(aud), 'voice_ratio': float((aud == 1).mean()) if len(aud) else np.nan})
pd.DataFrame(rows).to_csv(OUT, index=False); print(len(rows), 'rows ->', OUT)

import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import os
R = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results'))
F = ['sleep_hours', 'dark_hours', 'idle_hours', 'idle_episodes',
     'conv_count', 'conv_hours', 'activity_level', 'voice_ratio']
ROUNDS, LOCAL_EPOCHS, LR, L2, SEEDS = 30, 5, 0.5, 0.01, 5
REGIMES = ['Centralized', 'FedAvg', 'Local', 'PersonalizedFL']


def load(path=f'{R}/features.csv'):
    feat = pd.read_csv(path)
    df = feat[feat.activity_n >= 2000].dropna(subset=['activity_level', 'voice_ratio']).copy()
    return feat, df


def pick_clients(df):
    return [u for u, s in df.groupby('uid')
            if len(s) >= 15 and s.y.sum() >= 4 and (1 - s.y).sum() >= 4]


def sig(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train(X, y, w, b, epochs):
    n = len(y)
    for _ in range(epochs):
        p = sig(X @ w + b)
        w = w - LR * (X.T @ (p - y) / n + L2 * w)
        b = b - LR * (p - y).mean()
    return w, b


def fedavg(parts, rounds=ROUNDS, ep=LOCAL_EPOCHS, d=None, init=None, curve_fn=None):
    d = d or next(iter(parts.values()))[0].shape[1]
    gw, gb = (np.zeros(d), 0.0) if init is None else init
    for r in range(rounds):
        ws, bs, ns = [], [], []
        for Xt, yt, _, _ in parts.values():
            w, b = train(Xt, yt, gw.copy(), gb, ep); ws.append(w); bs.append(b); ns.append(len(yt))
        wt = np.array(ns) / sum(ns)
        gw = (np.array(ws) * wt[:, None]).sum(0); gb = float((np.array(bs) * wt).sum())
        if curve_fn: curve_fn(r, gw, gb, ws, bs)
    return gw, gb


def run_regimes(parts, pfl_epochs=LOCAL_EPOCHS * 5):
    d = next(iter(parts.values()))[0].shape[1]; res = {k: {} for k in REGIMES}
    for uid, (Xt, yt, Xe, ye) in parts.items():
        w, b = train(Xt, yt, np.zeros(d), 0.0, ROUNDS * LOCAL_EPOCHS)
        if len(np.unique(ye)) > 1: res['Local'][uid] = roc_auc_score(ye, sig(Xe @ w + b))
    gw, gb = fedavg(parts)
    for uid, (Xt, yt, Xe, ye) in parts.items():
        if len(np.unique(ye)) > 1:
            res['FedAvg'][uid] = roc_auc_score(ye, sig(Xe @ gw + gb))
            w, b = train(Xt, yt, gw.copy(), gb, pfl_epochs)
            res['PersonalizedFL'][uid] = roc_auc_score(ye, sig(Xe @ w + b))
    Xa = np.vstack([p[0] for p in parts.values()]); ya = np.concatenate([p[1] for p in parts.values()])
    cw, cb = train(Xa, ya, np.zeros(d), 0.0, ROUNDS * LOCAL_EPOCHS)
    for uid, (_, _, Xe, ye) in parts.items():
        if len(np.unique(ye)) > 1: res['Centralized'][uid] = roc_auc_score(ye, sig(Xe @ cw + cb))
    return res, (gw, gb), (cw, cb)


def split_parts(df, clients, split_fn, feats=F):
    parts, times = {}, {}
    for uid in clients:
        s = df[df.uid == uid]; Xr, yr, tt = s[feats].values, s.y.values, s.resp_time.values
        tr, te = split_fn(Xr, yr, tt)
        mu, sd = Xr[tr].mean(0), Xr[tr].std(0); sd[sd == 0] = 1.0
        parts[uid] = ((Xr[tr] - mu) / sd, yr[tr], (Xr[te] - mu) / sd, yr[te]); times[uid] = (tt[tr], tt[te])
    return parts, times


def random_split(seed):
    return lambda X, y, t: next(iter(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y)))


def chrono(gap_hours=0):
    def f(X, y, t):
        order = np.argsort(t); cut = int(np.ceil(len(y) * 2 / 3))
        tr, te = order[:cut], order[cut:]
        if gap_hours: tr = tr[t[tr] < t[te].min() - gap_hours * 3600]
        return tr, te
    return f


def ci95(x):
    x = np.asarray(x); m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); t = stats.t.ppf(.975, len(x) - 1)
    return m, m - t * se, m + t * se


def ci90(x):
    x = np.asarray(x); m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); t = stats.t.ppf(.95, len(x) - 1)
    return m, m - t * se, m + t * se

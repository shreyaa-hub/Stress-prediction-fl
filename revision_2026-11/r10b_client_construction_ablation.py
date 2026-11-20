import sys; sys.path.insert(0, '.')
from common import *

feat, df = load(); clients = pick_clients(df)

# ---------- (1) client construction confound: 4 arms, shared global standardisation ----------
dd = df[df.uid.isin(clients)].reset_index(drop=True); K = len(clients)
Xg = ((dd[F] - dd[F].mean()) / dd[F].std()).values; yg = dd.y.values
sizes = np.array([(dd.uid == u).sum() for u in clients]); npos = np.array([dd.y[dd.uid == u].sum() for u in clients])

def run(shards, seed):
    parts = {}
    for k, idx in enumerate(shards):
        Xs, ys = Xg[idx], yg[idx]
        if len(np.unique(ys)) < 2: continue
        tr, te = next(iter(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xs, ys)))
        parts[k] = (Xs[tr], ys[tr], Xs[te], ys[te])
    d = len(F)
    loc = [roc_auc_score(ye, sig(Xe @ w + b)) for k, (Xt, yt, Xe, ye) in parts.items()
           for w, b in [train(Xt, yt, np.zeros(d), 0.0, ROUNDS * LOCAL_EPOCHS)] if len(np.unique(ye)) > 1]
    gw, gb = fedavg(parts)
    fed = [roc_auc_score(ye, sig(Xe @ gw + gb)) for _, _, Xe, ye in parts.values() if len(np.unique(ye)) > 1]
    het = np.std([yt.mean() for _, yt, _, _ in parts.values()])
    return np.mean(loc), np.mean(fed), het

def shards_for(arm, seed):
    rng = np.random.RandomState(seed)
    if arm == 'real': return [dd.index[dd.uid == u].values for u in clients]
    if arm == 'random equal': return np.array_split(rng.permutation(len(dd)), K)
    if arm == 'random, real sizes':
        perm = rng.permutation(len(dd)); return np.split(perm, np.cumsum(sizes)[:-1])
    if arm == 'mixed, real sizes+labels':
        pos = rng.permutation(np.where(yg == 1)[0]); neg = rng.permutation(np.where(yg == 0)[0])
        ps = np.split(pos, np.cumsum(npos)[:-1]); ns = np.split(neg, np.cumsum(sizes - npos)[:-1])
        return [np.r_[a, b] for a, b in zip(ps, ns)]

print('Client construction (5 seeds): Local, FedAvg, Delta, Delta range, label sigma')
for arm in ['real', 'random equal', 'random, real sizes', 'mixed, real sizes+labels']:
    L, Fd, H = [], [], []
    for s in range(SEEDS):
        l, f, h = run(shards_for(arm, s), s); L.append(l); Fd.append(f); H.append(h)
    L, Fd, H = map(np.array, (L, Fd, H)); D = Fd - L
    print(f'  {arm:<26} L {L.mean():.3f}  F {Fd.mean():.3f}  D {D.mean():+.3f} [{D.min():+.3f},{D.max():+.3f}]  sigma {H.mean():.3f}')

# ---------- (2) feature ablation (random hold-out, per-client standardisation) ----------
print('\nDrop-one-feature ablation, random hold-out (mean AUC over 100 client-seed values)')
def regimes_feats(feats):
    out = {k: [] for k in REGIMES}
    for s in range(SEEDS):
        parts, _ = split_parts(df, clients, random_split(s), feats)
        r = run_regimes(parts)[0]
        for k in REGIMES: out[k] += [r[k][u] for u in clients]
    return {k: np.array(v) for k, v in out.items()}
full = regimes_feats(F)
print(f'  {"all features":<16} Local {full["Local"].mean():.3f}  FedAvg {full["FedAvg"].mean():.3f}  Central {full["Centralized"].mean():.3f}')
abl = {}
for f in F:
    r = regimes_feats([x for x in F if x != f]); abl[f] = r
    print(f'  -{f:<15} Local {r["Local"].mean() - full["Local"].mean():+.3f}  FedAvg {r["FedAvg"].mean() - full["FedAvg"].mean():+.3f}  Central {r["Centralized"].mean() - full["Centralized"].mean():+.3f}')
# univariate within-client AUC (seed-averaged local, single feature)
print('\nSingle-feature local models (random hold-out): mean AUC; and |AUC-0.5| over clients')
for f in F:
    r = regimes_feats([f])
    print(f'  {f:<15} Local {r["Local"].mean():.3f}  FedAvg {r["FedAvg"].mean():.3f}')

# ---------- (3) Holm on per-feature null p-values ----------
p = np.array([0.64, 0.16, 0.005, 0.10, 0.12, 0.20, 0.41, 0.45]); o = np.argsort(p)
print('\nHolm thresholds:', [(F[i], p[i], round(0.05 / (8 - j), 5), p[i] <= 0.05 / (8 - j)) for j, i in enumerate(o[:3])])

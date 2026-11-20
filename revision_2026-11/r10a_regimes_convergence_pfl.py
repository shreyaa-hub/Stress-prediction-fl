import sys; sys.path.insert(0, '.')
from common import *

feat, df = load(); clients = pick_clients(df)
print('instances', len(df), 'clients', len(clients), df.uid.isin(clients).sum())

# ---------- baseline reproduction (random hold-out) ----------
base, P = [], []
for seed in range(SEEDS):
    parts, _ = split_parts(df, clients, random_split(seed)); P.append(parts)
    base.append(run_regimes(parts)[0])
for k in REGIMES:
    v = np.array([r[k][u] for r in base for u in clients]); print(f'{k:<15}{v.mean():.3f} ± {v.std():.3f}')
pc = {k: np.array([np.mean([r[k][u] for r in base]) for u in clients]) for k in REGIMES}

# ---------- test-set sizes ----------
nte = np.array([[len(P[s][u][3]) for u in clients] for s in range(SEEDS)])
npos = np.array([[P[s][u][3].sum() for u in clients] for s in range(SEEDS)])
nneg = nte - npos
print('\ntest size per client: min', nte.min(), 'max', nte.max(), 'median', np.median(nte))
print('test positives min', npos.min(), 'negatives min', nneg.min())
pairs = npos * nneg
# pair-weighted mean (each client-seed weighted by number of positive-negative pairs that define its AUC)
print('\nPair-weighted mean AUC (random hold-out):')
for k in REGIMES:
    a = np.array([[base[s][k][u] for u in clients] for s in range(SEEDS)])
    print(f'  {k:<15}{(a * pairs).sum() / pairs.sum():.3f}   size-weighted {(a * nte).sum() / nte.sum():.3f}')
# restrict to clients whose (seed-avg) test set has >= 20 records
big = [i for i, u in enumerate(clients) if nte[:, i].mean() >= 20]
print(f'\nclients with >=20 test records: {len(big)}')
for k in REGIMES: print(f'  {k:<15}{pc[k][big].mean():.3f}')
d = pc['FedAvg'][big] - pc['Local'][big]
print('  FedAvg-Local CI', np.round(ci95(d), 3), 'Wilcoxon p', round(stats.wilcoxon(pc['FedAvg'][big], pc['Local'][big]).pvalue, 3))
print('  Friedman p', round(stats.friedmanchisquare(*[pc[k][big] for k in REGIMES]).pvalue, 3))
# weighted regression of difference: weighted mean diff with weights = mean pairs
wts = pairs.mean(0); dd = pc['FedAvg'] - pc['Local']
wm = (wts * dd).sum() / wts.sum()
print(f'  pair-weighted mean FedAvg-Local over 20 clients {wm:+.3f}')
# correlation of |FedAvg-Local| with test size
print('  corr(|diff|, test size)', round(stats.spearmanr(np.abs(dd), nte.mean(0)).correlation, 3))

# ---------- time-ordered: which 16 are scored; same-16 random hold-out ----------
for gap in [0, 24]:
    parts, _ = split_parts(df, clients, chrono(gap))
    parts = {u: v for u, v in parts.items() if len(np.unique(v[1])) > 1}
    r = run_regimes(parts)[0]; scored = sorted(r['Local'])
    print(f'\ntime-ordered gap={gap}: trained {len(parts)}, scored {len(scored)}')
    for k in REGIMES: print(f'  {k:<15}{np.mean([r[k][u] for u in scored]):.3f}')
    tn = [len(parts[u][3]) for u in scored]; print('  test sizes', min(tn), max(tn))
    if gap == 0: S16 = scored; R16 = r
idx16 = [clients.index(u) for u in S16]
print('\nunscored clients:', sorted(set(clients) - set(S16)))
print('Random hold-out restricted to the same 16 clients (seed-averaged):')
for k in REGIMES: print(f'  {k:<15}{pc[k][idx16].mean():.3f}')
d16 = pc['FedAvg'][idx16] - pc['Local'][idx16]
print('  FedAvg-Local', np.round(ci95(d16), 3), 'Wilcoxon p', round(stats.wilcoxon(pc['FedAvg'][idx16], pc['Local'][idx16]).pvalue, 3))
print('  Friedman p', round(stats.friedmanchisquare(*[pc[k][idx16] for k in REGIMES]).pvalue, 3))
# 100 client-seed version for same 16
v16 = {k: np.array([base[s][k][u] for s in range(SEEDS) for u in S16]) for k in REGIMES}
for k in REGIMES: print(f'  (client-seed) {k:<15}{v16[k].mean():.3f} ± {v16[k].std():.3f}')

# ---------- TOST / equivalence & minimum detectable difference ----------
print('\nEquivalence (90% CI, TOST margin 0.05) and minimum detectable effect (80% power, alpha .05)')
for name, x in [('random 20', pc['FedAvg'] - pc['Local']), ('random 16', d16),
                ('time-ord 16', np.array([R16['FedAvg'][u] - R16['Local'][u] for u in S16]))]:
    m, lo, hi = ci90(x); n = len(x)
    mde = (stats.t.ppf(.975, n - 1) + stats.t.ppf(.8, n - 1)) * x.std(ddof=1) / np.sqrt(n)
    print(f'  {name:<12} mean {m:+.3f} 90% CI [{lo:+.3f},{hi:+.3f}] equiv(±0.05)={lo > -0.05 and hi < 0.05}  MDE {mde:.3f}  sd {x.std(ddof=1):.3f}')

# ---------- PFL controls: personalization vs extra optimisation ----------
print('\nPFL controls (random hold-out, 100 client-seed values; mean AUC)')
ctl = {k: [] for k in ['FedAvg', 'PFL5', 'PFL25', 'PFL50', 'PFL150', 'FedAvg+25 pooled', 'FedAvg 35 rounds', 'Local150', 'Local175']}
for s in range(SEEDS):
    parts = P[s]; d = len(F)
    gw, gb = fedavg(parts)
    Xa = np.vstack([p[0] for p in parts.values()]); ya = np.concatenate([p[1] for p in parts.values()])
    pw, pb = train(Xa, ya, gw.copy(), gb, 25)  # 25 extra non-personal epochs on pooled data
    fw, fb = fedavg(parts, rounds=5, init=(gw.copy(), gb))  # 5 extra rounds x 5 epochs = 25 extra federated epochs
    for u in clients:
        Xt, yt, Xe, ye = parts[u]
        ctl['FedAvg'].append(roc_auc_score(ye, Xe @ gw + gb))
        for e in [5, 25, 50, 150]:
            w, b = train(Xt, yt, gw.copy(), gb, e); ctl[f'PFL{e}'].append(roc_auc_score(ye, Xe @ w + b))
        ctl['FedAvg+25 pooled'].append(roc_auc_score(ye, Xe @ pw + pb))
        ctl['FedAvg 35 rounds'].append(roc_auc_score(ye, Xe @ fw + fb))
        for e in [150, 175]:
            w, b = train(Xt, yt, np.zeros(d), 0.0, e); ctl[f'Local{e}'].append(roc_auc_score(ye, Xe @ w + b))
for k, v in ctl.items():
    v = np.array(v); print(f'  {k:<18}{v.mean():.3f} ± {v.std():.3f}')
for k in ['PFL25', 'FedAvg+25 pooled', 'FedAvg 35 rounds', 'Local175']:
    print(f'  Wilcoxon {k} vs FedAvg p={stats.wilcoxon(ctl[k], ctl["FedAvg"]).pvalue:.3f};  vs Local150 p={stats.wilcoxon(ctl[k], ctl["Local150"]).pvalue:.3f}')

# ---------- convergence: FedAvg curve vs centralized curve vs train AUC, drift ----------
print('\nConvergence diagnostics')
cur = np.zeros((SEEDS, ROUNDS)); tra = np.zeros((SEEDS, ROUNDS)); cen = np.zeros((SEEDS, ROUNDS)); drift = np.zeros((SEEDS, ROUNDS))
loss = np.zeros((SEEDS, ROUNDS))
for s in range(SEEDS):
    parts = P[s]; d = len(F)
    def cb(r, gw, gb, ws, bs):
        cur[s, r] = np.mean([roc_auc_score(ye, Xe @ gw + gb) for _, _, Xe, ye in parts.values() if len(np.unique(ye)) > 1])
        tra[s, r] = np.mean([roc_auc_score(yt, Xt @ gw + gb) for Xt, yt, _, _ in parts.values()])
        g = np.r_[gw, gb]
        drift[s, r] = np.mean([np.linalg.norm(np.r_[w, b] - g) / np.linalg.norm(g) for w, b in zip(ws, bs)])
        Xa = np.vstack([p[0] for p in parts.values()]); ya = np.concatenate([p[1] for p in parts.values()])
        pp = sig(Xa @ gw + gb); loss[s, r] = -np.mean(ya * np.log(pp + 1e-12) + (1 - ya) * np.log(1 - pp + 1e-12))
    fedavg(parts, curve_fn=cb)
    Xa = np.vstack([p[0] for p in parts.values()]); ya = np.concatenate([p[1] for p in parts.values()])
    w, b = np.zeros(d), 0.0
    for r in range(ROUNDS):
        w, b = train(Xa, ya, w, b, LOCAL_EPOCHS)
        cen[s, r] = np.mean([roc_auc_score(ye, Xe @ w + b) for _, _, Xe, ye in parts.values() if len(np.unique(ye)) > 1])
m = cur.mean(0)
print('  FedAvg test AUC rounds 1,6,10,20,30:', np.round(m[[0, 5, 9, 19, 29]], 3), 'peak round', m.argmax() + 1)
print('  per-seed peak rounds:', cur.argmax(1) + 1, ' per-seed decline peak->30:', np.round(cur.max(1) - cur[:, -1], 3))
print('  per-seed sd at r6, r30:', np.round(cur.std(0)[[5, 29]], 3))
print('  Centralized (matched epochs 5..150) AUC at 5,30,50,100,150 epochs:', np.round(cen.mean(0)[[0, 5, 9, 19, 29]], 3), 'peak at epoch', (cen.mean(0).argmax() + 1) * 5)
print('  FedAvg TRAIN AUC rounds 1,6,30:', np.round(tra.mean(0)[[0, 5, 29]], 3))
print('  pooled training loss rounds 1,6,30:', np.round(loss.mean(0)[[0, 5, 29]], 4))
print('  client drift (mean rel. dist local vs global) rounds 1,6,30:', np.round(drift.mean(0)[[0, 5, 29]], 3))
print('  corr(FedAvg curve, centralized curve) =', round(np.corrcoef(m, cen.mean(0))[0, 1], 3), ' max |diff| =', round(np.abs(m - cen.mean(0)).max(), 3))
np.save(f'{R}/central_curve.npy', cen)
print('  pooled training loss decreases monotonically:', bool(np.all(np.diff(loss.mean(0)) <= 0)))
d6_30 = cur[:, 5] - cur[:, 29]
print('  r6 - r30 per seed', np.round(d6_30, 3), ' paired t p', round(stats.ttest_1samp(d6_30, 0).pvalue, 3))

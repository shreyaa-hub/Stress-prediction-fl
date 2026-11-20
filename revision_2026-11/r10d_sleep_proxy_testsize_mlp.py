import sys; sys.path.insert(0, '.')
from common import *

feat, df = load(); clients = pick_clients(df)

# ---------- sleep proxy validation ----------
S = pd.read_csv(f'{R}/revision_2026-11/sleep_ema_vs_dark.csv').drop_duplicates(['uid', 'resp_time'])
S = S[S.longest_dark_hours > 0]
print('sleep EMA responses with dark data:', len(S), 'users', S.uid.nunique())
r, p = stats.spearmanr(S.sleep_ema_hours, S.longest_dark_hours)
print(f'pooled Spearman rho {r:.3f} p={p:.2g}; Pearson {stats.pearsonr(S.sleep_ema_hours, S.longest_dark_hours)[0]:.3f}')
print('mean EMA hours', round(S.sleep_ema_hours.mean(), 2), 'mean longest dark', round(S.longest_dark_hours.mean(), 2),
      'median abs diff', round((S.sleep_ema_hours - S.longest_dark_hours).abs().median(), 2))
# between-person: per-user means
um = S.groupby('uid')[['sleep_ema_hours', 'longest_dark_hours']].mean(); um = um[S.groupby('uid').size() >= 5]
print('between-person (user means, >=5 resp, n=%d) Spearman %.3f p=%.2g' % ((len(um),) + tuple(stats.spearmanr(um.iloc[:, 0], um.iloc[:, 1]))))
# within-person: centered
c = S.groupby('uid').filter(lambda g: len(g) >= 5).copy()
c['e'] = c.sleep_ema_hours - c.groupby('uid').sleep_ema_hours.transform('mean')
c['d'] = c.longest_dark_hours - c.groupby('uid').longest_dark_hours.transform('mean')
print('within-person (centered) Pearson %.3f p=%.2g n=%d' % (stats.pearsonr(c.e, c.d) + (len(c),)))
print('fraction where |EMA - dark| <= 2h:', round(((S.sleep_ema_hours - S.longest_dark_hours).abs() <= 2).mean(), 3))

# ---------- test-size: clients with >= median test size ----------
base, P = [], []
for s in range(SEEDS):
    parts, _ = split_parts(df, clients, random_split(s)); P.append(parts); base.append(run_regimes(parts)[0])
pc = {k: np.array([np.mean([r[k][u] for r in base]) for u in clients]) for k in REGIMES}
nte = np.array([np.mean([len(P[s][u][3]) for s in range(SEEDS)]) for u in clients])
big = nte >= np.median(nte)
print('\nclients with test size >= median (%.1f): %d; sizes %d-%d' % (np.median(nte), big.sum(), nte[big].min(), nte[big].max()))
for k in REGIMES: print(f'  {k:<15}{pc[k][big].mean():.3f}   small half {pc[k][~big].mean():.3f}')
d = pc['FedAvg'][big] - pc['Local'][big]
print('  big half FedAvg-Local', np.round(ci95(d), 3), 'Friedman p', round(stats.friedmanchisquare(*[pc[k][big] for k in REGIMES]).pvalue, 3))
# client-level s.d. of AUC across seeds vs test size (sampling noise)
sd_seed = np.array([np.std([base[s]['Local'][u] for s in range(SEEDS)]) for u in clients])
print('  mean across-seed s.d. of a client AUC (Local): small half %.3f, big half %.3f' % (sd_seed[~big].mean(), sd_seed[big].mean()))
# pair-weighted Wilcoxon-free estimate: weighted mean diff with cluster bootstrap CI
pairs = np.array([np.mean([P[s][u][3].sum() * (len(P[s][u][3]) - P[s][u][3].sum()) for s in range(SEEDS)]) for u in clients])
rng = np.random.default_rng(0); dd = pc['FedAvg'] - pc['Local']; bs = []
for _ in range(10000):
    i = rng.integers(0, 20, 20); bs.append((pairs[i] * dd[i]).sum() / pairs[i].sum())
print('  pair-weighted FedAvg-Local %.3f, cluster-bootstrap 95%% CI [%.3f, %.3f]' % ((pairs * dd).sum() / pairs.sum(), *np.percentile(bs, [2.5, 97.5])))
for k in REGIMES: print(f'   pair-weighted {k:<15}{(pairs * pc[k]).sum() / pairs.sum():.3f}')

# ---------- non-linear federated model: one-hidden-layer MLP ----------
H = 8
def mlp_init(d, seed):
    r = np.random.RandomState(1000 + seed); return [r.normal(0, 0.3, (d, H)), np.zeros(H), r.normal(0, 0.3, H), 0.0]
def mlp_fwd(p, X):
    h = np.tanh(X @ p[0] + p[1]); return h, h @ p[2] + p[3]
def mlp_train(X, y, p, ep, lr=0.5):
    W1, b1, w2, b2 = [q.copy() if isinstance(q, np.ndarray) else q for q in p]; n = len(y)
    for _ in range(ep):
        h = np.tanh(X @ W1 + b1); z = h @ w2 + b2; g = (sig(z) - y) / n
        gw2 = h.T @ g + L2 * w2; gb2 = g.sum()
        gh = np.outer(g, w2) * (1 - h ** 2); gW1 = X.T @ gh + L2 * W1; gb1 = gh.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; w2 -= lr * gw2; b2 -= lr * gb2
    return [W1, b1, w2, b2]
def avg(ps, wt): return [sum(w * p[i] for w, p in zip(wt, ps)) for i in range(4)]
res = {k: [] for k in REGIMES}
for s in range(SEEDS):
    parts = P[s]; d = len(F); p0 = mlp_init(d, s)
    for u, (Xt, yt, Xe, ye) in parts.items():
        res['Local'].append(roc_auc_score(ye, mlp_fwd(mlp_train(Xt, yt, p0, 150), Xe)[1]))
    g = p0
    for _ in range(ROUNDS):
        loc = [mlp_train(Xt, yt, g, LOCAL_EPOCHS) for Xt, yt, _, _ in parts.values()]
        ns = np.array([len(v[1]) for v in parts.values()]); g = avg(loc, ns / ns.sum())
    for u, (Xt, yt, Xe, ye) in parts.items():
        res['FedAvg'].append(roc_auc_score(ye, mlp_fwd(g, Xe)[1]))
        res['PersonalizedFL'].append(roc_auc_score(ye, mlp_fwd(mlp_train(Xt, yt, g, 25), Xe)[1]))
    Xa = np.vstack([v[0] for v in parts.values()]); ya = np.concatenate([v[1] for v in parts.values()])
    c = mlp_train(Xa, ya, p0, 150)
    for u, (Xt, yt, Xe, ye) in parts.items(): res['Centralized'].append(roc_auc_score(ye, mlp_fwd(c, Xe)[1]))
print('\nMLP (8 tanh units), random hold-out, 100 client-seed values')
for k in REGIMES: print(f'  {k:<15}{np.mean(res[k]):.3f} ± {np.std(res[k]):.3f}')
pcm = {k: np.array(res[k]).reshape(SEEDS, 20).mean(0) for k in REGIMES}
print('  Friedman p', round(stats.friedmanchisquare(*[pcm[k] for k in REGIMES]).pvalue, 3), ' FedAvg-Local CI', np.round(ci95(pcm['FedAvg'] - pcm['Local']), 3),
      ' Wilcoxon p', round(stats.wilcoxon(pcm['FedAvg'], pcm['Local']).pvalue, 3))
# time-ordered + 24h gap with MLP
parts, _ = split_parts(df, clients, chrono(24)); parts = {u: v for u, v in parts.items() if len(np.unique(v[1])) > 1}
p0 = mlp_init(len(F), 0); out = {k: {} for k in REGIMES}
g = p0
for _ in range(ROUNDS):
    loc = [mlp_train(Xt, yt, g, LOCAL_EPOCHS) for Xt, yt, _, _ in parts.values()]
    ns = np.array([len(v[1]) for v in parts.values()]); g = avg(loc, ns / ns.sum())
Xa = np.vstack([v[0] for v in parts.values()]); ya = np.concatenate([v[1] for v in parts.values()]); c = mlp_train(Xa, ya, p0, 150)
for u, (Xt, yt, Xe, ye) in parts.items():
    if len(np.unique(ye)) < 2: continue
    out['Local'][u] = roc_auc_score(ye, mlp_fwd(mlp_train(Xt, yt, p0, 150), Xe)[1])
    out['FedAvg'][u] = roc_auc_score(ye, mlp_fwd(g, Xe)[1])
    out['PersonalizedFL'][u] = roc_auc_score(ye, mlp_fwd(mlp_train(Xt, yt, g, 25), Xe)[1])
    out['Centralized'][u] = roc_auc_score(ye, mlp_fwd(c, Xe)[1])
sc = sorted(out['Local'])
print('  time-ordered+24h MLP (%d scored): ' % len(sc) + '  '.join(f'{k[:4]} {np.mean([out[k][u] for u in sc]):.3f}' for k in REGIMES),
      ' F-vs-L p', round(stats.wilcoxon([out['FedAvg'][u] for u in sc], [out['Local'][u] for u in sc]).pvalue, 3))

# ---------- sleep proxy, night window (paper Section IV-D): output of sleep_proxy_check.py ----------
d = pd.read_csv(f'{R}/revision_2026-11/sleep_ema_night_window.csv')
print('\nNight window 20:00-12:00, responses after 06:00:', len(d), 'users', d.uid.nunique())
for c in ['night_dark', 'night_lock', 'night_union', 'night_dark_total']:
    x = d[d[c] > 0]; r = stats.spearmanr(x.ema, x[c])
    c2 = x.groupby('uid').filter(lambda g: len(g) >= 5).copy()
    e = c2.ema - c2.groupby('uid').ema.transform('mean'); v = c2[c] - c2.groupby('uid')[c].transform('mean')
    print(f'  {c:<17} n={len(x)} users={x.uid.nunique()} Spearman rho={r[0]:.3f} p={r[1]:.2g}; within-person r={stats.pearsonr(e, v)[0]:.3f}; mean {x[c].mean():.2f} h vs EMA {x.ema.mean():.2f} h')

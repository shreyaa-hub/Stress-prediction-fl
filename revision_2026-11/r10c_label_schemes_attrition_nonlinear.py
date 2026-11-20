import sys; sys.path.insert(0, '.')
from common import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

ALL = pd.read_csv(f'{R}/revision_2026-11/features_all_levels.csv')
SCHEMES = {'A': {1: 1, 2: 1, 3: 1, 4: 0, 5: 0}, 'B': {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}, 'C': {2: 1, 3: 1, 4: 0, 5: 0}}

def scheme_df(s):
    a = ALL[ALL.level.isin(SCHEMES[s].keys())].copy(); a['y'] = a.level.map(SCHEMES[s])
    lab = a.copy()
    d = a[a.activity_n >= 2000].dropna(subset=['activity_level', 'voice_ratio']).copy()
    return lab, d

def global_eval(df, model_fn, split):
    X, y, g = df[F].values, df.y.values, df.uid.values; out = []
    for s in range(5):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=s) if split == 'user' else StratifiedKFold(5, shuffle=True, random_state=s)
        it = cv.split(X, y, g) if split == 'user' else cv.split(X, y)
        for tr, te in it:
            m = model_fn().fit(X[tr], y[tr]); out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.mean(out), np.std(out)

LRm = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
RFm = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
MLPm = lambda: make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=1e-2, max_iter=3000, random_state=0))
HGBm = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)

# ---------- attrition ----------
print('Parsed levels:', ALL.level.value_counts().sort_index().to_dict(), 'users', ALL.uid.nunique())
cov = ALL[ALL.activity_n >= 2000].dropna(subset=['activity_level', 'voice_ratio'])
print('coverage retained, all levels:', len(cov), 'of', len(ALL), f'({len(cov)/len(ALL):.1%})')
print('level distribution before/after coverage (%):')
print(pd.concat([ALL.level.value_counts(normalize=True).sort_index(), cov.level.value_counts(normalize=True).sort_index()], axis=1).round(3).T)
share1 = ALL.groupby('uid')['level'].apply(lambda s: (s == 1).mean())
print('per-user share of level-1 answers: median', round(share1.median(), 2), 'IQR', np.round(share1.quantile([.25, .75]).values, 2), 'range', round(share1.min(), 2), round(share1.max(), 2))
labC, dC = scheme_df('C')
print('Scheme C pos rate before/after coverage', round(labC.y.mean(), 3), round(dC.y.mean(), 3))
cl = pick_clients(dC)
print('responses (parsed) covered by the 20 clients:', ALL.uid.isin(cl).sum(), 'of', len(ALL))
# per-user retention: Scheme C kept / parsed
kept = dC.groupby('uid').size().reindex(ALL.uid.unique()).fillna(0); parsed = ALL.groupby('uid').size()
ret = (kept / parsed).dropna()
print('per-user retention parsed->final: median', round(ret.median(), 2), 'IQR', np.round(ret.quantile([.25, .75]).values, 2))
# do retained responses differ? stress-level mean among stress answers (2,3) and calm (4,5) per covered vs dropped
print('coverage-dropped Scheme C rows:', len(labC) - len(dC), 'pos rate among dropped', round(labC[~labC.index.isin(dC.index)].y.mean(), 3))
# association of per-user level-1 share with per-user stressed rate (Scheme C)
rate = dC.groupby('uid').y.mean(); j = pd.concat([rate, share1], axis=1, join='inner')
print('corr(level-1 share, Scheme-C stressed rate) spearman', np.round(stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1]), 3))

# ---------- scheme sensitivity ----------
print('\nLabel-scheme sensitivity')
for s in ['A', 'B', 'C']:
    lab, d = scheme_df(s); c = pick_clients(d)
    sig_u = d.groupby('uid').y.mean().std()
    lr_u, lr_r = global_eval(d, LRm, 'user')[0], global_eval(d, LRm, 'random')[0]
    rf_u, rf_r = global_eval(d, RFm, 'user')[0], global_eval(d, RFm, 'random')[0]
    base = [run_regimes(split_parts(d, c, random_split(sd))[0])[0] for sd in range(SEEDS)]
    pc = {k: np.array([np.mean([r[k][u] for r in base]) for u in c]) for k in REGIMES}
    fr = stats.friedmanchisquare(*[pc[k] for k in REGIMES]).pvalue
    m, lo, hi = ci95(pc['FedAvg'] - pc['Local'])
    parts, _ = split_parts(d, c, chrono(24)); parts = {u: v for u, v in parts.items() if len(np.unique(v[1])) > 1}
    rt = run_regimes(parts)[0]; sc = sorted(rt['Local'])
    tm = {k: np.mean([rt[k][u] for u in sc]) for k in REGIMES}
    print(f' Scheme {s}: n={len(d)} users={d.uid.nunique()} pos={d.y.mean():.3f} sigma={sig_u:.3f} clients={len(c)} ({d.uid.isin(c).sum()} inst)')
    print(f'   RQ1 LR subj {lr_u:.3f} rec {lr_r:.3f} | RF subj {rf_u:.3f} rec {rf_r:.3f}')
    print('   RQ2 random: ' + '  '.join(f'{k[:4]} {np.mean([v for r in base for v in r[k].values()]):.3f}' for k in REGIMES) + f'  Friedman p {fr:.2f}  F-L CI [{lo:+.3f},{hi:+.3f}]')
    print(f'   RQ2 time+24h ({len(sc)} scored): ' + '  '.join(f'{k[:4]} {tm[k]:.3f}' for k in REGIMES) + f'  F-vs-L p {stats.wilcoxon([rt["FedAvg"][u] for u in sc],[rt["Local"][u] for u in sc]).pvalue:.3f}')

# ---------- non-linear models, RQ1 ----------
print('\nNon-linear global models (Scheme C, 1,141 instances): subject-level vs record-level AUC')
for name, fn in [('MLP(16)', MLPm), ('HistGB', HGBm)]:
    u, r = global_eval(dC, fn, 'user'), global_eval(dC, fn, 'random')
    print(f'  {name:<8} subj {u[0]:.3f}±{u[1]:.3f}  rec {r[0]:.3f}±{r[1]:.3f}  inflation {r[0]-u[0]:+.3f}')

# ---------- (b) attrition: test-size subset ----------

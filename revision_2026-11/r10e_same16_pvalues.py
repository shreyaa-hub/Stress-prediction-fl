import sys; sys.path.insert(0,'.')
from common import *
feat, df = load(); clients = pick_clients(df)
S16 = [u for u in clients if u not in ['u01','u24','u33','u46']]
base=[]; near=[]
for s in range(SEEDS):
    parts,times = split_parts(df, clients, random_split(s)); base.append(run_regimes(parts)[0])
    for u in S16:
        ttr,tte = times[u]; near.extend(np.array([np.min(np.abs(ttr-t)) for t in tte])/3600 < 24)
print('overlap same16 random', np.mean(near))
f80=np.array([base[s]['FedAvg'][u] for s in range(SEEDS) for u in S16]); l80=np.array([base[s]['Local'][u] for s in range(SEEDS) for u in S16])
print('p 80 pairs', stats.wilcoxon(f80,l80).pvalue)
f100=np.array([base[s]['FedAvg'][u] for s in range(SEEDS) for u in clients]); l100=np.array([base[s]['Local'][u] for s in range(SEEDS) for u in clients])
print('p 100 pairs (Table VI, random hold-out; corrected from 0.299 to 0.308)', stats.wilcoxon(f100,l100).pvalue)
pc={k:np.array([np.mean([r[k][u] for r in base]) for u in clients]) for k in REGIMES}
print('p 20 clients', stats.wilcoxon(pc['FedAvg'],pc['Local']).pvalue, 'PFL vs FedAvg 20', stats.wilcoxon(pc['PersonalizedFL'],pc['FedAvg']).pvalue)

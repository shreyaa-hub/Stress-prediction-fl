"""Generate Fig. 3 and Fig. 4 of the paper from the result files in the repo.

Usage:
    python make_figures.py [RESULTS_DIR] [OUT_DIR]

RESULTS_DIR must contain labels.csv, features.csv, fl_comparison.csv and
fedavg_curve.npy (all in the GitHub repo); central_curve.npy (cell R10) is optional. Defaults: current folder for both.
Outputs: fig3_heterogeneity.pdf and fig4_convergence.pdf (vector, sized for one
IEEE column), plus PNG previews.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = sys.argv[1] if len(sys.argv) > 1 else "."
out = sys.argv[2] if len(sys.argv) > 2 else "."

labels = pd.read_csv(f"{res}/labels.csv")          # 1,193 Scheme-C responses, 47 users
feat = pd.read_csv(f"{res}/features.csv")
fl = pd.read_csv(f"{res}/fl_comparison.csv")       # 5 seeds x 20 clients (cell 11 order)
curve = np.load(f"{res}/fedavg_curve.npy")         # 5 seeds x 30 rounds (cell 11)
import os
cen_path = f"{res}/central_curve.npy"              # optional (cell R10): centralized AUC after 5r epochs
central = np.load(cen_path) if os.path.exists(cen_path) else None

df = feat[feat.activity_n >= 2000].dropna(subset=["activity_level", "voice_ratio"])
clients = [u for u, s in df.groupby("uid")
           if len(s) >= 15 and s.y.sum() >= 4 and (1 - s.y).sum() >= 4]

# ---- shared style: serif text matching IEEEtran, one accent colour ----
BLUE, GREY, INK, MUTED, BAND = "#2a78d6", "#a9a9a3", "#1f1f1d", "#6b6b66", "#cde2fb"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
COL_W = 3.5  # inches, one IEEE column

# ---------------- Fig. 3: heterogeneity ----------------
g = labels.groupby("uid").y.agg(["mean", "size"]).sort_values("mean")
cm = fl.values.reshape(5, 20, 4).mean(0)           # seed-averaged per-client AUC
loc = cm[:, list(fl.columns).index("Local")]
fed = cm[:, list(fl.columns).index("FedAvg")]

fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.5), gridspec_kw={"width_ratios": [1.25, 1]})
isc = g.index.isin(clients)
x = np.arange(len(g))
a.axhline(labels.y.mean(), color=MUTED, lw=0.6, ls=":", zorder=0)
a.scatter(x[~isc], g["mean"][~isc], s=10, facecolor="white", edgecolor=GREY, lw=0.8, label="other participants")
a.scatter(x[isc], g["mean"][isc], s=12, color=BLUE, edgecolor="white", lw=0.5, label="FL clients (20)")
a.set_xticks([]); a.set_xlabel("participants (sorted)"); a.set_ylabel("stressed rate")
a.set_ylim(-0.05, 1.05)
a.set_title("(a) label heterogeneity", fontsize=7.5, color=INK, loc="left")
a.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.2, borderaxespad=0.1)

b.plot((0, 1.02), (0, 1.02), color=MUTED, lw=0.6, ls=":", zorder=0)
b.scatter(loc, fed, s=12, color=BLUE, edgecolor="white", lw=0.5)
b.set_xlim(0, 1.02); b.set_ylim(0, 1.02); b.set_aspect("equal")
b.set_xticks([0, .5, 1]); b.set_yticks([0, .5, 1])
b.set_xlabel("local-only AUC"); b.set_ylabel("FedAvg AUC")
b.set_title("(b) per-client AUC", fontsize=7.5, color=INK, loc="left")
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(f"{out}/fig3_heterogeneity.pdf", bbox_inches="tight")
fig.savefig(f"{out}/fig3_heterogeneity.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Fig. 4: FedAvg convergence ----------------
m, s = curve.mean(0), curve.std(0)
r = np.arange(1, curve.shape[1] + 1)
pk = int(m.argmax())

fig, ax = plt.subplots(figsize=(COL_W, 1.45))
ax.axhline(0.5, color=MUTED, lw=0.6, ls=":", zorder=0)
ax.text(30, 0.505, "chance", ha="right", va="bottom", fontsize=6.5, color=MUTED)
ax.fill_between(r, m - s, m + s, color=BAND, lw=0, label=r"$\pm1$ s.d. over 5 seeds")
ax.plot(r, m, color=BLUE, lw=1.4, label="FedAvg (mean)")
if central is not None:
    ax.plot(r, central.mean(0), color=INK, lw=0.9, ls=(0, (3, 2)), label="centralized, same epochs")
for i, txt in [(0, f"{m[0]:.3f}"), (pk, f"{m[pk]:.3f} (peak)"), (len(m) - 1, f"{m[-1]:.3f}")]:
    ax.plot(r[i], m[i], "o", ms=3.6, color=BLUE, mec="white", mew=0.6, zorder=3)
    ha = "left" if i == 0 else ("right" if i == len(m) - 1 else "center")
    ax.annotate(txt, (r[i], m[i]), xytext=(0, -9 if i == 0 else 6), textcoords="offset points",
                ha=ha, fontsize=6.5, color=INK)
ax.set_xlim(0.5, 30.5); ax.set_ylim(0.40, 0.70)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("communication round" + (" (centralized: 5 epochs per round)" if central is not None else "")); ax.set_ylabel("held-out AUC")
ax.legend(frameon=False, fontsize=6, loc="lower left", ncol=2, handlelength=1.6, borderaxespad=0.1, columnspacing=0.9)
fig.tight_layout(pad=0.3)
fig.savefig(f"{out}/fig4_convergence.pdf", bbox_inches="tight")
fig.savefig(f"{out}/fig4_convergence.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- numbers quoted in the paper, for checking ----------------
cr = g.loc[clients, "mean"]
print(f"Fig 3a: {len(g)} participants, sigma {g['mean'].std():.3f}; clients' rates {cr.min():.2f}-{cr.max():.2f}")
print(f"Fig 3b: FedAvg > Local for {(fed > loc).sum()} of 20 clients; median |diff| {np.median(np.abs(fed - loc)):.3f}")
print(f"Fig 4 : round 1 {m[0]:.3f}, peak {m[pk]:.3f} at round {pk + 1}, round 30 {m[-1]:.3f}; "
      f"seed s.d. {s.min():.3f}-{s.max():.3f}")
if central is not None:
    print(f"Fig 4 : centralized peak {central.mean(0).max():.3f} after {5 * (central.mean(0).argmax() + 1)} epochs; "
          f"max |FedAvg - centralized| = {np.abs(m - central.mean(0)).max():.3f}")

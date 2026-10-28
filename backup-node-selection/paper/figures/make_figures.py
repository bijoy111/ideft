"""Generate the three result figures for the backup-selection paper.

Latency data is read directly from the archived per-call records in
findings/raw/*.csv.gz via analysis.py, so the figures and the tables cannot drift
apart and nothing is transcribed by hand. Selection-overhead data is parsed from
findings/selection_overhead.txt.

Figure 1 plots mean and p95. It previously plotted p99, which in this testbed is not
a latency measurement: every RPC opens a fresh TCP connection against a listen(5)
backlog, so the tail collapses onto Linux's SYN-retransmission timer. Across all
method/scale combinations exactly zero calls land between 500 and 900 ms, and p99
therefore reads either ~45 ms or ~1045 ms depending only on whether that run's
SYN-drop rate crosses 1%. p95 sits below the quantum and discriminates properly.

Each method keeps a fixed color across figures, with distinct markers/linestyles
as a secondary encoding so the figures stay readable in grayscale.
"""
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE                                                        # backup-node-selection/paper/figures/
WEB = os.path.normpath(os.path.join(HERE, "../../../docs/assets"))  # website copies (SVG)
os.makedirs(OUT, exist_ok=True)
os.makedirs(WEB, exist_ok=True)

# ---- chart chrome (light mode) ----
INK      = "#0b0b0b"
INK2     = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---- fixed identity: color + marker + linestyle per method ----
METHODS = ["concentrated", "random", "round-robin", "heuristic", "optimal", "PPO"]
STYLE = {
    "concentrated": dict(color="#2a78d6", marker="o", ls="-"),
    "random":       dict(color="#1baf7a", marker="s", ls="-"),
    "round-robin":  dict(color="#eda100", marker="^", ls="-"),
    "heuristic":    dict(color="#008300", marker="D", ls="-"),
    "optimal":      dict(color="#4a3aa7", marker="v", ls="-"),
    "PPO":          dict(color="#e34948", marker="P", ls="-"),
}

HOSTS = [9, 16, 32, 128]

sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../code")))
import analysis  # noqa: E402

ARCHIVE = os.path.normpath(os.path.join(HERE, "../../findings/raw/{}hosts.csv.gz"))
OVERHEAD_TXT = os.path.normpath(os.path.join(HERE, "../../findings/selection_overhead.txt"))
# Figure labels vs. the method names used in the data files.
KEY = {"concentrated": "concentrated", "random": "random", "round-robin": "round_robin",
       "heuristic": "heuristic", "optimal": "optimal", "PPO": "rl"}


def load_latency():
    """Per-run service-latency statistics per method per scale, plus bootstrap CIs.

    Statistics are computed within each run and then averaged across runs, matching
    the tables; CIs are cluster bootstraps over runs, so the error bars carry
    between-run variance rather than treating correlated calls as independent.
    """
    mean, p95, lo, hi = ({m: [] for m in METHODS} for _ in range(4))
    for hosts in HOSTS:
        runs = analysis.summarize_runs(ARCHIVE.format(hosts), metric="service_ms")
        for label in METHODS:
            per_run = list(runs.get(KEY[label], {}).values())
            if not per_run:
                for d in (mean, p95, lo, hi):
                    d[label].append(np.nan)
                continue
            means = [r["mean"] for r in per_run]
            ci_lo, ci_hi = analysis.cluster_bootstrap_ci(means)
            mean[label].append(float(np.mean(means)))
            p95[label].append(float(np.mean([r["p95"] for r in per_run])))
            lo[label].append(ci_lo)
            hi[label].append(ci_hi)
    return mean, p95, lo, hi


def load_overhead():
    """Parse mean selection time per method per scale from the overhead run."""
    blocks = re.split(r"^--- ", open(OVERHEAD_TXT).read(), flags=re.M)[1:]
    by_scale = {}
    for block in blocks:
        hosts = int(re.search(r"\((\d+) hosts", block).group(1))
        by_scale[hosts] = {
            m: float(v) for m, v in re.findall(r"^(\w+)\s+([\d.]+)", block, flags=re.M)}
    return {label: [by_scale[h][KEY[label]] for h in HOSTS] for label in METHODS}


def load_rl_spread():
    """Distinct backups each policy used at 128 hosts, from the archived mappings."""
    seen = {}
    for row in analysis.iter_rows(ARCHIVE.format(128)):
        if row["method"] != "rl":
            continue
        seen.setdefault((int(row["context_seed"]), int(row["trial"])), set()).add(row["backup"])
    counts = [len(v) for v in seen.values()]
    heur = {}
    for row in analysis.iter_rows(ARCHIVE.format(128)):
        if row["method"] != "heuristic":
            continue
        heur.setdefault((int(row["context_seed"]), int(row["trial"])), set()).add(row["backup"])
    return (min(counts), max(counts),
            min(len(v) for v in heur.values()), max(len(v) for v in heur.values()))


MEAN, P95, CI_LO, CI_HI = load_latency()
OVERHEAD = load_overhead()


def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def host_axis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(HOSTS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticklabels([str(h) for h in HOSTS])
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_series(ax, data, lo=None, hi=None):
    for m in METHODS:
        s = STYLE[m]
        if lo is not None:
            err = [np.subtract(data[m], lo[m]), np.subtract(hi[m], data[m])]
            ax.errorbar(HOSTS, data[m], yerr=err, color=s["color"], ecolor=s["color"],
                        elinewidth=0.8, capsize=2, capthick=0.8, fmt="none",
                        zorder=2, clip_on=False)
        ax.plot(HOSTS, data[m], color=s["color"], marker=s["marker"],
                ls=s["ls"], lw=1.3, ms=4.2, markeredgecolor="white",
                markeredgewidth=0.5, label=m, clip_on=False, zorder=3)


# ================= Figure 1: latency vs scale (mean + p99) =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True,
                               gridspec_kw=dict(hspace=0.32))
plot_series(ax1, MEAN, CI_LO, CI_HI)
plot_series(ax2, P95)
for ax, ttl in ((ax1, "(a) mean, 95% CI"), (ax2, "(b) p95")):
    ax.set_yscale("log")
    strip(ax)
    ax.set_title(ttl, loc="left", color=INK, pad=3)
host_axis(ax2)
ax1.set_ylabel("latency (ms)")
ax2.set_ylabel("latency (ms)")
ax2.set_xlabel("hosts")
for ax in (ax1, ax2):
    ax.yaxis.set_major_locator(FixedLocator([30, 100, 300, 1000, 3000]))
    ax.set_yticklabels(["30", "100", "300", "1000", "3000"])
    ax.yaxis.set_minor_formatter(NullFormatter())
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center",
           bbox_to_anchor=(0.54, 1.02), columnspacing=1.0,
           handlelength=1.9, handletextpad=0.5)
fig.savefig(f"{OUT}/fig_latency_scale.pdf")
fig.savefig(f"{WEB}/fig_latency_scale.svg")
plt.close(fig)

# ================= Figure 2: selection overhead =================
fig, ax = plt.subplots(figsize=(3.45, 2.35))
plot_series(ax, OVERHEAD)
ax.set_yscale("log")
strip(ax)
host_axis(ax)
ax.set_xlabel("hosts")
ax.set_ylabel("selection time (ms)")
ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.04),
          columnspacing=1.0, handlelength=1.9, handletextpad=0.5)
fig.savefig(f"{OUT}/fig_overhead.pdf")
fig.savefig(f"{WEB}/fig_overhead.svg")
plt.close(fig)

# ============ PPO policy spread at 128 hosts (website only) ============
# Not included in the paper (removed for length); docs/scaled128.html still embeds the
# SVG, so it is kept here so that figure stays reproducible.
# distinct backups used across the 80 primaries (section V-F). The two committed
# checkpoints: degenerate baseline (collapse_check_128hosts.txt, per-seed 2,4,1,5,2)
# and the final fixed model (latency_128hosts.txt, per-seed 29,24,30,24,29);
# heuristic reference measured from the same run (74-76 distinct, max load 2).
configs = ["ent 0\n[64,64]\n400k steps", "ent 0.02\n[256,256]\n1M steps"]
# The collapsed baseline checkpoint is not re-run here (it crashed the harness), so its
# range stays as recorded in findings/collapse_check_128hosts.txt. The shipped model's
# spread and the heuristic reference are measured from the current 128-host archive.
RL_LO, RL_HI, HEUR_LO, HEUR_HI = load_rl_spread()
lo = [1, RL_LO]
hi = [5, RL_HI]
HEURISTIC_SPREAD = HEUR_LO
mid = [(a + b) / 2 for a, b in zip(lo, hi)]
yerr = [[m - a for m, a in zip(mid, lo)], [b - m for m, b in zip(mid, hi)]]
ramp = ["#86b6ef", "#104281"]  # ordinal blue ramp

fig, ax = plt.subplots(figsize=(3.45, 2.2))
x = range(2)
bars = ax.bar(x, mid, width=0.5, color=ramp, zorder=3)
ax.errorbar(x, mid, yerr=yerr, fmt="none", ecolor=INK2, elinewidth=0.9,
            capsize=3, capthick=0.9, zorder=4)
ax.axhline(HEURISTIC_SPREAD, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.text(0.5, HEURISTIC_SPREAD + 1.6, f"heuristic ({HEUR_LO}–{HEUR_HI})",
        ha="center", va="bottom", fontsize=6.8, color=INK2)
for xi, (a, b) in enumerate(zip(lo, hi)):
    lbl = str(a) if a == b else f"{a}–{b}"
    ax.text(xi, hi[xi] + 1.4, lbl, ha="center", va="bottom",
            fontsize=7, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels(configs, fontsize=6.8)
ax.set_ylabel("distinct backups used")
ax.set_ylim(0, 82)
strip(ax)
fig.savefig(f"{OUT}/fig_rl_spread.pdf")
fig.savefig(f"{WEB}/fig_rl_spread.svg")
plt.close(fig)

print("wrote figures to", OUT)

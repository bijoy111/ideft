"""Generate the three result figures for the backup-selection paper.

Data is transcribed from Tables I, II, IV, V, VI and section V-F of paper.tex.
Each method keeps a fixed color across figures, with distinct markers/linestyles
as a secondary encoding so the figures stay readable in grayscale.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator
import os

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

MEAN = {
    "concentrated": [94.99, 157.37, 210.35, 772.41],
    "random":       [45.98, 124.97,  91.00, 101.54],
    "round-robin":  [35.67, 106.94,  98.97,  48.13],
    "heuristic":    [41.06,  39.11,  36.49,  47.92],
    "optimal":      [37.00,  37.25,  35.65,  49.06],
    "PPO":          [36.20,  37.71,  36.41, 457.86],
}
P99 = {
    "concentrated": [3125.54, 3361.56, 5538.62, 18971.94],
    "random":       [1063.38, 1069.84, 1050.90,  1087.47],
    "round-robin":  [1035.42, 1040.29,  363.41,  1058.35],
    "heuristic":    [1054.36, 1050.06,  105.63,  1058.17],
    "optimal":      [1034.23, 1036.45,  103.48,  1057.86],
    "PPO":          [1038.38, 1039.31,  211.59,  7178.12],
}
OVERHEAD = {
    "concentrated": [0.005, 0.013, 0.055,  0.731],
    "random":       [0.021, 0.029, 0.084,  0.639],
    "round-robin":  [0.004, 0.005, 0.015,  0.030],
    "heuristic":    [0.055, 0.161, 0.646,  7.260],
    "optimal":      [0.097, 0.208, 0.913,  8.943],
    "PPO":          [1.81,  3.76, 11.29,  49.64],
}


def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def host_axis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(HOSTS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticklabels([str(h) for h in HOSTS])
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_series(ax, data):
    for m in METHODS:
        s = STYLE[m]
        ax.plot(HOSTS, data[m], color=s["color"], marker=s["marker"],
                ls=s["ls"], lw=1.3, ms=4.2, markeredgecolor="white",
                markeredgewidth=0.5, label=m, clip_on=False, zorder=3)


# ================= Figure 1: latency vs scale (mean + p99) =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True,
                               gridspec_kw=dict(hspace=0.32))
plot_series(ax1, MEAN)
plot_series(ax2, P99)
for ax, ttl in ((ax1, "(a) mean"), (ax2, "(b) p99")):
    ax.set_yscale("log")
    strip(ax)
    ax.set_title(ttl, loc="left", color=INK, pad=3)
host_axis(ax2)
ax1.set_ylabel("latency (ms)")
ax2.set_ylabel("latency (ms)")
ax2.set_xlabel("hosts")
ax1.yaxis.set_major_locator(FixedLocator([30, 100, 300, 1000]))
ax1.set_yticklabels(["30", "100", "300", "1000"])
ax1.yaxis.set_minor_formatter(NullFormatter())
ax2.yaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
ax2.set_yticklabels(["100", "1000", "10000"])
ax2.yaxis.set_minor_formatter(NullFormatter())
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
lo = [1, 24]
hi = [5, 30]
HEURISTIC_SPREAD = 74
mid = [(a + b) / 2 for a, b in zip(lo, hi)]
yerr = [[m - a for m, a in zip(mid, lo)], [b - m for m, b in zip(mid, hi)]]
ramp = ["#86b6ef", "#104281"]  # ordinal blue ramp

fig, ax = plt.subplots(figsize=(3.45, 2.2))
x = range(2)
bars = ax.bar(x, mid, width=0.5, color=ramp, zorder=3)
ax.errorbar(x, mid, yerr=yerr, fmt="none", ecolor=INK2, elinewidth=0.9,
            capsize=3, capthick=0.9, zorder=4)
ax.axhline(HEURISTIC_SPREAD, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.text(0.5, HEURISTIC_SPREAD + 1.6, f"heuristic ({HEURISTIC_SPREAD}–{HEURISTIC_SPREAD + 2})",
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

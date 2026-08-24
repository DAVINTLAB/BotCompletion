#!/usr/bin/env python3
"""Small-multiples figure for the post-budget sensitivity sweep (Sec. 5.3).

One panel per evolved program (rows = reflection model, columns = selection
mode). Each panel shades the .024 noise band anchored at that program's best
sweep value, so the paper's claim ("19 of the 30 cells lie within the noise
band of their program's best") is directly visible: points below the gray band
are the exceptions.

Values transcribed from the paper's sweep (single evaluation runs at
temperature 1.0). Dotted vertical line = the program's training budget.

Output: figures/budget_sweep_small_multiples.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "figures"

NOISE_BAND = 0.024
KS = [5, 10, 20, 30, 40]

# Fixed categorical slots (validated palette): centrality=blue, cluster=aqua,
# latest=yellow. Columns follow the paper's Sec. 3.3 presentation order.
# Training-budget lines match the paper figure as published.
MODES = [
    ("Latest", "#eda100", 30),
    ("Centrality", "#2a78d6", 10),
    ("Cluster", "#1baf7a", 30),
]
ROWS = ["Claude Opus 4.6", "Gemini 3.1 Pro"]

# Submitted Table 4, macro F1 on BotSay-340; keys (row, mode).
F1 = {
    ("Claude Opus 4.6", "Latest"):     [.712, .731, .700, .738, .729],
    ("Claude Opus 4.6", "Centrality"): [.755, .761, .757, .756, .755],
    ("Claude Opus 4.6", "Cluster"):    [.722, .705, .713, .709, .751],
    ("Gemini 3.1 Pro", "Latest"):      [.738, .747, .732, .765, .759],
    ("Gemini 3.1 Pro", "Centrality"):  [.746, .744, .758, .749, .751],
    ("Gemini 3.1 Pro", "Cluster"):     [.750, .761, .767, .791, .747],
}


def main() -> None:
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 3, figsize=(4.8, 2.75), sharex=True, sharey=True, dpi=300)

    n_in_band = 0
    for r, row in enumerate(ROWS):
        for c, (mode, color, train_k) in enumerate(MODES):
            ax = axes[r][c]
            vals = F1[(row, mode)]
            best = max(vals)
            n_in_band += sum(v >= best - NOISE_BAND for v in vals)

            # Noise band anchored at this program's best sweep value.
            ax.axhspan(best - NOISE_BAND, best, color="0.90", zorder=0)
            # Training budget.
            ax.axvline(train_k, color="0.45", linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
            ax.plot(KS, vals, color=color, linewidth=1.4, marker="o",
                    markersize=3, zorder=3)

            # Direct-label the panel's best value at its peak.
            k_best = KS[vals.index(best)]
            ax.annotate(f"{best:.3f}".lstrip("0"), (k_best, best),
                        textcoords="offset points", xytext=(0, 3.5),
                        ha="center", fontsize=6.2, color="0.15", zorder=4)

            ax.set_title(f"{mode} + {'Opus' if 'Opus' in row else 'Gemini'}", loc="left")
            ax.set_xticks(KS)
            ax.set_ylim(0.68, 0.80)
            ax.set_yticks([0.70, 0.75, 0.80])
            ax.set_yticklabels([".70", ".75", ".80"])
            ax.grid(axis="y", linewidth=0.3, alpha=0.3, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, width=0.5)

    print(f"cells within their program's noise band: {n_in_band}/30 (paper says 19)")

    fig.supxlabel("posts per user $k$", fontsize=7.5)
    fig.supylabel("macro F1", fontsize=7.5)
    fig.tight_layout(pad=0.35, w_pad=0.7, h_pad=0.9)
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"budget_sweep_small_multiples.{ext}", bbox_inches="tight")
    print("wrote", FIGDIR / "budget_sweep_small_multiples.pdf")


if __name__ == "__main__":
    main()

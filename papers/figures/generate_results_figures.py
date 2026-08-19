"""Redraw the original historical summaries; no training or benchmark evaluation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"
DATA_PATH = ROOT / "metrics" / "results_summary_data.json"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.2,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.2,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_data() -> dict[str, np.ndarray]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {key: np.array(value) for key, value in raw.items()}


def save_combined_summary(data: dict[str, np.ndarray]) -> None:
    cog_color = "#1f77b4"
    low_color = "#ff7f0e"
    accent = "#2ca02c"
    cosine_color = "#7f7f7f"
    pos_color = "#2ca02c"
    neg_color = "#d62728"
    ablation_colors = ["#1f77b4", "#2ca02c", "#9467bd"]

    fig = plt.figure(figsize=(7.0, 3.0))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.08, 0.92],
        wspace=0.12,
        left=0.055,
        right=0.995,
        bottom=0.145,
        top=0.885,
    )

    left = outer[0].subgridspec(2, 2, hspace=0.56, wspace=0.42)
    ax1 = fig.add_subplot(left[0, 0])
    ax2 = fig.add_subplot(left[0, 1])
    ax3 = fig.add_subplot(left[1, 0])
    ax4 = fig.add_subplot(left[1, 1])

    sigma = data["sigma"]
    ax1.errorbar(
        sigma,
        data["cog_auroc"],
        yerr=data["cog_auroc_std"],
        color=cog_color,
        marker="o",
        linewidth=1.4,
        capsize=2,
        label="CoGround",
    )
    ax1.errorbar(
        sigma,
        data["low_auroc"],
        yerr=data["low_auroc_std"],
        color=low_color,
        marker="s",
        linewidth=1.4,
        capsize=2,
        label="Low-rank",
    )
    ax1.axhline(0.5, color="0.55", linestyle="--", linewidth=0.9)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sigma)
    ax1.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8", "1.6"])
    ax1.set_ylim(0.45, 0.98)
    ax1.set_title(r"Noise $\sigma$ $\to$ AUROC")
    ax1.set_ylabel("AUROC")
    ax1.grid(axis="y", alpha=0.22)

    ax2.errorbar(
        sigma,
        data["cog_shift"],
        yerr=data["cog_shift_std"],
        color=cog_color,
        marker="o",
        linewidth=1.4,
        capsize=2,
    )
    ax2.errorbar(
        sigma,
        data["low_shift"],
        yerr=data["low_shift_std"],
        color=low_color,
        marker="s",
        linewidth=1.4,
        capsize=2,
    )
    ax2.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sigma)
    ax2.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8", "1.6"])
    ax2.set_ylim(-0.16, 0.11)
    ax2.set_title(r"Noise $\sigma$ $\to$ mass shift")
    ax2.set_ylabel("Shift")
    ax2.grid(axis="y", alpha=0.22)

    k_pred = data["k_pred"]
    ax3.errorbar(
        k_pred,
        data["km_cog_auroc"],
        yerr=data["km_cog_auroc_std"],
        color=cog_color,
        marker="o",
        linewidth=1.4,
        capsize=2,
    )
    ax3.errorbar(
        k_pred,
        data["km_low_auroc"],
        yerr=data["km_low_auroc_std"],
        color=low_color,
        marker="s",
        linewidth=1.4,
        capsize=2,
    )
    ax3.axhline(0.5, color="0.55", linestyle="--", linewidth=0.9)
    ax3.axvline(8, color=accent, linestyle=":", linewidth=1.0)
    ax3.text(8.35, 0.94, r"$K_{\mathrm{true}}=8$", color=accent, fontsize=6.3)
    ax3.set_xticks(k_pred)
    ax3.set_ylim(0.30, 1.02)
    ax3.set_title(r"$K_{\mathrm{pred}}$ $\to$ AUROC")
    ax3.set_xlabel(r"Predicted clusters $K_{\mathrm{pred}}$")
    ax3.set_ylabel("AUROC")
    ax3.grid(axis="y", alpha=0.22)

    ax4.plot(k_pred, data["km_cog_shift"], color=cog_color, marker="o", linewidth=1.4)
    ax4.plot(k_pred, data["km_low_shift"], color=low_color, marker="s", linewidth=1.4)
    ax4.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
    ax4.axvline(8, color=accent, linestyle=":", linewidth=1.0)
    ax4.text(8.35, 0.073, r"$K_{\mathrm{true}}=8$", color=accent, fontsize=6.3)
    ax4.set_xticks(k_pred)
    ax4.set_ylim(-0.14, 0.10)
    ax4.set_title(r"$K_{\mathrm{pred}}$ $\to$ mass shift")
    ax4.set_xlabel(r"Predicted clusters $K_{\mathrm{pred}}$")
    ax4.set_ylabel("Shift")
    ax4.grid(axis="y", alpha=0.22)

    right = outer[1].subgridspec(2, 2, height_ratios=[1.08, 1.0], hspace=0.58, wspace=0.44)
    ax5 = fig.add_subplot(right[0, :])
    ax6 = fig.add_subplot(right[1, 0])
    ax7 = fig.add_subplot(right[1, 1])

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7):
        ax.tick_params(axis="both", which="major", pad=1.6, length=2.5)

    taps = data["taps"]
    head = data["head"]
    cosine = data["cosine"]
    delta = head - cosine
    ax5.plot(taps, head, color=cog_color, marker="o", linewidth=1.5, label="CoGround")
    ax5.plot(
        taps, cosine, color=cosine_color, marker="s", linewidth=1.35, linestyle="--", label="Cosine"
    )
    ax5.axvline(0, color="0.65", linestyle=":", linewidth=0.9)
    ax5.text(0.45, 0.246, "best tap", fontsize=6.3, color="0.35")
    ax5.set_xticks(taps)
    ax5.set_ylim(0.13, 0.255)
    ax5.set_title("RefCOCO layer scan")
    ax5.set_xlabel("Tap layer")
    ax5.set_ylabel("Pointing")
    ax5.grid(axis="y", alpha=0.22)
    ax5.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.5, columnspacing=0.8)

    bar_colors = [pos_color if value >= 0 else neg_color for value in delta]
    ax6.bar(np.arange(len(taps)), delta, color=bar_colors, width=0.72)
    ax6.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
    ax6.set_xticks(np.arange(len(taps)))
    ax6.set_xticklabels([str(tap) for tap in taps])
    ax6.set_ylim(-0.06, 0.12)
    ax6.set_title(r"Gap $\Delta$")
    ax6.set_xlabel("Tap")
    ax6.set_ylabel(r"$\Delta$ point.")
    ax6.grid(axis="y", alpha=0.22)

    conditions = ["Head", r"+$\lambda_g$"]
    cosine_baseline = 0.138
    ax7.bar(
        np.arange(2),
        data["pointing"][:2],
        yerr=data["pointing_std"][:2],
        color=ablation_colors[:2],
        edgecolor="none",
        width=0.68,
        capsize=3,
    )
    ax7.axhline(cosine_baseline, color=cosine_color, linestyle="--", linewidth=1.3)
    ax7.text(-0.38, cosine_baseline + 0.004, "cos = 0.138", color=cosine_color, fontsize=6.1)
    ax7.set_xticks(np.arange(2))
    ax7.set_xticklabels(conditions)
    ax7.set_ylim(0.10, 0.26)
    ax7.set_title("Layer-0 multi-seed")
    ax7.set_ylabel("Pointing")
    ax7.grid(axis="y", alpha=0.22)

    fig.text(
        0.245,
        0.972,
        "Historical synthetic summaries",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    fig.text(
        0.755,
        0.972,
        "Historical RefCOCO summaries",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    fig.savefig(OUTDIR / "results_summary.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    set_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    save_combined_summary(get_data())


if __name__ == "__main__":
    main()

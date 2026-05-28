"""
plot_avoided_crossing.py
─────────────────────────
Visualise results produced by study_avoided_crossing.py.

Usage
-----
    # Single run — full time-series panels
    python plot_avoided_crossing.py --file results/avoided_crossing_single_tau10.0_np16.pkl

    # Tau sweep — summary + per-tau traces
    python plot_avoided_crossing.py --file results/avoided_crossing_vs_tau_np16.pkl

    # n_params sweep
    python plot_avoided_crossing.py --file results/avoided_crossing_vs_params_tau10.0.pkl
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "linear": "#3B82F6",  # blue
    "catalyst": "#22C55E",  # green
    "optimal": "#EF4444",  # red
    "gs": "#A855F7",  # purple (adiabatic GS reference)
}
LABELS = {
    "linear": "Linear QA",
    "catalyst": "Catalyst (XX)",
    "optimal": "Optimal control (GRAPE)",
    "gs": "Adiabatic GS ref.",
}
LW = 2.5


def _mark_ac(ax, r):
    ax.axvline(
        r["s_min_gap"],
        color="grey",
        lw=1.5,
        ls="--",
        alpha=0.7,
        label=f"AC  s={r['s_min_gap']:.2f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single result: 6-panel figure + zoom around avoided crossing
# ─────────────────────────────────────────────────────────────────────────────
def plot_single(r: dict, output_dir: Path, show: bool = True):
    time = r["time"]
    tau = r["tau"]
    s = time / tau

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"Avoided crossing — τ={tau:.1f}, N_c={r['n_params']}  |  "
        f"min-gap={r['min_gap']:.5f} @ s={r['s_min_gap']:.2f}",
        fontsize=13,
    )

    # ── spectrum ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    spec = r["spectrum"]
    for k in range(min(6, spec.shape[1])):
        ax.plot(s, spec[:, k], lw=1.2, alpha=0.8)
    _mark_ac(ax, r)
    ax.set_xlabel("s = t/τ")
    ax.set_ylabel("Energy")
    ax.set_title("Spectrum of linear H(s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── optimal schedule ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(
        s, r["h_driver_opt"], color=COLORS["optimal"], lw=LW, label="h_driver (opt)"
    )
    ax.plot(
        s, r["h_target_opt"], color="orange", lw=LW, ls="--", label="h_target (opt)"
    )
    ax.plot(
        s,
        1 - s,
        color=COLORS["linear"],
        lw=1.5,
        ls=":",
        alpha=0.6,
        label="1-s  (linear)",
    )
    ax.plot(s, s, color="grey", lw=1.5, ls=":", alpha=0.6, label="s  (linear)")
    _mark_ac(ax, r)
    ax.set_xlabel("s = t/τ")
    ax.set_ylabel("Amplitude")
    ax.set_title("Optimal control schedule")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── magic M2(t) ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(s, r["m2_lin"], color=COLORS["linear"], lw=LW, label=LABELS["linear"])
    ax.plot(s, r["m2_cat"], color=COLORS["catalyst"], lw=LW, label=LABELS["catalyst"])
    ax.plot(s, r["m2_opt"], color=COLORS["optimal"], lw=LW, label=LABELS["optimal"])
    ax.plot(s, r["magic_gs"], color=COLORS["gs"], lw=1.5, ls="--", label=LABELS["gs"])
    _mark_ac(ax, r)
    ax.set_xlabel("s = t/τ")
    ax.set_ylabel(r"$\mathcal{M}_2(t)$")
    ax.set_title("Stabilizer Rényi Entropy (magic)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── entanglement entropy S_A(t) ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(s, r["ee_lin"], color=COLORS["linear"], lw=LW, label=LABELS["linear"])
    ax.plot(s, r["ee_cat"], color=COLORS["catalyst"], lw=LW, label=LABELS["catalyst"])
    ax.plot(s, r["ee_opt"], color=COLORS["optimal"], lw=LW, label=LABELS["optimal"])
    ax.plot(s, r["ee_gs"], color=COLORS["gs"], lw=1.5, ls="--", label=LABELS["gs"])
    _mark_ac(ax, r)
    ax.set_xlabel("s = t/τ")
    ax.set_ylabel(r"$\mathcal{S}_A(t)$")
    ax.set_title("Entanglement entropy (half-chain)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── ground-state population p0(t) ─────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(
        s, r["probs_lin"][:, 0], color=COLORS["linear"], lw=LW, label=LABELS["linear"]
    )
    ax.plot(
        s,
        r["probs_cat"][:, 0],
        color=COLORS["catalyst"],
        lw=LW,
        label=LABELS["catalyst"],
    )
    ax.plot(
        s, r["probs_opt"][:, 0], color=COLORS["optimal"], lw=LW, label=LABELS["optimal"]
    )
    ax.axhline(1.0, color="grey", lw=1, ls=":", alpha=0.5)
    _mark_ac(ax, r)
    ax.set_xlabel("s = t/τ")
    ax.set_ylabel(r"$p_0(t)$")
    ax.set_title("Ground-state population")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── energy ⟨H_linear(t)⟩ ─────────────────────────────────────────────────
    ax = axes[2, 1]
    ax.plot(s, r["energy_lin"], color=COLORS["linear"], lw=LW, label=LABELS["linear"])
    ax.plot(
        s, r["energy_cat"], color=COLORS["catalyst"], lw=LW, label=LABELS["catalyst"]
    )
    ax.plot(s, r["energy_opt"], color=COLORS["optimal"], lw=LW, label=LABELS["optimal"])
    ax.plot(
        s,
        r["spectrum"][:, 0],
        color=COLORS["gs"],
        lw=1.5,
        ls="--",
        label="GS energy",
        alpha=0.8,
    )
    _mark_ac(ax, r)
    ax.set_xlabel("s = t/τ")
    ax.set_ylabel(r"$\langle H_{lin}(t)\rangle$")
    ax.set_title("Energy (on linear H)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    stem = f"avoided_crossing_single_tau{r['tau']:.1f}_np{r['n_params']}"
    out = output_dir / f"{stem}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    plt.close(fig)

    # ── zoomed figure around the avoided crossing ─────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f"Magic & Entanglement near AC — τ={tau:.1f}", fontsize=12)
    s_ac = r["s_min_gap"]
    window = 0.25
    mask = (s >= max(0, s_ac - window)) & (s <= min(1, s_ac + window))

    for ax, key_lin, key_cat, key_opt, key_gs, ylabel, title in [
        (
            ax1,
            "m2_lin",
            "m2_cat",
            "m2_opt",
            "magic_gs",
            r"$\mathcal{M}_2$",
            "Magic (zoomed)",
        ),
        (
            ax2,
            "ee_lin",
            "ee_cat",
            "ee_opt",
            "ee_gs",
            r"$\mathcal{S}_A$",
            "Entanglement (zoomed)",
        ),
    ]:
        ax.plot(
            s[mask],
            r[key_lin][mask],
            color=COLORS["linear"],
            lw=LW,
            label=LABELS["linear"],
        )
        ax.plot(
            s[mask],
            r[key_cat][mask],
            color=COLORS["catalyst"],
            lw=LW,
            label=LABELS["catalyst"],
        )
        ax.plot(
            s[mask],
            r[key_opt][mask],
            color=COLORS["optimal"],
            lw=LW,
            label=LABELS["optimal"],
        )
        ax.plot(
            s[mask],
            r[key_gs][mask],
            color=COLORS["gs"],
            lw=1.5,
            ls="--",
            label=LABELS["gs"],
        )
        ax.axvline(s_ac, color="grey", lw=1.5, ls="--", alpha=0.7)
        ax.set_xlabel("s = t/τ", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out2 = output_dir / f"{stem}_zoom_ac.pdf"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
    if show:
        plt.show()
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep results: integrated quantities and fidelity vs sweep parameter
# ─────────────────────────────────────────────────────────────────────────────
def plot_sweep(data: dict, sweep_key: str, output_dir: Path, show: bool = True):
    results = data["results"]
    param_vals = [r[sweep_key] for r in results]
    xlabel = {"tau": r"$\tau$", "n_params": r"$N_c$"}[sweep_key]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Avoided crossing — sweep over {sweep_key}", fontsize=13)

    # fidelity
    ax = axes[0]
    ax.plot(
        param_vals,
        [r["fidelity_lin"] for r in results],
        "o-",
        color=COLORS["linear"],
        lw=LW,
        label=LABELS["linear"],
    )
    ax.plot(
        param_vals,
        [r["fidelity_cat"] for r in results],
        "s-",
        color=COLORS["catalyst"],
        lw=LW,
        label=LABELS["catalyst"],
    )
    ax.plot(
        param_vals,
        [r["fidelity_opt"] for r in results],
        "^-",
        color=COLORS["optimal"],
        lw=LW,
        label=LABELS["optimal"],
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Final fidelity", fontsize=12)
    ax.set_title("Ground-state fidelity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # integrated magic
    ax = axes[1]
    ax.plot(
        param_vals,
        [r["int_magic_lin"] for r in results],
        "o-",
        color=COLORS["linear"],
        lw=LW,
        label=LABELS["linear"],
    )
    ax.plot(
        param_vals,
        [r["int_magic_cat"] for r in results],
        "s-",
        color=COLORS["catalyst"],
        lw=LW,
        label=LABELS["catalyst"],
    )
    ax.plot(
        param_vals,
        [r["int_magic_opt"] for r in results],
        "^-",
        color=COLORS["optimal"],
        lw=LW,
        label=LABELS["optimal"],
    )
    ax.plot(
        param_vals,
        [r["int_magic_gs"] for r in results],
        "D--",
        color=COLORS["gs"],
        lw=1.5,
        label=LABELS["gs"],
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$\int \mathcal{M}_2\,dt$", fontsize=12)
    ax.set_title("Integrated magic")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # integrated EE
    ax = axes[2]
    ax.plot(
        param_vals,
        [r["int_ee_lin"] for r in results],
        "o-",
        color=COLORS["linear"],
        lw=LW,
        label=LABELS["linear"],
    )
    ax.plot(
        param_vals,
        [r["int_ee_cat"] for r in results],
        "s-",
        color=COLORS["catalyst"],
        lw=LW,
        label=LABELS["catalyst"],
    )
    ax.plot(
        param_vals,
        [r["int_ee_opt"] for r in results],
        "^-",
        color=COLORS["optimal"],
        lw=LW,
        label=LABELS["optimal"],
    )
    ax.plot(
        param_vals,
        [r["int_ee_gs"] for r in results],
        "D--",
        color=COLORS["gs"],
        lw=1.5,
        label=LABELS["gs"],
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$\int \mathcal{S}_A\,dt$", fontsize=12)
    ax.set_title("Integrated entanglement")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"avoided_crossing_sweep_{sweep_key}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    plt.close(fig)

    # ── overlay of M2(t) and S_A(t) for all protocols, coloured by param ─────
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 13))
    fig2.suptitle(f"Time-traces — sweep over {sweep_key}", fontsize=13)
    cmap = plt.cm.plasma
    n = len(results)

    proto_keys = [
        ("m2_lin", "ee_lin", COLORS["linear"], "linear"),
        ("m2_cat", "ee_cat", COLORS["catalyst"], "catalyst"),
        ("m2_opt", "ee_opt", COLORS["optimal"], "optimal"),
    ]
    for row, (m2_key, ee_key, _, proto) in enumerate(proto_keys):
        ax_m, ax_e = axes2[row, 0], axes2[row, 1]
        for k, r in enumerate(results):
            c = cmap(k / max(n - 1, 1))
            lbl = f"{sweep_key}={r[sweep_key]}"
            s = r["time"] / r["tau"]
            ax_m.plot(s, r[m2_key], color=c, lw=LW, label=lbl)
            ax_e.plot(s, r[ee_key], color=c, lw=LW, label=lbl)
        for ax in (ax_m, ax_e):
            ax.set_xlabel("s = t/τ", fontsize=11)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=0.3)
        ax_m.set_ylabel(r"$\mathcal{M}_2$", fontsize=12)
        ax_e.set_ylabel(r"$\mathcal{S}_A$", fontsize=12)
        ax_m.set_title(f"Magic — {LABELS[proto]}")
        ax_e.set_title(f"Entanglement — {LABELS[proto]}")

    plt.tight_layout()
    out2 = output_dir / f"avoided_crossing_sweep_{sweep_key}_traces.pdf"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
    if show:
        plt.show()
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help=".pkl from study script")
    parser.add_argument(
        "--output", default=None, help="Output dir (default: same as input)"
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    pkl_path = Path(args.file)
    output_dir = Path(args.output) if args.output else pkl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    sweep = data.get("sweep", "single")

    if sweep == "single":
        plot_single(data["result"], output_dir=output_dir, show=show)

    elif sweep == "tau":
        for r in data["results"]:
            plot_single(r, output_dir=output_dir, show=False)
        plot_sweep(data, sweep_key="tau", output_dir=output_dir, show=show)

    elif sweep == "n_params":
        for r in data["results"]:
            plot_single(r, output_dir=output_dir, show=False)
        plot_sweep(data, sweep_key="n_params", output_dir=output_dir, show=show)


if __name__ == "__main__":
    main()

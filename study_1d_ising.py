"""
study_1d_ising.py
─────────────────
Three independent studies for 1D antiferromagnetic Ising model:

    study_vs_tau        : fix nqubits, n_params  — sweep tau
    study_vs_params     : fix nqubits, tau       — sweep n_params
    study_vs_size       : fix tau, n_params      — sweep nqubits

For each (protocol × hyperparameter) we collect:
    - full M2(t) and S_A(t) along the evolution
    - probabilities onto instantaneous eigenstates
    - optimal schedule h_driver(t), h_target(t)
    - integrated magic  ∫ M2 dt
    - integrated entropy ∫ S_A dt
    - final fidelity

Protocols compared:
    - linear           : standard QA
    - optimal_control  : JAX positive fourier schedule
    - adiabatic        : exact counterdiabatic reference — M2 and EE of the
                         instantaneous GS of H(t) at each time step.
                         This is the ideal limit of perfect counterdiabatic
                         driving, measured without evolving any state.

Usage
-----
    python study_1d_ising.py --study tau    --nqubits 6 --n_params 5  --output results/
    python study_1d_ising.py --study params --nqubits 6 --tau 10.0   --output results/
    python study_1d_ising.py --study size   --tau 10.0  --n_params 5  --output results/
"""

import argparse
import pickle
import time as _time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.jax_utils import JaxSchedulerModel, JaxTrainer, SREJax
from src.sparse_grape_method import SparseGRAPEModel, SparseGRAPETrainer
from src.utils import Sector, EntanglementEntropy


# ─────────────────────────────────────────────────────────────────────────────
# 1D Ising coupling matrix  (periodic antiferromagnetic chain)
# ─────────────────────────────────────────────────────────────────────────────
def build_1d_ising_jij(nqubits: int) -> np.ndarray:
    return np.roll(np.eye(nqubits), shift=1, axis=1) + np.roll(
        np.eye(nqubits), shift=-1, axis=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fidelity helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_fidelity(psi, target_hamiltonian_s):
    """Projection onto GS subspace of target Hamiltonian in sector."""
    evals, evecs = eigsh(target_hamiltonian_s.astype(complex), which="SA", k=4)
    order = np.argsort(evals)
    evecs = evecs[:, order]
    gs_subspace = evecs[:, :1]
    return float(np.sum(np.abs(gs_subspace.conj().T @ psi) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Core: run one full experiment for a given (nqubits, tau, n_params)
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    nqubits: int,
    tau: float,
    n_params: int,
    schedule_type: str = "positive fourier",
    nlevels: int = 6,
    maxiter: int = 500,
    verbose: bool = False,
    method: str = "sparse_grape",
    inner_steps_per_tau: int = 20,
) -> dict:
    """
    Run linear and optimal control protocols.
    The counteradiabatic reference is the instantaneous GS of H(t) — the
    ideal limit of perfect adiabatic following — measured directly from
    the eigenstates of H(t) without time-evolving any state.
    """
    t0 = _time.time()

    # ── build system ──────────────────────────────────────────────────────────
    jij = build_1d_ising_jij(nqubits)
    PS = Sector(nqubits=nqubits)
    dim = 2**nqubits
    dim_s = dim // 2
    time_steps = int(inner_steps_per_tau * tau)
    time = np.linspace(0, tau, time_steps)
    delta_t = time[1] - time[0]

    target_hamiltonian = get_longitudinal_hamiltonian(jij)
    driver_hamiltonian = get_driver_hamiltonian(nqubits=nqubits)
    target_hamiltonian_s = PS.project(target_hamiltonian)
    driver_hamiltonian_s = PS.project(driver_hamiltonian)

    # ── initial state: all-plus projected to sector ───────────────────────────
    psi_init_full = np.ones(dim, dtype=complex) / np.sqrt(dim)
    psi_init_s = PS.project(psi_init_full)

    # ── SRE + EE ──────────────────────────────────────────────────────────────
    sre = SREJax(n_qubits=nqubits, batch_size=4096)
    ee = EntanglementEntropy(nqubits=nqubits, n_A=nqubits // 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: compute M2, EE, probabilities for a state history
    # projected against instantaneous eigenstates
    # ─────────────────────────────────────────────────────────────────────────
    def analyse_history(psi_history, eigstates_ref):
        m2 = np.zeros(time_steps)
        s_a = np.zeros(time_steps)
        probs = np.zeros((time_steps, nlevels))
        for i in range(time_steps):
            psi_full = PS.lift(psi_history[i])
            m2[i] = sre(psi_full)
            s_a[i] = ee.von_neumann(psi_full)
            probs[i] = np.abs(eigstates_ref[i].conj().T @ psi_history[i]) ** 2
        return m2, s_a, probs

    # ─────────────────────────────────────────────────────────────────────────
    # Adiabatic reference: instantaneous GS and 1st excited state of H(t)
    # This IS the counteradiabatic ideal — no evolution needed, just
    # diagonalize H(t) at each step and measure M2 and EE of the GS.
    # ─────────────────────────────────────────────────────────────────────────
    print(
        f"  [{nqubits}q τ={tau} Np={n_params}] Adiabatic reference (instantaneous GS)..."
    )
    magic_gs = np.zeros(time_steps)
    ee_gs = np.zeros(time_steps)
    magic_ex = np.zeros(time_steps)
    ee_ex = np.zeros(time_steps)
    spectrum = np.zeros((time_steps, nlevels))
    eigstates_ref = np.zeros((time_steps, dim_s, nlevels), dtype=complex)

    for i, t in enumerate(time):
        H_t = (1 - t / tau) * driver_hamiltonian_s + (t / tau) * target_hamiltonian_s
        evals, evecs = eigsh(H_t.astype(complex), which="SA", k=nlevels)
        order = np.argsort(evals)
        spectrum[i] = evals[order]
        evecs = evecs[:, order].astype(complex)
        eigstates_ref[i] = evecs

        # GS of H(t) — the ideal counteradiabatic state
        psi_gs_i = evecs[:, 0]
        magic_gs[i] = sre(PS.lift(psi_gs_i))
        ee_gs[i] = ee.von_neumann(PS.lift(psi_gs_i))

        # 1st excited state of H(t)
        psi_ex_i = evecs[:, 1]
        magic_ex[i] = sre(PS.lift(psi_ex_i))
        ee_ex[i] = ee.von_neumann(PS.lift(psi_ex_i))

    min_gap = float(np.min(spectrum[:, 1] - spectrum[:, 0]))
    t_min_gap = float(time[np.argmin(spectrum[:, 1] - spectrum[:, 0])])

    # ─────────────────────────────────────────────────────────────────────────
    # Protocol 1: LINEAR
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [{nqubits}q τ={tau} Np={n_params}] Linear evolution...")
    psi_lin = psi_init_s.copy()
    psi_history_lin = np.zeros((time_steps, dim_s), dtype=complex)
    energy_lin = np.zeros(time_steps)

    for i, t in enumerate(time):
        H_t = (1 - t / tau) * driver_hamiltonian_s + (t / tau) * target_hamiltonian_s
        psi_lin = expm_multiply(-1j * delta_t * H_t, psi_lin)
        psi_history_lin[i] = psi_lin
        energy_lin[i] = (psi_lin.conj() @ H_t @ psi_lin).real

    m2_lin, ee_lin, probs_lin = analyse_history(psi_history_lin, eigstates_ref)
    fidelity_lin = compute_fidelity(psi_lin, target_hamiltonian_s)

    # ─────────────────────────────────────────────────────────────────────────
    # Protocol 2: OPTIMAL CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [{nqubits}q τ={tau} Np={n_params}] Optimal control...")
    if method == "sparse_grape":
        model = SparseGRAPEModel(
            initial_state=psi_init_s,
            target_hamiltonian=target_hamiltonian_s,
            initial_hamiltonian=driver_hamiltonian_s,
            reference_hamiltonian=target_hamiltonian_s,
            tf=tau,
            number_of_parameters=n_params,
            nsteps=time_steps,
            type=schedule_type,
            seed=42,
            mode="annealing ansatz",
            random=False,
        )
        trainer = SparseGRAPETrainer(
            model,
            maxiter=maxiter,
            tol=1e-3,
            ftol=1e-5,
            gtol=1e-4,
            verbose=verbose,
        )
    else:
        model = JaxSchedulerModel(
            initial_state=psi_init_s,
            target_hamiltonian=target_hamiltonian_s,
            initial_hamiltonian=driver_hamiltonian_s,
            reference_hamiltonian=target_hamiltonian_s,
            tf=tau,
            number_of_parameters=n_params,
            nsteps=time_steps,
            type=schedule_type,
            seed=42,
            mode="annealing ansatz",
            random=False,
        )
        trainer = JaxTrainer(
            model, maxiter=maxiter, tol=1e-3, ftol=1e-5, gtol=1e-4, verbose=verbose
        )
    opt_results = trainer.run()
    h_driver_opt = opt_results["h_driver"]
    h_target_opt = opt_results["h_target"]

    psi_opt = psi_init_s.copy()
    psi_history_opt = np.zeros((time_steps, dim_s), dtype=complex)
    energy_opt = np.zeros(time_steps)

    for i, t in enumerate(time):
        H_t = (
            h_driver_opt[i] * driver_hamiltonian_s
            + h_target_opt[i] * target_hamiltonian_s
        )
        psi_opt = expm_multiply(-1j * delta_t * H_t, psi_opt)
        psi_history_opt[i] = psi_opt
        # measure energy on the linear Hamiltonian for fair comparison
        H_lin = (1 - t / tau) * driver_hamiltonian_s + (t / tau) * target_hamiltonian_s
        energy_opt[i] = (psi_opt.conj() @ H_lin @ psi_opt).real

    m2_opt, ee_opt, probs_opt = analyse_history(psi_history_opt, eigstates_ref)
    fidelity_opt = compute_fidelity(psi_opt, target_hamiltonian_s)

    elapsed = _time.time() - t0
    print(
        f"  Done in {elapsed:.1f}s  "
        f"F_lin={fidelity_lin:.3f}  F_opt={fidelity_opt:.3f}  "
        f"gap={min_gap:.4f}"
    )

    return {
        # ── hyperparameters ───────────────────────────────────────────────────
        "nqubits": nqubits,
        "tau": tau,
        "n_params": n_params,
        "schedule_type": schedule_type,
        "time": time,
        "jij": jij,
        # ── spectrum & gap ────────────────────────────────────────────────────
        "spectrum": spectrum,
        "min_gap": min_gap,
        "t_min_gap": t_min_gap,
        # ── adiabatic reference: instantaneous GS / 1st excited of H(t) ──────
        # This is the ideal counteradiabatic limit — what perfect CD would give
        "magic_gs": magic_gs,
        "ee_gs": ee_gs,
        "magic_ex": magic_ex,
        "ee_ex": ee_ex,
        "int_magic_gs": float(np.trapz(magic_gs, time)),
        "int_ee_gs": float(np.trapz(ee_gs, time)),
        # ── linear ────────────────────────────────────────────────────────────
        "energy_lin": energy_lin,
        "magic_lin": m2_lin,
        "ee_lin": ee_lin,
        "probs_lin": probs_lin,
        "fidelity_lin": fidelity_lin,
        "int_magic_lin": float(np.trapz(m2_lin, time)),
        "int_ee_lin": float(np.trapz(ee_lin, time)),
        # ── optimal control ───────────────────────────────────────────────────
        "h_driver_opt": h_driver_opt,
        "h_target_opt": h_target_opt,
        "opt_energy_history": opt_results["history_energy"],
        "energy_opt": energy_opt,
        "magic_opt": m2_opt,
        "ee_opt": ee_opt,
        "probs_opt": probs_opt,
        "fidelity_opt": fidelity_opt,
        "int_magic_opt": float(np.trapz(m2_opt, time)),
        "int_ee_opt": float(np.trapz(ee_opt, time)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Three sweep functions
# ─────────────────────────────────────────────────────────────────────────────
def study_vs_tau(nqubits, n_params, taus, output_dir, **kwargs):
    print(f"\n══ Study vs tau  (nqubits={nqubits}, n_params={n_params}) ══")
    schedule_type = kwargs.get("schedule_type", "fourier")
    results = []
    for tau in taus:
        print(f"\n── tau={tau} ──")
        results.append(
            run_experiment(nqubits=nqubits, tau=tau, n_params=n_params, **kwargs)
        )
    path = (
        Path(output_dir) / f"study_vs_tau_n{nqubits}_np{n_params}_{schedule_type}.pkl"
    )
    with open(path, "wb") as f:
        pickle.dump({"sweep": "tau", "taus": taus, "results": results}, f)
    print(f"\nSaved → {path}")
    _print_summary(results, sweep_key="tau")
    return results


def study_vs_params(nqubits, tau, n_params_list, output_dir, **kwargs):
    print(f"\n══ Study vs n_params  (nqubits={nqubits}, tau={tau}) ══")
    schedule_type = kwargs.get("schedule_type", "fourier")
    results = []
    for n_params in n_params_list:
        print(f"\n── n_params={n_params} ──")
        results.append(
            run_experiment(nqubits=nqubits, tau=tau, n_params=n_params, **kwargs)
        )
    path = (
        Path(output_dir)
        / f"study_vs_params_n{nqubits}_tau{tau:.1f}_{schedule_type}.pkl"
    )
    with open(path, "wb") as f:
        pickle.dump(
            {"sweep": "n_params", "n_params_list": n_params_list, "results": results}, f
        )
    print(f"\nSaved → {path}")
    _print_summary(results, sweep_key="n_params")
    return results


def study_vs_size(tau, n_params, nqubits_list, output_dir, **kwargs):
    print(f"\n══ Study vs size  (tau={tau}, n_params={n_params}) ══")
    schedule_type = kwargs.get("schedule_type", "fourier")
    results = []
    for nqubits in nqubits_list:
        print(f"\n── nqubits={nqubits} ──")
        results.append(
            run_experiment(nqubits=nqubits, tau=tau, n_params=n_params, **kwargs)
        )
    path = (
        Path(output_dir)
        / f"study_vs_size_tau{tau:.1f}_np{n_params}_{schedule_type}.pkl"
    )
    with open(path, "wb") as f:
        pickle.dump(
            {"sweep": "nqubits", "nqubits_list": nqubits_list, "results": results}, f
        )
    print(f"\nSaved → {path}")
    _print_summary(results, sweep_key="nqubits")
    return results


def _print_summary(results, sweep_key):
    print(f'\n{"─"*80}')
    print(
        f'{"param":>8}  {"gap":>8}  {"F_lin":>8}  {"F_opt":>8}  '
        f'{"∫M2_lin":>10}  {"∫M2_opt":>10}  {"∫M2_gs":>10}'
    )
    print(f'{"─"*80}')
    for r in results:
        print(
            f"{r[sweep_key]:>8}  "
            f'{r["min_gap"]:>8.4f}  '
            f'{r["fidelity_lin"]:>8.4f}  '
            f'{r["fidelity_opt"]:>8.4f}  '
            f'{r["int_magic_lin"]:>10.4f}  '
            f'{r["int_magic_opt"]:>10.4f}  '
            f'{r["int_magic_gs"]:>10.4f}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["tau", "params", "size"], required=True)
    parser.add_argument("--nqubits", type=int, default=6)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--time_steps", type=int, default=20)
    parser.add_argument("--n_params", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--schedule_type", type=str, default="F-CRAB")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument(
        "--method", choices=["sparse_grape", "jax"], default="sparse_grape"
    )
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        schedule_type=args.schedule_type,
        maxiter=args.maxiter,
        verbose=True,
        inner_steps_per_tau=args.time_steps,
    )

    if args.study == "tau":
        taus = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        study_vs_tau(
            nqubits=args.nqubits,
            n_params=args.n_params,
            taus=taus,
            output_dir=args.output,
            **kwargs,
        )

    elif args.study == "params":
        n_params_list = [1, 2, 5, 10, 15, 20, 40, 100, 200, 300]
        study_vs_params(
            nqubits=args.nqubits,
            tau=args.tau,
            n_params_list=n_params_list,
            output_dir=args.output,
            **kwargs,
        )

    elif args.study == "size":
        nqubits_list = [4, 6, 8, 10, 12]
        study_vs_size(
            tau=args.tau,
            n_params=args.n_params,
            nqubits_list=nqubits_list,
            output_dir=args.output,
            **kwargs,
        )


if __name__ == "__main__":
    main()

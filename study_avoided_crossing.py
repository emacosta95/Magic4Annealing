"""
study_avoided_crossing.py
─────────────────────────
Study of the 5-qubit MWIS avoided-crossing instance under three protocols:

    linear          : standard QA  H(t) = (1-s) Hd + s Hp
    catalyst        : non-stoquastic driver  H(t) = (1-s) Hd + s Hp + s(1-s) Hcat
    optimal_control : F-CRAB warm-start (GRAPE)

For each protocol we measure along the time evolution:
    - M2(t)   : Stabilizer Rényi Entropy (magic)  via SREJax
    - S_A(t)  : von Neumann entanglement entropy   via EntanglementEntropy
    - p_k(t)  : overlap onto instantaneous eigenstates of the *linear* H(t)
    - energy(t): ⟨ψ|H_linear(t)|ψ⟩  (fair comparison across protocols)

An adiabatic reference track is also computed:
    - M2 and S_A of the instantaneous GS of the linear H(t)

Studies available:
    study_vs_tau    : sweep annealing time  (fix n_params)
    study_vs_params : sweep Fourier modes   (fix tau)

Usage
-----
    python study_avoided_crossing.py --study tau   --n_params 5  --output results/
    python study_avoided_crossing.py --study params --tau 10.0   --output results/
    python study_avoided_crossing.py --study single --tau 10.0 --n_params 5 --output results/
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
    get_unbiased_catalyst_term,
)
from src.jax_utils import JaxSchedulerModel, JaxTrainer, SREJax
from src.sparse_grape_method import SparseGRAPEModel, SparseGRAPETrainer
from src.utils import Sector, EntanglementEntropy
from ManyBodyQutip.qutip_class import SpinOperator


# ─────────────────────────────────────────────────────────────────────────────
# MWIS instance: bipartite graph with n0, n1 nodes
# Reproduces the 5-qubit avoided-crossing example from the paper
# ─────────────────────────────────────────────────────────────────────────────
def build_mwis_instance(n0: int = 2, n1: int = 3, dW: float = 0.01, Jzz: float = 5.33):
    n = n0 + n1
    G0 = list(range(n0))
    G1 = list(range(n0, n))

    jij = np.zeros((n, n))
    for i in G0:
        for j in G1:
            jij[i, j] = Jzz
            jij[j, i] = Jzz

    h0 = n1 * Jzz - 2 * (1 + dW) / n0
    h1 = n0 * Jzz - 2 / n1

    hz = np.zeros(n)
    for i in G0:
        hz[i] = h0
    for j in G1:
        hz[j] = h1

    return jij, hz


# ─────────────────────────────────────────────────────────────────────────────
# Build non-stoquastic (XX) catalyst term
# Uses SpinOperator for sum of XX couplings across all pairs
# ─────────────────────────────────────────────────────────────────────────────
def build_catalyst_hamiltonian(nqubits: int, Jxx: float = 1.0):
    """
    Build unbiased XX catalyst:  H_cat = Jxx * sum_{i<j} X_i X_j
    Falls back to the src utility if available, otherwise builds manually.
    """
    try:
        cat = get_unbiased_catalyst_term(nqubits=nqubits)
        return cat
    except Exception:
        pass

    # Manual fallback via SpinOperator
    cat = None
    for i in range(nqubits):
        for j in range(i + 1, nqubits):
            term = SpinOperator(
                index=[("x", i, "x", j)],
                coupling=[Jxx],
                size=nqubits,
            ).qutip_op.data_as("csr_matrix")
            cat = term if cat is None else cat + term
    return cat


# ─────────────────────────────────────────────────────────────────────────────
# Fidelity: projection onto GS of target Hamiltonian (in sector)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fidelity(psi, target_hamiltonian_s):
    evals, evecs = eigsh(target_hamiltonian_s.astype(complex), which="SA", k=4)
    order = np.argsort(evals)
    evecs = evecs[:, order]
    return float(np.sum(np.abs(evecs[:, :1].conj().T @ psi) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# F-CRAB warm-start GRAPE  (from the notebook)
# Progressively grows the Fourier basis: dim_schedule = [2,4,8,16,32]
# At each level it keeps the best params from the previous level and
# adds randomised new frequencies (nr restarts per level).
# ─────────────────────────────────────────────────────────────────────────────
def run_fcrab_warm_start(
    psi_init,
    target_hamiltonian_s,
    driver_hamiltonian_s,
    tau,
    time_steps,
    dim_schedule=(2, 4, 8, 16, 32),
    nr: int = 5,
    maxiter: int = 500,
    ftol: float = 1e-9,
    gtol: float = 1e-9,
    verbose: bool = True,
):
    """
    Warm-start F-CRAB optimisation as implemented in the notebook.

    Returns dict with keys:
        h_driver, h_target  : (time_steps,) schedule arrays
        energy              : final energy
        psi                 : final state
        parameters, omegas  : optimised F-CRAB coefficients / frequencies
    """
    best_params = None
    best_omegas = None
    best_energy = np.inf
    best_psi = None
    best_h_driver = None
    best_h_target = None

    for level_idx, dim in enumerate(dim_schedule):
        dim_prev = dim_schedule[level_idx - 1] if level_idx > 0 else 0
        delta = dim - dim_prev

        if verbose:
            print(f"\n{'='*50}")
            print(f"  N_c = {dim}  (adding {delta} new modes, {nr} restarts)")
            print(f"{'='*50}")

        level_best_energy = np.inf
        level_best_params = None
        level_best_omegas = None
        level_best_psi = None
        level_best_h_driver = None
        level_best_h_target = None

        for restart in range(nr):
            rng = np.random.default_rng()

            new_omegas = (
                np.pi
                * np.arange(dim_prev + 1, dim + 1)
                * (1 + rng.uniform(-0.5, 0.5, delta))
                / tau
            )

            if best_omegas is not None:
                full_omegas = np.concatenate([best_omegas, new_omegas])
            else:
                full_omegas = new_omegas

            # warm-start params: [a_drv(dim), a_tgt(dim)]
            init_params = np.zeros(2 * dim)
            if best_params is not None:
                init_params[:dim_prev] = best_params[:dim_prev]
                init_params[dim : dim + dim_prev] = best_params[dim_prev:]

            model = SparseGRAPEModel(
                initial_state=psi_init,
                target_hamiltonian=target_hamiltonian_s,
                initial_hamiltonian=driver_hamiltonian_s,
                reference_hamiltonian=target_hamiltonian_s,
                tf=tau,
                number_of_parameters=dim,
                nsteps=time_steps,
                type="F-CRAB",
                seed=42,
                mode="annealing ansatz",
                random=False,
            )

            model.omegas = full_omegas
            model._sin_basis = np.sin(np.outer(full_omegas, model.time))
            model.parameters = init_params.copy()

            trainer = SparseGRAPETrainer(
                model,
                maxiter=maxiter,
                ftol=ftol,
                gtol=gtol,
                verbose=verbose,
            )
            results = trainer.run()

            if verbose:
                print(f"  restart {restart+1}/{nr}  E = {results['energy']:.6f}")

            if results["energy"] < level_best_energy:
                level_best_energy = results["energy"]
                level_best_params = results["parameters"].copy()
                level_best_omegas = full_omegas.copy()
                level_best_psi = results["psi"].copy()
                level_best_h_driver = results["h_driver"].copy()
                level_best_h_target = results["h_target"].copy()

        if level_best_energy < best_energy:
            best_energy = level_best_energy
            best_params = level_best_params
            best_omegas = level_best_omegas
            best_psi = level_best_psi
            best_h_driver = level_best_h_driver
            best_h_target = level_best_h_target

        if verbose:
            print(f"\n  → best energy at N_c={dim}: {best_energy:.6f}")

    return {
        "h_driver": best_h_driver,
        "h_target": best_h_target,
        "energy": best_energy,
        "psi": best_psi,
        "parameters": best_params,
        "omegas": best_omegas,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    tau: float,
    n_params: int,
    n0: int = 2,
    n1: int = 3,
    dW: float = 0.01,
    Jzz: float = 5.33,
    Jxx_catalyst: float = 1.0,
    nlevels: int = 10,
    maxiter: int = 500,
    grape_dim_schedule: tuple = (2, 4, 8, 16, 32),
    grape_nr: int = 5,
    inner_steps_per_tau: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run all three protocols for one (tau, n_params) point.

    Returns a dict with full time-series data for magic, entanglement,
    probabilities, energies, and schedules.
    """
    t0 = _time.time()
    nqubits = n0 + n1

    # ── time grid ─────────────────────────────────────────────────────────────
    time_steps = int(inner_steps_per_tau * tau)
    time = np.linspace(0, tau, time_steps)
    delta_t = time[1] - time[0]

    # ── build Hamiltonians ────────────────────────────────────────────────────
    jij, hz = build_mwis_instance(n0, n1, dW, Jzz)
    dim = 2**nqubits
    dim_s = dim

    target_hamiltonian = get_longitudinal_hamiltonian(jij, hz)
    driver_hamiltonian = get_driver_hamiltonian(nqubits=nqubits)
    catalyst_hamiltonian = build_catalyst_hamiltonian(nqubits, Jxx=Jxx_catalyst)

    target_hamiltonian_s = target_hamiltonian
    driver_hamiltonian_s = driver_hamiltonian
    catalyst_hamiltonian_s = catalyst_hamiltonian

    # ── initial state (ground state of Hd projected to sector) ───────────────
    _, eigvecs_init = eigsh(driver_hamiltonian_s.astype(complex), which="SA", k=2)
    psi_init_s = eigvecs_init[:, 0].astype(complex)

    # ── SRE + EE tools ────────────────────────────────────────────────────────
    sre = SREJax(n_qubits=nqubits, batch_size=4096)
    ee = EntanglementEntropy(nqubits=nqubits, n_A=nqubits // 2)

    def analyse_history(psi_history, eigstates_ref):
        """M2(t), S_A(t), p_k(t) against the linear-H eigenstates reference."""
        m2 = np.zeros(time_steps)
        s_a = np.zeros(time_steps)
        probs = np.zeros((time_steps, nlevels))
        for i in range(time_steps):
            psi_full = psi_history[i]
            m2[i] = sre(psi_full)
            s_a[i] = ee.von_neumann(psi_full)
            probs[i] = np.abs(eigstates_ref[i].conj().T @ psi_history[i]) ** 2
        return m2, s_a, probs

    # ─────────────────────────────────────────────────────────────────────────
    # Adiabatic reference: instantaneous GS of linear H(t)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [τ={tau} Np={n_params}] Adiabatic reference...")
    magic_gs = np.zeros(time_steps)
    ee_gs = np.zeros(time_steps)
    spectrum = np.zeros((time_steps, nlevels))
    eigstates_ref = np.zeros((time_steps, dim_s, nlevels), dtype=complex)

    for i, t in enumerate(time):
        H_t = (1 - t / tau) * driver_hamiltonian_s + (t / tau) * target_hamiltonian_s
        evals, evecs = eigsh(H_t.astype(complex), which="SA", k=nlevels)
        order = np.argsort(evals)
        spectrum[i] = evals[order]
        evecs = evecs[:, order].astype(complex)
        eigstates_ref[i] = evecs
        psi_gs_i = evecs[:, 0]
        magic_gs[i] = sre((psi_gs_i))
        ee_gs[i] = ee.von_neumann((psi_gs_i))

    min_gap = float(np.min(spectrum[:, 1] - spectrum[:, 0]))
    t_min_gap = float(time[np.argmin(spectrum[:, 1] - spectrum[:, 0])])
    s_min_gap = t_min_gap / tau

    print(f"  Minimum gap: {min_gap:.6f}  at s = {s_min_gap:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Protocol 1: LINEAR
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [τ={tau} Np={n_params}] Linear evolution...")
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
    # Protocol 2: CATALYST  (non-stoquastic XX driver)
    # H(t) = (1-s) Hd + s Hp + s(1-s) Hcat
    # Probabilities and energy are measured on the *linear* Hamiltonian
    # for a fair comparison with the linear protocol.
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [τ={tau} Np={n_params}] Catalyst evolution...")
    psi_cat = psi_init_s.copy()
    psi_history_cat = np.zeros((time_steps, dim_s), dtype=complex)
    energy_cat = np.zeros(time_steps)

    for i, t in enumerate(time):
        s = t / tau
        H_cat_t = (
            (1 - s) * driver_hamiltonian_s
            + s * target_hamiltonian_s
            + s * (1 - s) * catalyst_hamiltonian_s
        )
        psi_cat = expm_multiply(-1j * delta_t * H_cat_t, psi_cat)
        psi_history_cat[i] = psi_cat
        # energy on the linear Hamiltonian for fair comparison
        H_lin_t = (1 - s) * driver_hamiltonian_s + s * target_hamiltonian_s
        energy_cat[i] = (psi_cat.conj() @ H_lin_t @ psi_cat).real

    m2_cat, ee_cat, probs_cat = analyse_history(psi_history_cat, eigstates_ref)
    fidelity_cat = compute_fidelity(psi_cat, target_hamiltonian_s)

    # ─────────────────────────────────────────────────────────────────────────
    # Protocol 3: OPTIMAL CONTROL  (F-CRAB warm-start GRAPE)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [τ={tau} Np={n_params}] Optimal control (warm-start GRAPE)...")
    grape_results = run_fcrab_warm_start(
        psi_init=psi_init_s,
        target_hamiltonian_s=target_hamiltonian_s,
        driver_hamiltonian_s=driver_hamiltonian_s,
        tau=tau,
        time_steps=time_steps,
        dim_schedule=tuple(d for d in grape_dim_schedule if d <= n_params)
        or (n_params,),
        nr=grape_nr,
        maxiter=maxiter,
        ftol=1e-9,
        gtol=1e-9,
        verbose=verbose,
    )

    h_driver_opt = grape_results["h_driver"]
    h_target_opt = grape_results["h_target"]

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
        # energy on the linear Hamiltonian for fair comparison
        H_lin_t = (1 - t / tau) * driver_hamiltonian_s + (
            t / tau
        ) * target_hamiltonian_s
        energy_opt[i] = (psi_opt.conj() @ H_lin_t @ psi_opt).real

    m2_opt, ee_opt, probs_opt = analyse_history(psi_history_opt, eigstates_ref)
    fidelity_opt = compute_fidelity(psi_opt, target_hamiltonian_s)

    elapsed = _time.time() - t0
    print(
        f"  Done in {elapsed:.1f}s  "
        f"F_lin={fidelity_lin:.4f}  "
        f"F_cat={fidelity_cat:.4f}  "
        f"F_opt={fidelity_opt:.4f}  "
        f"gap={min_gap:.5f}"
    )

    return {
        # ── system info ───────────────────────────────────────────────────────
        "nqubits": nqubits,
        "tau": tau,
        "n_params": n_params,
        "time": time,
        "spectrum": spectrum,
        "min_gap": min_gap,
        "t_min_gap": t_min_gap,
        "s_min_gap": s_min_gap,
        # ── adiabatic reference ───────────────────────────────────────────────
        "magic_gs": magic_gs,
        "ee_gs": ee_gs,
        # ── linear protocol ───────────────────────────────────────────────────
        "m2_lin": m2_lin,
        "ee_lin": ee_lin,
        "probs_lin": probs_lin,
        "energy_lin": energy_lin,
        "fidelity_lin": fidelity_lin,
        "int_magic_lin": float(np.trapz(m2_lin, time)),
        "int_ee_lin": float(np.trapz(ee_lin, time)),
        # ── catalyst protocol ─────────────────────────────────────────────────
        "m2_cat": m2_cat,
        "ee_cat": ee_cat,
        "probs_cat": probs_cat,
        "energy_cat": energy_cat,
        "fidelity_cat": fidelity_cat,
        "int_magic_cat": float(np.trapz(m2_cat, time)),
        "int_ee_cat": float(np.trapz(ee_cat, time)),
        # ── optimal control protocol ──────────────────────────────────────────
        "m2_opt": m2_opt,
        "ee_opt": ee_opt,
        "probs_opt": probs_opt,
        "energy_opt": energy_opt,
        "fidelity_opt": fidelity_opt,
        "int_magic_opt": float(np.trapz(m2_opt, time)),
        "int_ee_opt": float(np.trapz(ee_opt, time)),
        "h_driver_opt": h_driver_opt,
        "h_target_opt": h_target_opt,
        "grape_energy": grape_results["energy"],
        "grape_omegas": grape_results["omegas"],
        # ── integrated magic of the adiabatic reference ───────────────────────
        "int_magic_gs": float(np.trapz(magic_gs, time)),
        "int_ee_gs": float(np.trapz(ee_gs, time)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Study wrappers
# ─────────────────────────────────────────────────────────────────────────────
def study_vs_tau(n_params, taus, output_dir, **kwargs):
    print(f"\n══ Study vs tau  (n_params={n_params}) ══")
    results = []
    for tau in taus:
        print(f"\n── tau={tau} ──")
        results.append(run_experiment(tau=tau, n_params=n_params, **kwargs))
    path = Path(output_dir) / f"avoided_crossing_vs_tau_np{n_params}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"sweep": "tau", "taus": taus, "results": results}, f)
    print(f"\nSaved → {path}")
    _print_summary(results, sweep_key="tau")
    return results


def study_vs_params(tau, n_params_list, output_dir, **kwargs):
    print(f"\n══ Study vs n_params  (tau={tau}) ══")
    results = []
    for n_params in n_params_list:
        print(f"\n── n_params={n_params} ──")
        results.append(run_experiment(tau=tau, n_params=n_params, **kwargs))
    path = Path(output_dir) / f"avoided_crossing_vs_params_tau{tau:.1f}.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {"sweep": "n_params", "n_params_list": n_params_list, "results": results}, f
        )
    print(f"\nSaved → {path}")
    _print_summary(results, sweep_key="n_params")
    return results


def study_single(tau, n_params, output_dir, **kwargs):
    print(f"\n══ Single run  (tau={tau}, n_params={n_params}) ══")
    r = run_experiment(tau=tau, n_params=n_params, **kwargs)
    path = Path(output_dir) / f"avoided_crossing_single_tau{tau:.1f}_np{n_params}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"sweep": "single", "result": r}, f)
    print(f"\nSaved → {path}")
    _print_summary([r], sweep_key="tau")
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────
def _print_summary(results, sweep_key):
    print(f'\n{"─"*100}')
    print(
        f'{"param":>8}  {"gap":>8}  '
        f'{"F_lin":>7}  {"F_cat":>7}  {"F_opt":>7}  '
        f'{"∫M2_lin":>9}  {"∫M2_cat":>9}  {"∫M2_opt":>9}  {"∫M2_gs":>9}'
    )
    print(f'{"─"*100}')
    for r in results:
        print(
            f"{r[sweep_key]:>8}  "
            f'{r["min_gap"]:>8.5f}  '
            f'{r["fidelity_lin"]:>7.4f}  '
            f'{r["fidelity_cat"]:>7.4f}  '
            f'{r["fidelity_opt"]:>7.4f}  '
            f'{r["int_magic_lin"]:>9.4f}  '
            f'{r["int_magic_cat"]:>9.4f}  '
            f'{r["int_magic_opt"]:>9.4f}  '
            f'{r["int_magic_gs"]:>9.4f}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Avoided-crossing study: linear vs catalyst vs optimal control"
    )
    parser.add_argument(
        "--study",
        choices=["tau", "params", "single"],
        required=True,
        help="Which sweep to run",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=10.0,
        help="Annealing time (used when --study=params or single)",
    )
    parser.add_argument(
        "--n_params",
        type=int,
        default=5,
        help="Fourier modes for OC / GRAPE (used when --study=tau or single)",
    )
    parser.add_argument(
        "--output", type=str, default="results/", help="Output directory"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Max GRAPE optimiser iterations per restart",
    )
    parser.add_argument(
        "--grape_nr",
        type=int,
        default=5,
        help="Number of random restarts per GRAPE level",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=100,
        help="Time steps per unit tau (inner_steps_per_tau)",
    )
    parser.add_argument("--Jzz", type=float, default=5.33)
    parser.add_argument("--dW", type=float, default=0.01)
    parser.add_argument(
        "--Jxx", type=float, default=1.0, help="Catalyst XX coupling strength"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    shared_kwargs = dict(
        Jzz=args.Jzz,
        dW=args.dW,
        Jxx_catalyst=args.Jxx,
        maxiter=args.maxiter,
        grape_nr=args.grape_nr,
        inner_steps_per_tau=args.time_steps,
        verbose=args.verbose,
    )

    if args.study == "tau":
        taus = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        study_vs_tau(
            n_params=args.n_params,
            taus=taus,
            output_dir=args.output,
            **shared_kwargs,
        )

    elif args.study == "params":
        n_params_list = [2, 4, 8, 16, 32]
        study_vs_params(
            tau=args.tau,
            n_params_list=n_params_list,
            output_dir=args.output,
            **shared_kwargs,
        )

    elif args.study == "single":
        study_single(
            tau=args.tau,
            n_params=args.n_params,
            output_dir=args.output,
            **shared_kwargs,
        )


if __name__ == "__main__":
    main()

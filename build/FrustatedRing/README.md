# build/FrustatedRing

Data generation for the frustrated Ising ring (J, JL, JR = 1, 0.5, 0.45).
Run from this folder; every script writes to `../../generated/FrustatedRing/`.

## Pipeline

1. `run_*.py T N`: one job per T (cluster array), writes
   `QuantumResourcesvsT_N={N}_T={T}_{LZR{tag} | linear}.npz`.
2. `merge_*.py`: joins the per-T files into one `..._N={N}_...npz` (keys
   prefixed `T={T}_`). Range from the env vars `T_MIN`, `T_MAX`, `STEP`.
3. Optional post-processing of the merged file: `generate_more_spectrum.py`,
   `run_fidelity_robustness.py`.

Common to every LZR run: LZS ansatz with M = 2 plateaus (7 parameters), best
of several random initial points (`src/parallel.py`), then one evolution of
the best schedule saving schedule, energy, e0/e1/gap, p0/p1, and magic and
entanglement every 10 steps.

## LZR, spin basis (Z2 +1 sector, `SparseGRAPEModel`)

| Script | tag | Optimizer | Bounds | Steps / unit T | Initial points |
|---|---|---|---|---|---|
| `run_LZR.py` | (none) | GRAPE | no | 10 | best of 50 |
| `run_LZR_bounded.py` | `_bounded` | GRAPE | yes | 100 | best of 50 |
| `run_LZR_bounded_more_T.py` | `_bounded_more_T` | GRAPE | yes | 10 (cheaper, for a larger T range) | best of 50 |
| `run_LZR_SA.py` | `_SA` | simulated annealing | no | 50 | best of 5 |
| `run_LZR_bounded_SA.py` | `_bounded_SA` | simulated annealing | yes | 50 | best of 5 |
| `run_LZR_NoGrad.py` | `_NoGrad` | COBYLA, gradient-free (`SchedulerTrainer`) | no | 100 | seed 6 only |
| `run_LZR_no_random.py` | `_no_random` | GRAPE | no | 100 | deterministic (`random=False`) |

Obsolete tests:

- `run_LZR_step_test.py`: `run_LZR_no_random.py` from the random initial
  point of seed 6.
- `run_LZR_bounded_magic_test.py`: `run_LZR_bounded_more_T.py` with the magic
  of the lifted 2^N state instead of the sector state.
- `run_LZR_bounded_SA_test.py`: T = 80 and N = 7 hard-coded, one SA run
  (seed 0), no parallelism. Despite the name, `bounds_opt` is not set.

## LZR, free fermions (`NambuGRAPEModel`, O(N^3) per step)

All of them: GRAPE, 100 steps per unit T, magic by Majorana sampling
(1000 samples).

| Script | Arguments | tag | What differs |
|---|---|---|---|
| `run_LZR_FF.py` | `T N` | `_FF` | unbounded, best of 200 |
| `run_LZR_FF_bounded.py` | `T N` | `_FF_bounded` | bounded, best of 200 (keep in sync with `run_LZR_FF.py`) |
| `run_LZR_FF_bounded_ws.py` | `T0 N T_MIN T_MAX STEP` | `_FF_bounded_ws_T0={T0}` | best of 200 at T0 only; every other T starts from its neighbour's optimum (warm start). One job does the whole T range |
| `run_LZR_FF_bounded_ws_fixed_seed.py` | `T0 N T_MIN T_MAX STEP SEED` | `_FF_bounded_ws_fixed_seed_T0={T0}_seed={SEED}` | same, but T0 is a single run from SEED |
| `run_LZR_seed_scan_FF_bounded.py` | `T N SEED` | `_FF_bounded` | fixed T, one run per seed (0..199 = the initial points of `run_LZR_FF_bounded.py`). Writes `QuantumResourcesSeedScan_..._seed={SEED}_...`; seed 0 also stores s* and the minimum gap |

## Linear schedule (free fermions)

- `run_linear.py T N`: s = t/T, 100 steps per unit T. Also the magic of the
  instantaneous ground state. Reference for the LZR plots.
- `run_linear_more_T.py`: identical, only the output name changes
  (`_linear_more_T`), to run a larger T range separately.

## Merge

| Script | Arguments | Joins |
|---|---|---|
| `merge_LZR.py` | `N TAG` | `run_LZR*` outputs (default T = 1..200) |
| `merge_linear.py` / `merge_linear_more_T.py` | `N` | `run_linear*` outputs (default T = 5..400, step 5) |
| `merge_LZR_seed_scan.py` | `N T TAG` | seed scan (env `SEED_MIN`, `SEED_MAX`, default 0..199) |
| `merge_fidelity_robustness.py` | `N [TAG]` | `run_fidelity_robustness.py` outputs |

## Other

- `parameter_generation_LZR.py T N`: spin basis, bounded GRAPE from seeds
  0..9, saves all 10 optima to `ParametersLZR_T={T}_N={N}.npz`. Input of the
  `graphs/FrustatedRing/LZR*Landscape.py` scripts.
- `run_fidelity_robustness.py T N [TAG]`: reads the merged
  `QuantumResourcesvsT_N={N}_LZR{TAG}.npz`. Full 2^N space: robustness
  matrices (second order) and exact fidelity under global fields, for the
  LZR schedule and the linear one at the same T.
- `generate_more_spectrum.py N TAG NLEVELS`: recomputes the spectrum of a
  merged LZR file with NLEVELS levels and **overwrites it in place** (also
  moves old files to the midpoint grid).

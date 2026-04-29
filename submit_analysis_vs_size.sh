#!/bin/bash
#SBATCH --job-name=ising_vs_size
#SBATCH --output=logs/ising_vs_size_%j.out
#SBATCH --error=logs/ising_vs_size_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # JAX + scipy eigsh use threading
#SBATCH --time=18:00:00            # conservative for tau=10, maxiter=500
#SBATCH --account=bsc21
#SBATCH --qos=gp_bsccase
#SBATCH --partition=gpp


# tell JAX/XLA to use the allocated CPUs only
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── paths ─────────────────────────────────────────────────────────────────────
WORKDIR=$HOME/Magic4Annealing     # adjust to your repo root
OUTDIR=$WORKDIR/results/vs_size

mkdir -p $OUTDIR logs

cd $WORKDIR

# ── run ───────────────────────────────────────────────────────────────────────
python -u study_1d_ising.py \
    --study size \
    --tau 5.0 \
    --n_params 5 \
    --schedule_type fourier \
    --maxiter 500 \
    --output $OUTDIR
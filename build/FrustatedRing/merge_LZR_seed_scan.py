"""Merge the per-seed outputs of run_LZR_seed_scan_FF_bounded.py. See README.md."""

import glob
import os
import re
import sys

import numpy as np

N = int(sys.argv[1])  # odd; N=9,11,13 feasible for full 2^N exact diagonalization
T = int(sys.argv[2])  # the fixed T of the seed scan
TAG = sys.argv[3]

# --- configuración ---
directorio = "../../generated/FrustatedRing"
patron_archivo = os.path.join(
    directorio, f"QuantumResourcesSeedScan_N={N}_T={T}_seed=*_LZR{TAG}.npz"
)
archivo_salida = os.path.join(
    directorio, f"QuantumResourcesSeedScan_N={N}_T={T}_LZR{TAG}.npz"
)

# SEED_MIN, SEED_MAX se leen de variables de entorno (definidas en submit.sh)
# con valores por defecto por si se ejecuta manualmente sin pasarlas
try:
    SEED_MIN = int(os.environ.get("SEED_MIN", 0))
    SEED_MAX = int(os.environ.get("SEED_MAX", 199))
except ValueError:
    raise RuntimeError(
        "SEED_MIN o SEED_MAX no son enteros válidos. "
        f"Valores recibidos: SEED_MIN={os.environ.get('SEED_MIN')}, "
        f"SEED_MAX={os.environ.get('SEED_MAX')}"
    )

print(f"Rango esperado: SEED_MIN={SEED_MIN}, SEED_MAX={SEED_MAX}, T={T}, TAG='{TAG}'")

# regex para extraer la seed del nombre de archivo
patron_regex = re.compile(
    rf"QuantumResourcesSeedScan_N={N}_T={T}_seed=(\d+)_LZR{re.escape(TAG)}\.npz$"
)

archivos = sorted(
    (f for f in glob.glob(patron_archivo) if patron_regex.search(f)),
    key=lambda f: int(patron_regex.search(f).group(1)),
)

if not archivos:
    raise RuntimeError(f"No se encontraron archivos en {patron_archivo}")

seeds_esperadas = set(range(SEED_MIN, SEED_MAX + 1))
seeds_encontradas = set()

combinado = {}

for archivo in archivos:
    seed = int(patron_regex.search(archivo).group(1))
    seeds_encontradas.add(seed)
    d = np.load(archivo)

    for key in d.files:
        combinado[f"seed={seed}_{key}"] = d[key]

faltantes = seeds_esperadas - seeds_encontradas
if faltantes:
    raise RuntimeError(
        f"Faltan {len(faltantes)} seeds: {sorted(faltantes)}. "
        f"No se borrarán los archivos individuales."
    )

extra = seeds_encontradas - seeds_esperadas
if extra:
    print(f"Aviso: se encontraron seeds fuera del rango esperado: {sorted(extra)}")

# s* (avoided crossing) is only computed in the job of seed 0
if "seed=0_s_star" not in combinado:
    print(
        "Aviso: falta seed=0_s_star (s* solo se calcula en el job de la seed 0); "
        "el script de gráficas lo necesita."
    )


np.savez(archivo_salida, **combinado)
print(f"Guardado: {archivo_salida}")
print(f"Total de claves: {len(combinado)}")

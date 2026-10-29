"""
Write step shared by the merge scripts of build/ (merge_LZR.py,
merge_linear.py, merge_fidelity_robustness.py).

A merged file holds the arrays of every T with the keys prefixed "T={T}_".
With append=True the new T are added to the merged file that already exists
instead of replacing it, e.g. to extend the T range of an earlier run. The
write goes through a temporary file, so a job that dies halfway leaves the
previous merged file intact.
"""

import os
import re

import numpy as np

_T_PREFIX = re.compile(r"^T=(\d+)_")


def _T_values(keys):
    return {int(_T_PREFIX.match(key).group(1)) for key in keys}


def _per_T_keys(keys):
    return {_T_PREFIX.sub("", key) for key in keys}


def save_merged(archivo_salida, combinado, append=False):
    """Save `combinado` ({"T={T}_{key}": array}) to `archivo_salida`.

    append=True: `archivo_salida` must exist and none of the new T may be in
    it (RuntimeError otherwise, so the per-T files are not deleted)."""
    if append:
        if not os.path.exists(archivo_salida):
            raise RuntimeError(
                f"APPEND=1 pero no existe {archivo_salida}. "
                f"No se borrarán los archivos individuales."
            )

        with np.load(archivo_salida) as d:
            previo = {key: d[key] for key in d.files}

        T_previos = _T_values(previo)
        T_nuevos = _T_values(combinado)
        repetidos = T_previos & T_nuevos
        if repetidos:
            raise RuntimeError(
                f"{len(repetidos)} valores de T ya están en {archivo_salida}: "
                f"{sorted(repetidos)}. No se borrarán los archivos individuales."
            )

        # same keys per T in the old and the new data (otherwise the merged
        # file is left heterogeneous, e.g. after generate_more_spectrum.py)
        claves_previas = _per_T_keys(previo)
        claves_nuevas = _per_T_keys(combinado)
        if claves_previas != claves_nuevas:
            print(
                "Aviso: las claves por T difieren. "
                f"Solo en el archivo previo: {sorted(claves_previas - claves_nuevas)}; "
                f"solo en los nuevos: {sorted(claves_nuevas - claves_previas)}"
            )

        print(
            f"Añadiendo {len(T_nuevos)} T's a los {len(T_previos)} de {archivo_salida}"
        )
        combinado = {**previo, **combinado}

    archivo_tmp = archivo_salida + ".tmp"
    with open(archivo_tmp, "wb") as f:  # file object: np.savez adds no suffix
        np.savez(f, **combinado)
    os.replace(archivo_tmp, archivo_salida)
    print(f"Guardado: {archivo_salida}")
    print(f"Total de claves: {len(combinado)}")

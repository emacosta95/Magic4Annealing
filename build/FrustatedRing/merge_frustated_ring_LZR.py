import numpy as np
import glob
import re
import os

# --- configuración ---
directorio = "../../generated/FrustatedRing"
patron_archivo = os.path.join(directorio, "QuantumResourcesvsT_T=*_LZR.npz")
archivo_salida = os.path.join(directorio, "QuantumResourcesvsT_LZR.npz")

# T_MIN, T_MAX, STEP se leen de variables de entorno (definidas en submit.sh)
# con valores por defecto por si se ejecuta manualmente sin pasarlas
try:
    T_MIN = int(os.environ.get("T_MIN", 1))
    T_MAX = int(os.environ.get("T_MAX", 200))
    STEP = int(os.environ.get("STEP", 1))
except ValueError:
    raise RuntimeError(
        "T_MIN, T_MAX o STEP no son enteros válidos. "
        f"Valores recibidos: T_MIN={os.environ.get('T_MIN')}, "
        f"T_MAX={os.environ.get('T_MAX')}, STEP={os.environ.get('STEP')}"
    )

print(f"Rango esperado: T_MIN={T_MIN}, T_MAX={T_MAX}, STEP={STEP}")

# regex para extraer el T_str del nombre de archivo
patron_regex = re.compile(r"QuantumResourcesvsT_T=([\d.]+)\_LZR.npz$")

archivos = sorted(
    glob.glob(patron_archivo), key=lambda f: int(patron_regex.search(f).group(1))
)

if not archivos:
    raise RuntimeError(f"No se encontraron archivos en {patron_archivo}")

T_esperados = set(range(T_MIN, T_MAX + 1, STEP))
T_encontrados = set()

combinado = {}

for archivo in archivos:
    match = patron_regex.search(archivo)
    if not match:
        print(f"Aviso: no se pudo extraer T de '{archivo}', se omite.")
        continue

    T_str = match.group(1)
    T_encontrados.add(int(T_str))
    d = np.load(archivo)

    for key in d.files:
        combinado[f"T={T_str}_{key}"] = d[key]

faltantes = T_esperados - T_encontrados
if faltantes:
    raise RuntimeError(
        f"Faltan {len(faltantes)} valores de T: {sorted(faltantes)}. "
        f"No se borrarán los archivos individuales."
    )

extra = T_encontrados - T_esperados
if extra:
    print(f"Aviso: se encontraron T's fuera del rango esperado: {sorted(extra)}")


np.savez(archivo_salida, **combinado)
print(f"Guardado: {archivo_salida}")
print(f"Total de claves: {len(combinado)}")

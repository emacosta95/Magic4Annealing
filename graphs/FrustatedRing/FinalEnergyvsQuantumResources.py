import os
import re

import matplotlib.pyplot as plt
import numpy as np

tag = "_FF_bounded"  # "_FF", "_bounded", "_bounded_SA" or ""
tag_T = ""  # "_more_T" or ""
N = 7
# "magic" or "magic_filtered" (the latter only in the FF files)
magic_key = "magic"

data = np.load(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR"
    + tag
    + tag_T
    + ".npz"
)

# only the keys that are plotted are read from the file
patron = re.compile(r"^T=(\d+)_evo_energy$")
Tlist = sorted(int(m.group(1)) for m in map(patron.match, data.files) if m)

energy_error = []
max_magic = []
max_entanglement = []
T_open = []
for T in Tlist:
    # error with respect to the instantaneous ground state at the final time
    energy_error.append(data[f"T={T}_evo_energy"][-1] - data[f"T={T}_e0"][-1])
    max_magic.append(np.max(data[f"T={T}_{magic_key}"]))
    max_entanglement.append(np.max(data[f"T={T}_entanglement"]))
    if not np.isclose(data[f"T={T}_schedule"][-1], 1.0):
        T_open.append(T)

if T_open:
    print(
        f"Aviso: el schedule no termina en 1 para T = {T_open}; ahí e0[-1] no es "
        "la energía fundamental del Hamiltoniano problema."
    )

path = "../../images/FrustatedRing/FinalEnergyvsQuantumResources/"
if not os.path.exists(path):
    os.makedirs(path)
name_tag = (tag + tag_T).lstrip("_")
suffix = (name_tag + "_" if name_tag else "") + f"N={N}.png"

resources = {
    "MaxMagic": (max_magic, "Maximum Magic"),
    "MaxEntanglement": (max_entanglement, "Maximum Entanglement"),
}
for name, (resource, xlabel) in resources.items():
    plt.figure(figsize=(7, 5))
    plt.scatter(
        resource,
        energy_error,
        c=Tlist,
        cmap="viridis",
        s=22,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    plt.colorbar(label="T")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Error in Energy at final time")
    plt.title(f"N={N}, LZR{tag + tag_T}")
    plt.grid()

    filename = f"{path}{name}_{suffix}"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Guardado: {filename}")

plt.show()

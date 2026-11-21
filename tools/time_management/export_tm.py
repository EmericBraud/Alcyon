"""Exporte le MLP de gestion du temps (tm_mlp.npz) en en-tete C++.

Usage : python export_tm.py tm_mlp.npz src/engine/search/tm_mlp_weights.hpp
"""

import sys

import numpy as np


def arr(name, a):
    vals = ",".join(f"{x:.8e}f" for x in np.asarray(a).ravel())
    return f"inline constexpr float {name}[{np.asarray(a).size}] = {{{vals}}};"


def main():
    z = np.load(sys.argv[1])
    n = len([k for k in z.files if k.startswith("W")])
    out = [
        "// Genere par tools/time_management/export_tm.py : P(le meilleur coup",
        "// change a l'iteration suivante). Ne pas editer a la main.",
        "#pragma once",
        "namespace tm_mlp",
        "{",
        f"inline constexpr int kIn = {z['W0'].shape[0]};",
        f"inline constexpr int kLayers = {n};",
        "inline constexpr int kSizes[] = {" + ",".join(str(z[f'W{i}'].shape[1]) for i in range(n)) + "};",
        arr("kMean", z["mean"]),
        arr("kScale", z["scale"]),
    ]
    for i in range(n):
        # W stocke [entree][sortie], comme sklearn.
        out.append(arr(f"kW{i}", z[f"W{i}"]))
        out.append(arr(f"kB{i}", z[f"b{i}"]))
    out.append("}")
    with open(sys.argv[2], "w") as f:
        f.write("\n".join(out) + "\n")
    print([z[f"W{i}"].shape for i in range(n)])


if __name__ == "__main__":
    main()

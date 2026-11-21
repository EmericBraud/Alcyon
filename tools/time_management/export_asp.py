"""Exporte le modele de fenetre d'aspiration (asp_l0.npz) en en-tete C++.

Usage : python export_asp.py asp_l0.npz src/engine/search/asp_mlp_weights.hpp
"""

import sys

import numpy as np


def arr(name, a):
    a = np.asarray(a, dtype=np.float32).ravel()
    return f"inline constexpr float {name}[{a.size}] = {{{','.join(f'{x:.8e}f' for x in a)}}};"


def main():
    z = np.load(sys.argv[1])
    assert z["l0.weight"].shape == (16, 1024)
    n_scal = z["mean"].shape[0]
    assert z["h.weight"].shape == (32, 16 + n_scal)
    out = [
        "// Genere par tools/time_management/export_asp.py : log(1 + |variation du",
        "// score a l'iteration suivante|). Ne pas editer a la main.",
        "#pragma once",
        "namespace asp_mlp",
        "{",
        f"inline constexpr int kScal = {n_scal};",
        arr("kMean", z["mean"]),
        arr("kScale", z["scale"]),
        arr("kL0w", z["l0.weight"]),  # [16][1024]
        arr("kL0b", z["l0.bias"]),
        arr("kHw", z["h.weight"]),    # [32][16 + kScal]
        arr("kHb", z["h.bias"]),
        arr("kOw", z["o.weight"]),    # [1][32]
        arr("kOb", z["o.bias"]),
        "}",
    ]
    with open(sys.argv[2], "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

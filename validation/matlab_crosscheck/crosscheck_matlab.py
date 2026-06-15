#!/usr/bin/env python3
"""
Cross-check the package nu_gap against MATLAB's gapmetric (Robust Control
Toolbox) -- the canonical reference implementation of the Vinnicombe nu-gap.

Two steps:
  1. In MATLAB (this folder):  >> reference_values
     This reads panel.json and writes reference_values.csv (id, gap, nugap),
     where `nugap` is the Vinnicombe nu-gap (the SECOND output of gapmetric).
  2. In Python (this folder):  python crosscheck_matlab.py
     Rebuilds the SAME systems from panel.json, computes nu_gap, and asserts
     agreement with the MATLAB `nugap` column.

panel.json is shared by both sides, so the two implementations are guaranteed
to be comparing identical systems (no transcription drift). If
reference_values.csv is not present yet, this script just prints the package
values and the command to generate the MATLAB side.

MATLAB's gapmetric returns an upper bound accurate to a relative tolerance
(default 1e-3); the comparison tolerance below allows a little slack.
"""
import csv as _csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from nugap import tf, nu_gap   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ATOL = 5e-3   # gapmetric default relative accuracy is 1e-3; allow a little slack


def build_pair(c):
    def b(num, den):
        return tf(num, den, dt=c["dt"]) if c["dt"] is not None else tf(num, den)
    return b(c["num1"], c["den1"]), b(c["num2"], c["den2"])


def main():
    panel = json.load(open(os.path.join(HERE, "panel.json")))
    pkg = {c["id"]: float(nu_gap(*build_pair(c))) for c in panel}
    csv_path = os.path.join(HERE, "reference_values.csv")

    if not os.path.exists(csv_path):
        print("reference_values.csv not found -- showing package values only.")
        print("Generate the MATLAB reference first, in this folder:")
        print("    matlab -batch reference_values        (or run reference_values.m)\n")
        print(f"{'id':26s}{'pkg nu_gap':>12s}  description")
        for c in panel:
            print(f"{c['id']:26s}{pkg[c['id']]:12.6f}  {c['desc']}")
        return 0

    ref = {}
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            ref[r["id"].strip().strip('"')] = float(r["nugap"])

    print(f"{'id':26s}{'pkg':>10s}{'MATLAB':>10s}{'|diff|':>10s}  status")
    worst, fails = 0.0, 0
    for c in panel:
        i = c["id"]
        p = pkg[i]
        m = ref.get(i)
        if m is None:
            print(f"{i:26s}{p:10.6f}{'(missing)':>20s}")
            continue
        d = abs(p - m)
        worst = max(worst, d)
        ok = d <= ATOL
        fails += (not ok)
        print(f"{i:26s}{p:10.6f}{m:10.6f}{d:10.2e}  {'ok' if ok else 'MISMATCH'}")
    print(f"\nmax |pkg - MATLAB nugap| = {worst:.2e}   (tolerance {ATOL:g})")
    print("ALL MATCH" if fails == 0 else f"{fails} MISMATCH(es)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

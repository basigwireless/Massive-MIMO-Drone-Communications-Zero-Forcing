"""
Reproduce the figures of ``SPAWC2018_ZF.m`` in Python.

Usage
-----
    python run_figures.py                 # full run (200 drops x 300 noise, as in MATLAB)
    python run_figures.py --quick         # fast preview (50 drops x 100 noise)
    python run_figures.py --drops 100 --noise 200 --seed 1 --out ../figures
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mimo_drone_zf import C, calc_spawc2018

F = 2.5e9               # carrier (Hz)
LAM = C / F             # wavelength (m)
DEL = 0.5 * LAM         # half-wavelength spacing
R = 500.0               # cell radius (m)
K = 10                  # number of drones


def db2lin(x):
    return 10 ** (np.asarray(x) / 10)


def pilot_power(rhop):
    """Pilot SNR referenced to the cell edge -> transmit pilot power."""
    return rhop * (4 * np.pi * R / LAM) ** 2


def sweep(params, **kw):
    return [calc_spawc2018(**p, **kw) for p in params]


def plot_bounds(ax, x, res, xlabel, title):
    ax.plot(x, [r.rate_ub_exact for r in res], "r-o", ms=4, label="Upper bound (exact SINR)")
    ax.plot(x, [r.rate_ub_approx for r in res], "r--", label="Upper bound (approx. SINR)")
    ax.plot(x, [r.rate_zf for r in res], "m-s", ms=4, label="Simulation (ZF)")
    ax.plot(x, [r.rate_lb_exact for r in res], "b-^", ms=4, label="Lower bound (exact SINR)")
    ax.plot(x, [r.rate_lb_approx for r in res], "b--", label="Lower bound (approx. SINR)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ergodic rate (bits/s/Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", type=int, default=200)
    ap.add_argument("--noise", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=2018)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "figures"))
    a = ap.parse_args()
    if a.quick:
        a.drops, a.noise = 50, 100
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    kw = dict(K=K, delta=DEL, lam=LAM, n_drops=a.drops, n_noise=a.noise, radius=R, rng=rng)

    def save(fig, name):
        path = os.path.join(a.out, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved {path}")

    t0 = time.time()

    # ---------------- Figure 1a: rate vs pilot SNR, M = 100 ----------------
    print("Figure 1a ...")
    rhop_db = np.arange(-10, 31, 3)
    res = sweep([dict(M=100, pu_p=pilot_power(r), rhou=db2lin(10)) for r in db2lin(rhop_db)], **kw)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plot_bounds(ax, rhop_db, res, r"Pilot SNR $\rho_p$ (dB)",
                rf"ZF, M = 100, K = {K}, $\rho_u$ = 10 dB")
    save(fig, "fig1a_rate_vs_pilot_snr_M100.png")

    # ---------------- Figure 1b: rate vs pilot SNR, M = 500 ----------------
    print("Figure 1b ...")
    rhop_db = np.arange(-30, 31, 3)
    res = sweep([dict(M=500, pu_p=pilot_power(r), rhou=db2lin(10)) for r in db2lin(rhop_db)], **kw)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plot_bounds(ax, rhop_db, res, r"Pilot SNR $\rho_p$ (dB)",
                rf"ZF, M = 500, K = {K}, $\rho_u$ = 10 dB")
    save(fig, "fig1b_rate_vs_pilot_snr_M500.png")

    # ---------------- Figure 2: rate vs number of GS antennas --------------
    print("Figure 2 ...")
    M_arr = np.arange(50, 501, 50)
    res = sweep([dict(M=int(M), pu_p=pilot_power(db2lin(30)), rhou=10.0) for M in M_arr], **kw)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plot_bounds(ax, M_arr, res, "Number of GS antennas M",
                rf"ZF, K = {K}, $\rho_p$ = 30 dB, $\rho_u$ = 10 dB")
    save(fig, "fig2_rate_vs_antennas.png")

    # ---------------- Figure 3: ZF vs MRC, rate vs data SNR ----------------
    print("Figure 3 ...")
    rhou_db = np.arange(-30, 31, 3)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for rp_db, ls in [(30, "-"), (-10, "--")]:
        res = sweep([dict(M=100, pu_p=pilot_power(db2lin(rp_db)), rhou=r) for r in db2lin(rhou_db)], **kw)
        ax.plot(rhou_db, [r.rate_zf for r in res], "b" + ls, marker="o", ms=4,
                label=rf"ZF ($\rho_p$ = {rp_db} dB)")
        ax.plot(rhou_db, [r.rate_mrc for r in res], "r" + ls, marker="s", ms=4,
                label=rf"MRC ($\rho_p$ = {rp_db} dB)")
    ax.set_xlabel(r"Data SNR $\rho_u$ (dB)")
    ax.set_ylabel("Ergodic rate (bits/s/Hz)")
    ax.set_title(f"ZF vs MRC, M = 100, K = {K}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    save(fig, "fig3_zf_vs_mrc.png")

    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

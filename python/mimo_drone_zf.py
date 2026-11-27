"""
Uplink massive MIMO drone communications with Zero-Forcing (ZF) and
Maximum-Ratio Combining (MRC) receivers under imperfect CSI.

Python / NumPy port of ``Calc_SPAWC2018`` (MATLAB) by Prabhu Chandhar,
Chandhar Research Labs, Chennai, India.

System model
------------
* A ground station (GS) with an M-element uniform linear array (ULA),
  element spacing ``del``.
* K drones (UAVs) dropped uniformly at random inside a sphere of radius
  ``R = 500 m`` centred at the array.
* Line-of-sight channel with free-space path loss and a spherical-wavefront
  (Fresnel / near-field) phase model:

      g_k[m] = sqrt(beta_k) * exp(-j 2 pi / lambda *
                (d_k + (m*del)^2 / (2 d_k) - m*del*sin(theta_k)cos(phi_k)))

  with beta_k = (lambda / (4 pi d_k))^2.
* Channel-inversion power control: drone k transmits with power
  p_k = rho_u / beta_k, so every drone arrives with the same SNR rho_u.
* Least-squares pilot-based estimate  G_est = G + W / sqrt(p_p).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 3e8  # speed of light (m/s)


@dataclass
class Result:
    """Outputs of :func:`calc_spawc2018` (mirrors the MATLAB function)."""

    exp_sinr_exact: float      # E[SINR]   – exact (Monte-Carlo inner expectation)
    exp_sinr_approx: float     # E[SINR]   – closed-form approximation
    exp_inv_sinr_exact: float  # E[1/SINR] – exact
    exp_inv_sinr_approx: float # E[1/SINR] – approximation
    rate_zf: float             # simulated ergodic rate, ZF  (bits/s/Hz)
    rate_mrc: float            # simulated ergodic rate, MRC (bits/s/Hz)

    # Rate bounds derived from the above (Jensen's inequality)
    @property
    def rate_ub_exact(self) -> float:
        return float(np.log2(1 + self.exp_sinr_exact))

    @property
    def rate_ub_approx(self) -> float:
        return float(np.log2(1 + self.exp_sinr_approx))

    @property
    def rate_lb_exact(self) -> float:
        return float(np.log2(1 + 1 / self.exp_inv_sinr_exact))

    @property
    def rate_lb_approx(self) -> float:
        return float(np.log2(1 + 1 / self.exp_inv_sinr_approx))


def drop_drones(K: int, radius: float, rng: np.random.Generator):
    """Uniformly drop K drones inside a sphere; return distance and
    direction cosine along the array axis (sin(el)*cos(az))."""
    el = np.arcsin(2 * rng.random(K) - 1) + np.pi / 2
    az = 2 * np.pi * rng.random(K)
    d = radius * rng.random(K) ** (1 / 3)
    z1 = np.sin(el) * np.cos(az)
    return d, z1


def los_channel(M: int, d: np.ndarray, z1: np.ndarray, delta: float, lam: float):
    """M x K near-field LoS channel matrix."""
    m = np.arange(M)[:, None]                         # (M,1)
    beta = (lam / (4 * np.pi * d)) ** 2               # (K,)
    phase = d + (m * delta) ** 2 / (2 * d) - delta * m * z1
    G = np.sqrt(beta) * np.exp(-1j * 2 * np.pi * phase / lam)
    return G, beta


def _crandn(rng, *shape):
    return np.sqrt(0.5) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def calc_spawc2018(M: int, K: int, delta: float, lam: float, pu_p: float,
                   rhou: float, n_drops: int = 200, n_noise: int = 300,
                   radius: float = 500.0, rng: np.random.Generator | None = None
                   ) -> Result:
    """Port of ``Calc_SPAWC2018.m``.

    Parameters
    ----------
    M, K      : number of GS antennas / drones
    delta     : antenna spacing (m)
    lam       : wavelength (m)
    pu_p      : pilot transmit power (linear)
    rhou      : received data SNR per drone (linear)
    n_drops   : number of random drone drops (outer loop, MATLAB: 200)
    n_noise   : noise / estimation-error realisations per drop (MATLAB: 300)
    """
    rng = np.random.default_rng() if rng is None else rng
    # Independent streams for drone positions and for noise, so that the
    # drone drops do not depend on M (useful for common random numbers).
    drop_rng, noise_rng = rng.spawn(2)
    e = 1 / np.sqrt(pu_p)
    I = np.eye(K)

    sinr_exact = np.empty(n_drops)
    sinr_approx = np.empty(n_drops)
    inv_exact = np.empty(n_drops)
    inv_approx = np.empty(n_drops)
    sinr_mrc = np.empty(n_drops)

    for t in range(n_drops):
        d, z1 = drop_drones(K, radius, drop_rng)
        G, beta = los_channel(M, d, z1, delta, lam)
        pu = rhou / beta                                  # power control
        sq_pu = np.sqrt(pu)

        # ---- n_noise realisations, vectorised over the batch axis ----
        W = _crandn(noise_rng, n_noise, M, K)
        G_est = G[None] + e * W                           # (B,M,K)
        n = _crandn(noise_rng, n_noise, M, 1)
        q = _crandn(noise_rng, n_noise, K, 1)
        x = sq_pu[None, :, None] * q                      # (B,K,1)

        # ZF: G_est^+ = (G_est^H G_est)^-1 G_est^H
        G_estH = np.conj(np.swapaxes(G_est, 1, 2))        # (B,K,M)
        G_pinv = np.linalg.solve(G_estH @ G_est, G_estH)  # (B,K,M)
        z = (G_pinv @ G - I) @ x + G_pinv @ n             # (B,K,1)
        Ez = np.mean(np.abs(z[..., 0]) ** 2, axis=0)      # diag E[zz^H]

        # Closed-form approximation of the effective noise covariance
        GHG_inv = np.linalg.inv(G.conj().T @ G)
        Ez_apx = np.real(np.diag(
            (1 + pu.sum() / pu_p + np.trace(GHG_inv).real / pu_p) * GHG_inv))

        sinr_exact[t] = pu[0] / Ez[0]
        sinr_approx[t] = pu[0] / Ez_apx[0]
        inv_exact[t] = Ez[0] / pu[0]
        inv_approx[t] = Ez_apx[0] / pu[0]

        # MRC for drone 1: combiner = g_est_1
        a = G_estH[:, 0, :] @ G                           # (B,K): g_est1^H g_k
        rx = np.abs(a) ** 2 * pu[None] * np.abs(q[..., 0]) ** 2
        des = rx[:, 0].mean()
        intf = rx[:, 1:].sum(axis=1).mean()
        # NOTE: the MATLAB code uses G_est(:,K) here (loop-variable leftover);
        # drone 1's combiner G_est(:,1) is used as intended.
        noise = np.mean(np.abs(np.sum(G_estH[:, 0, :] * n[..., 0], axis=1)) ** 2)
        sinr_mrc[t] = des / (intf + noise)

    return Result(
        exp_sinr_exact=float(sinr_exact.mean()),
        exp_sinr_approx=float(sinr_approx.mean()),
        exp_inv_sinr_exact=float(inv_exact.mean()),
        exp_inv_sinr_approx=float(inv_approx.mean()),
        rate_zf=float(np.mean(np.log2(1 + sinr_exact))),
        rate_mrc=float(np.mean(np.log2(1 + sinr_mrc))),
    )

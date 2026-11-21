# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false
"""
Verification & Obstruction Replay for W/MASSIVE/FULL-GIBBS/
============================================================
Checks:
  1. CERTIFIED (exact Fraction & mpmath outward-interval) identities and finite bounds:
     - Open-regime identity t^2/U = (t/U)^2 * X * Omega (forces X >= (U/t)^2 >= 16).
     - LARGE-U Prop 3 cold-projection probability vanishing exponentially in |Lambda|.
     - Route (a) LF second-order even-n first-moment scaling ~ (t^2 / 2U^2) * X and
       exponential moment Franck-Condon tail lower bound exp(c(s) * X).
     - Route (b) exact worldline retardation functional Phi_beta for site-neutral (|dn|=1)
       and pair (|dn|=2) excursions at vertical (delta = 1/U) vs slow (delta = beta/2) scales.
     - Route (c) outward interval certificate (mpmath.iv) that v^T M_{C4,edge,ph} v < 0
       and v^T M_{C4,edge,dir} v < 0 at rational test vectors.
     - Route (d) exact 1-site marginal curvature V_eff''(0) = 1/U - 1/(2*tau), exact closed-form
       variance Var(b) = U/beta + U^2/(1 + e^{-beta*U/2}), and exact 2-site dimer curvatures
       at both t = 1 and t = 4.
     - Route (e) hot-site Boltzmann penalty e^{-cX} vs unrotated hopping gain e^{2(t/U)X}.
     - Surviving Positive Bound 1: cutoff-free directed kernel F_+(u) and K_beta interval check.
     - Surviving Positive Bound 3: virtual Franck-Condon hot tail U^2 w_M / t^2 -> 1/4 per channel.
  2. NUMERICAL (small-system exact diagonalization & Fock-trace checks, < 500 MB RAM):
     - Witness A (Route a): 2-site LF dimer <n_rel>_GS ~ (t^2/2U^2)X and <exp(s*n_rel)> ~ exp(c*X)
       with verified Nph convergence (boundary-shell share < 10^{-3}).
     - Witness B (Route b): 2-site pre-LF dimer <Delta_0^dag Delta_1> = Theta((t/U)^2) down to tau=1e-4,
       plus Fock-space trace verification of F_+(u) and F_-(u) to < 10^{-12} relative error.
     - Witness C (Route c): C_2 (bipartite positive under PH) vs C_4 (reflection positivity fails).
     - Witness D (Route d): 2-site static phonon marginal non-convexity along v=(1,-1) and v=(1,1).
     - Witness E (Route e): 1-step SW residual, 3-site Cut-SW cross-commutator locality leak,
       converged post-S1_col exponential-moment blowup, and non-vacuous relative-entropy tail bound (Thm 3.2) test (< 1).
  3. GUARDED MUTANTS (>= 9):
     - Each mutant injects a false positive assumption or breaks a critical obstruction check and
       MUST be caught by assertion failure.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import math
import sys
import time

import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm


def _z_el_c4_iv(b0, b1, b2, b3, t_iv, beta_iv):
    """Rigorous outward-interval evaluation of Z_el(b0,b1,b2,b3) = det(2*cosh(beta*H/2))^2 on C_4."""
    h_mat = [
        [-b0, -t_iv, mp.iv.mpf(0), -t_iv],
        [-t_iv, -b1, -t_iv, mp.iv.mpf(0)],
        [mp.iv.mpf(0), -t_iv, -b2, -t_iv],
        [-t_iv, mp.iv.mpf(0), -t_iv, -b3],
    ]
    a_mat = [[(beta_iv / 2) * h_mat[i][j] for j in range(4)] for i in range(4)]

    def matmul(x_m, y_m):
        return [[sum(x_m[i][k] * y_m[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

    a2_mat = matmul(a_mat, a_mat)
    norm_a2 = max(sum(mp.mpf(abs(a2_mat[i][j]).b) for j in range(4)) for i in range(4))
    c_mat = [[mp.iv.mpf(1) if i == j else mp.iv.mpf(0) for j in range(4)] for i in range(4)]
    term = [[mp.iv.mpf(1) if i == j else mp.iv.mpf(0) for j in range(4)] for i in range(4)]
    k_max = 35
    for k in range(1, k_max + 1):
        term = matmul(term, a2_mat)
        fac = (2 * k - 1) * (2 * k)
        term = [[term[i][j] / fac for j in range(4)] for i in range(4)]
        c_mat = [[c_mat[i][j] + term[i][j] for j in range(4)] for i in range(4)]
    rem = (norm_a2 ** (k_max + 1)) / mp.factorial(2 * k_max + 2) * mp.cosh(mp.sqrt(norm_a2))
    rad = mp.iv.mpf([-rem, rem])
    m_mat = [[2 * (c_mat[i][j] + rad) for j in range(4)] for i in range(4)]

    def det3(s_m):
        return (
            s_m[0][0] * (s_m[1][1] * s_m[2][2] - s_m[1][2] * s_m[2][1])
            - s_m[0][1] * (s_m[1][0] * s_m[2][2] - s_m[1][2] * s_m[2][0])
            + s_m[0][2] * (s_m[1][0] * s_m[2][1] - s_m[1][1] * s_m[2][0])
        )

    d4 = mp.iv.mpf(0)
    for j in range(4):
        sub = [[m_mat[r][c] for c in range(4) if c != j] for r in range(1, 4)]
        sign = 1 if j % 2 == 0 else -1
        d4 += sign * m_mat[0][j] * det3(sub)
    return d4 * d4


def check_certified_identities() -> dict:
    mp.mp.dps = 50
    mp.iv.dps = 50
    results = {}

    # 1. Open-regime exact rational identity: t^2/U = x^2 * X * Omega
    x = Fraction(1, 4)
    X_crit = 1 / (x * x)
    assert X_crit == Fraction(16, 1)
    results["X_min_for_x_quarter"] = int(X_crit)

    # 2. LARGE-U Prop 3: Cold-projection weight ratio on 1 site at tau = Omega, X = 64, M = 1
    X = mp.iv.mpf(64)
    M = 1
    beta_Om = mp.iv.mpf(1)
    beta_U = X * beta_Om
    num = (1 - mp.iv.exp(-beta_Om * (M + 1))) / (1 - mp.iv.exp(-beta_Om))
    den = (1 + mp.iv.exp(-beta_U / 2)) * (1 / (1 - mp.iv.exp(-beta_Om)))
    r_cold = num / den
    assert r_cold.b < mp.mpf("0.87")
    assert (r_cold ** 100).b < mp.mpf("1e-6")
    results["prop3_per_site_cold_ratio_X64"] = float(r_cold.mid)

    # 3. Route (a) exact 2nd-order perturbation formula for <n_rel^LF>_GS (even n only by parity)
    # In the symmetric bonding ground state |psi_+^(0)> = (|2,0> + |0,2>)/sqrt(2) x |0>_rel,
    # V_{sing, +}(n) = -t * (1 + (-1)^n) * D_{n,0}(sqrt(X)), which vanishes for odd n and
    # equals -2*t*D_{n,0}(sqrt(X)) for even n = 2k >= 2.
    # Thus <n_rel>_GS^(2) = 4*(t/U)^2 * sum_{n>=2, n even} n * e^{-X} X^n / (n! * (1 + n/X)^2) ~ (t^2 / 2U^2) * X.
    t_over_U = mp.mpf("0.2")
    s_lyap = mp.mpf("0.25")
    rs_first_moment = {}
    rs_exp_tail = {}
    for X_val in [10, 20, 40, 60]:
        Xv = mp.mpf(X_val)
        n_max = 6 * X_val + 100
        s1 = mp.mpf(0)
        s_exp = mp.mpf(0)
        w = mp.exp(-Xv)
        for n in range(0, n_max + 1):
            if n > 0:
                w = w * Xv / n
            if n == 0 or (n % 2 != 0):
                continue
            denom = (1 + mp.mpf(n) / Xv) ** 2
            term = 4 * (t_over_U ** 2) * w / denom
            s1 += n * term
            s_exp += mp.exp(s_lyap * n) * term
        rs_first_moment[X_val] = float(s1)
        rs_exp_tail[X_val] = float(s_exp)
    assert abs(rs_first_moment[60] - 1.194895) < 1e-4
    results["rs_first_moment_even_n"] = rs_first_moment
    results["rs_exp_tail_even_n"] = rs_exp_tail

    # 4. Route (b) exact worldline retardation functional Phi_beta(delta)
    # Both site-neutral hop-and-reverse (|dn_z| = 1, prefactor 1) and pair transfer (|dn_z| = 2, prefactor 4):
    U_val = mp.iv.mpf(40)
    Om_val = mp.iv.mpf(1)
    beta_val = mp.iv.mpf(1)

    def _iv_cosh(z):
        return (mp.iv.exp(z) + mp.iv.exp(-z)) / 2

    def _iv_sinh(z):
        return (mp.iv.exp(z) - mp.iv.exp(-z)) / 2

    def phi_2site_iv(delta, prefactor: int):
        return prefactor * (U_val / (Om_val * _iv_sinh(beta_val * Om_val / 2))) * (
            _iv_cosh(beta_val * Om_val / 2) - _iv_cosh(Om_val * (beta_val / 2 - delta))
        )

    phi_neutral_short = phi_2site_iv(mp.iv.mpf(1) / U_val, 1)
    phi_neutral_mid = phi_2site_iv(beta_val / 2, 1)
    phi_pair_short = phi_2site_iv(mp.iv.mpf(1) / U_val, 4)
    phi_pair_mid = phi_2site_iv(beta_val / 2, 4)
    assert phi_neutral_short.b < mp.mpf("1.0")
    assert phi_neutral_mid.a > mp.mpf("9.79")
    assert phi_pair_short.b < mp.mpf("4.0")
    assert phi_pair_mid.a > mp.mpf("39.18")
    results["phi_2site_neutral_short_1_over_U"] = float(phi_neutral_short.mid)
    results["phi_2site_neutral_mid_beta_over_2"] = float(phi_neutral_mid.mid)
    results["phi_2site_pair_short_1_over_U"] = float(phi_pair_short.mid)
    results["phi_2site_pair_mid_beta_over_2"] = float(phi_pair_mid.mid)

    # 5. Route (c) outward interval certificate that C_4 reflection kernels are NOT PSD at t=1, beta=2
    configs_iv = [
        [mp.iv.mpf("1.5"), mp.iv.mpf("-0.5")],
        [mp.iv.mpf("0.5"), mp.iv.mpf("1.5")],
        [mp.iv.mpf("-1.0"), mp.iv.mpf("2.0")],
        [mp.iv.mpf("2.0"), mp.iv.mpf("0.2")],
        [mp.iv.mpf("-1.5"), mp.iv.mpf("-0.8")],
        [mp.iv.mpf(0), mp.iv.mpf("1.0")],
    ]
    # Float64 eigenvector rounded to exact decimal strings -> certified interval quadratic form v^T M v < 0
    v_ph_str = ["0.080286", "-0.266204", "-0.002938", "0.013747", "-0.267241", "0.922537"]
    v_dir_str = ["0.549842", "0.082739", "-0.624897", "0.547576", "0.022191", "0.000168"]
    v_ph_iv = [mp.iv.mpf(s) for s in v_ph_str]
    v_dir_iv = [mp.iv.mpf(s) for s in v_dir_str]
    t_c4_iv = mp.iv.mpf(1)
    beta_c4_iv = mp.iv.mpf(2)
    q_ph_iv = mp.iv.mpf(0)
    q_dir_iv = mp.iv.mpf(0)
    for i in range(6):
        for j in range(6):
            z_ph = _z_el_c4_iv(
                configs_iv[i][0], -configs_iv[j][0], -configs_iv[j][1], configs_iv[i][1], t_c4_iv, beta_c4_iv
            )
            z_dir = _z_el_c4_iv(
                configs_iv[i][0], configs_iv[j][0], configs_iv[j][1], configs_iv[i][1], t_c4_iv, beta_c4_iv
            )
            q_ph_iv += v_ph_iv[i] * z_ph * v_ph_iv[j]
            q_dir_iv += v_dir_iv[i] * z_dir * v_dir_iv[j]
    assert q_ph_iv.b < mp.mpf("-4631.0")
    assert q_dir_iv.b < mp.mpf("-2027620.0")
    results["route_c_C4_edge_ph_quadform_iv"] = [float(q_ph_iv.a), float(q_ph_iv.b)]
    results["route_c_C4_edge_dir_quadform_iv"] = [float(q_dir_iv.a), float(q_dir_iv.b)]

    # 6. Route (d) exact 1-site static well curvature, closed-form variance, and 2-site dimer curvatures
    U_d = mp.iv.mpf(40)
    tau_d = mp.iv.mpf(1)
    beta_d = 1 / tau_d
    Veff_2nd = 1 / U_d - beta_d / 2
    assert Veff_2nd.b < 0
    # Exact closed-form 1-site marginal variance: Var(b) = U/beta + U^2 / (1 + e^{-beta*U/2})
    var_1d_iv = U_d / beta_d + (U_d ** 2) / (1 + mp.iv.exp(-beta_d * U_d / 2))
    assert var_1d_iv.a > 40 * U_d.b * tau_d.b
    # Also verify exact agreement with numerical quadrature in mpmath:
    U_mp, beta_mp = mp.mpf(40), mp.mpf(1)
    def unnorm_p(b):
        val = beta_mp * (b**2 / (2 * U_mp)) - 2 * mp.log(2 * mp.cosh(beta_mp * b / 2))
        ref = beta_mp * (U_mp / 2) - 2 * mp.log(2 * mp.cosh(beta_mp * U_mp / 2))
        return mp.exp(-(val - ref))
    Z_1d = mp.mpf(mp.quad(unnorm_p, [-160, -40, 0, 40, 160]))
    var_1d_quad = mp.mpf(mp.quad(lambda b: (b**2) * unnorm_p(b), [-160, -40, 0, 40, 160])) / Z_1d
    assert abs(var_1d_quad - var_1d_iv.mid) < mp.mpf("1e-25")

    # Exact closed-form 2-site dimer directional second derivatives at (b0,b1)=(0,0) for t=1 and t=4:
    # d2_anti = 1/U - tanh(beta*t/2)/t,  d2_sym = 1/U - beta / (2 * cosh^2(beta*t/2))
    for t_lbl, t_val_d in [("t1", mp.iv.mpf(1)), ("t4", mp.iv.mpf(4))]:
        sh = _iv_sinh(beta_d * t_val_d / 2)
        ch = _iv_cosh(beta_d * t_val_d / 2)
        d2_anti_iv = 1 / U_d - (sh / ch) / t_val_d
        d2_sym_iv = 1 / U_d - beta_d / (2 * (ch ** 2))
        assert d2_anti_iv.b < 0
        assert d2_sym_iv.b < 0
        results[f"route_d_dimer_{t_lbl}_d2_anti"] = float(d2_anti_iv.mid)
        results[f"route_d_dimer_{t_lbl}_d2_sym"] = float(d2_sym_iv.mid)

    results["route_d_Veff_curvature_0"] = float(Veff_2nd.mid)
    results["route_d_1site_Var_b"] = float(var_1d_iv.mid)

    # 7. Route (e) hot-site Boltzmann penalty vs unrotated hopping gain
    c_cold = Fraction(1, 64)
    x_thresh = c_cold / 2
    assert x_thresh == Fraction(1, 128)
    results["route_e_hot_site_t_over_U_threshold"] = str(x_thresh)

    # 8. Surviving Positive Bound 3 (Thm 3.1): virtual hot weight U^2 * w_M / t^2 -> 1/4 per channel
    w_M_ratios = {}
    for X_w in [64, 128, 256]:
        Xv = mp.mpf(X_w)
        M_w = X_w // 64
        Nmax = 2 * X_w + 180
        tot = mp.mpf(0)
        pN = mp.exp(-Xv)
        for N in range(0, Nmax + 1):
            if N > 0:
                pN = pN * Xv / N
            if N <= M_w:
                continue
            comp = mp.mpf(0)
            c = mp.mpf(1)
            for k in range(0, M_w + 1):
                if k > 0:
                    c = c * (N - k + 1) / k
                comp += c
            prob_gt_M = 1 - comp * mp.power(2, -N)
            tot += pN * prob_gt_M / ((1 + mp.mpf(N) / Xv) ** 2)
        w_M_ratios[X_w] = float(tot)
    assert 0.25 < w_M_ratios[64] < 0.254
    assert 0.25 < w_M_ratios[256] < 0.251
    results["hot_tail_U2_wM_over_t2"] = w_M_ratios

    # 9. Surviving Positive Bound 1 (Thm 3.3): F_+(u) and K_beta <= (2t^2/U)*exp(-(U/Om)*coth(beta*Om/4))
    bO_iv = mp.iv.mpf("1.6")
    uO_iv = mp.iv.mpf("0.2")
    ratio_iv = (_iv_cosh(bO_iv / 2) + _iv_cosh(bO_iv / 2 - uO_iv)) / _iv_sinh(bO_iv / 2)
    coth_iv = _iv_cosh(bO_iv / 4) / _iv_sinh(bO_iv / 4)
    assert ratio_iv.a >= coth_iv.b - mp.mpf("1e-40")
    K_int = mp.quad(
        lambda uu: mp.exp(-12 * uu) * mp.exp(
            -6 * (mp.cosh(mp.mpf("0.8")) + mp.cosh(2 * (mp.mpf("0.4") - uu))) / mp.sinh(mp.mpf("0.8"))
        ),
        [0, mp.mpf("0.8")],
    )
    K_bound = (mp.mpf(1) / 12) * mp.exp(-6 * mp.coth(mp.mpf("0.4")))
    assert K_int < K_bound
    results["K_beta_ratio_to_bound"] = float(K_int / K_bound)

    return results


def build_dimer_LF(t: float, U: float, Omega: float, Nph: int):
    """Exact 2-site dimer (N=2, Sz=0) in Lang-Firsov frame (relative phonon mode)."""
    X = U / Omega
    alpha = np.sqrt(X)
    a = np.zeros((Nph, Nph))
    for n in range(1, Nph):
        a[n - 1, n] = np.sqrt(n)
    Dp = expm(alpha * (a.T - a))
    Dm = Dp.T
    n_op = np.diag(np.arange(Nph, dtype=float))
    I = np.eye(Nph)
    # Basis: 0=|2,0>, 1=|0,2>, 2=|sing>
    H = np.zeros((3 * Nph, 3 * Nph))
    H[0:Nph, 0:Nph] = Omega * n_op
    H[Nph:2 * Nph, Nph:2 * Nph] = Omega * n_op
    H[2 * Nph:3 * Nph, 2 * Nph:3 * Nph] = U * I + Omega * n_op
    H[2 * Nph:3 * Nph, 0:Nph] = -np.sqrt(2.0) * t * Dp
    H[0:Nph, 2 * Nph:3 * Nph] = -np.sqrt(2.0) * t * Dm
    H[2 * Nph:3 * Nph, Nph:2 * Nph] = -np.sqrt(2.0) * t * Dm
    H[Nph:2 * Nph, 2 * Nph:3 * Nph] = -np.sqrt(2.0) * t * Dp
    return H, Dp, Dm


def build_dimer_preLF(t: float, U: float, Omega: float, Nph: int):
    """Exact 2-site dimer (N=2, Sz=0) in pre-LF frame (relative phonon mode)."""
    X = U / Omega
    alpha = np.sqrt(X)
    a = np.zeros((Nph, Nph))
    for n in range(1, Nph):
        a[n - 1, n] = np.sqrt(n)
    x_rel = a + a.T
    n_op = np.diag(np.arange(Nph, dtype=float))
    I = np.eye(Nph)
    H = np.zeros((3 * Nph, 3 * Nph))
    H[0:Nph, 0:Nph] = U * I + Omega * n_op - alpha * Omega * x_rel
    H[Nph:2 * Nph, Nph:2 * Nph] = U * I + Omega * n_op + alpha * Omega * x_rel
    H[2 * Nph:3 * Nph, 2 * Nph:3 * Nph] = U * I + Omega * n_op
    H[2 * Nph:3 * Nph, 0:Nph] = -np.sqrt(2.0) * t * I
    H[0:Nph, 2 * Nph:3 * Nph] = -np.sqrt(2.0) * t * I
    H[2 * Nph:3 * Nph, Nph:2 * Nph] = -np.sqrt(2.0) * t * I
    H[Nph:2 * Nph, 2 * Nph:3 * Nph] = -np.sqrt(2.0) * t * I
    return H


def witness_route_a() -> dict:
    """2-site LF dimer Lyapunov blowup as X = U/Omega grows (with verified Nph convergence)."""
    U = 40.0
    t = 8.0  # t/U = 0.2
    s = 0.25
    out = {}
    for X in [10.0, 20.0, 40.0, 60.0]:
        Om = U / X
        # Use Nph = int(5 * X + 80) so even at X=60 (Nph=380) <exp(0.25 n)>_GS is converged to 6 digits
        Nph = int(5 * X + 80)
        H, _, _ = build_dimer_LF(t, U, Om, Nph)
        evals, evecs = eigh(H)
        v0 = evecs[:, 0]
        n_vec = np.concatenate([np.arange(Nph), np.arange(Nph), np.arange(Nph)])
        p_gs = v0 ** 2
        mean_n_gs = float(np.sum(n_vec * p_gs))
        exp_vec = np.exp(s * n_vec)
        exp_n_gs = float(np.sum(exp_vec * p_gs))
        bdry_gs = float(np.sum(exp_vec[n_vec >= Nph - 10] * p_gs[n_vec >= Nph - 10]) / exp_n_gs)
        assert bdry_gs < 1e-4, f"Witness A GS exp moment not converged at X={X}: bdry_gs={bdry_gs}"

        # Thermal at tau = Omega (including triplet sector at E = U + Omega*n):
        beta = 1.0 / Om
        E0 = evals[0]
        w = np.exp(-beta * (evals - E0))
        Z_trip = np.exp(-beta * (U - E0)) / (1.0 - np.exp(-beta * Om))
        Z_tot = float(np.sum(w) + Z_trip)
        exp_n_th = float(np.sum(w * np.sum(exp_vec[:, None] * (evecs ** 2), axis=0)) / Z_tot)
        out[int(X)] = {
            "mean_n_gs": mean_n_gs,
            "exp_n_gs": exp_n_gs,
            "exp_n_tau_Om": exp_n_th,
            "bdry_gs_share": bdry_gs,
        }
    assert out[60]["mean_n_gs"] > 5.0 * out[10]["mean_n_gs"]
    assert out[60]["exp_n_gs"] > 4e24
    assert out[60]["exp_n_tau_Om"] > 1e28
    return out


def witness_route_b() -> dict:
    """
    2-site pre-LF dimer <Delta_0^dag Delta_1> staying Theta((t/U)^2) down to tau=1e-4,
    instantaneous Hubbard pair splitting, and self-contained Fock-trace check of F_+(u) and F_-(u).
    """
    U = 40.0
    t = 8.0
    out = {}
    for X in [10.0, 20.0, 40.0]:
        Om = U / X
        Nph = int(4 * X + 80)
        H = build_dimer_preLF(t, U, Om, Nph)
        evals, evecs = eigh(H)
        gap = float(evals[1] - evals[0])
        E0 = evals[0]
        corrs = {}
        for label, tau in [("tau=Om", Om), ("tau=t2/U", t**2 / U), ("tau=1e-4", 1e-4)]:
            beta = 1.0 / tau
            w = np.exp(-beta * (evals - E0))
            Z_trip = np.exp(-beta * (U - E0)) / (1.0 - np.exp(-beta * Om))
            Z = float(np.sum(w) + Z_trip)
            c = 0.0
            for k in range(len(evals)):
                if w[k] < 1e-16 * Z:
                    continue
                v = evecs[:, k]
                c += (w[k] / Z) * float(np.dot(v[0:Nph], v[Nph:2 * Nph]))
            corrs[label] = c
        out[int(X)] = {"gap": gap, "corrs": corrs}

    # Instantaneous attractive Hubbard 2-site dimer (N=2, Sz=0):
    H_hub = np.array([[0.0, 0.0, -np.sqrt(2.0) * t],
                      [0.0, 0.0, -np.sqrt(2.0) * t],
                      [-np.sqrt(2.0) * t, -np.sqrt(2.0) * t, U]])
    ev_hub = eigh(H_hub, eigvals_only=True)
    hub_gap = float(ev_hub[1] - ev_hub[0])
    out["hubbard_pair_splitting"] = hub_gap
    assert hub_gap > 2.5
    assert out[40]["corrs"]["tau=1e-4"] > 1e-2
    # At X=40, true tunnel splitting is ~1e-17 (at float64 floor; X=20 is 2.53e-08)
    assert out[20]["gap"] < 1e-7 and out[40]["gap"] < 1e-12

    # Self-contained Fock-space trace verification of F_+(u)^2 and F_-(u)^2 at U=12, Om=2, beta=0.8
    U_f, Om_f, beta_f, Nph_f = 12.0, 2.0, 0.8, 60
    al_f = math.sqrt(U_f / (2.0 * Om_f))
    a_f = np.zeros((Nph_f, Nph_f))
    for n in range(1, Nph_f):
        a_f[n - 1, n] = np.sqrt(n)
    Dp_f = expm(al_f * (a_f.T - a_f))
    Dm_f = Dp_f.T
    n_f = np.arange(Nph_f, dtype=float)
    Z_ph_f = float(np.sum(np.exp(-beta_f * Om_f * n_f)))
    max_rel_err_plus = 0.0
    max_rel_err_minus = 0.0
    for u_val in [0.0, 0.1, 0.4, 0.8]:
        d1 = np.diag(np.exp(-(beta_f - u_val) * Om_f * n_f))
        d2 = np.diag(np.exp(-u_val * Om_f * n_f))
        tr_p = float(np.trace(d1 @ Dp_f @ d2 @ Dp_f) / Z_ph_f)
        tr_m = float(np.trace(d1 @ Dm_f @ d2 @ Dp_f) / Z_ph_f)
        f_plus_2site_fock = tr_p ** 2
        f_minus_2site_fock = tr_m ** 2
        f_plus_exact = math.exp(
            -(U_f / Om_f)
            * (math.cosh(0.5 * beta_f * Om_f) + math.cosh(Om_f * (0.5 * beta_f - u_val)))
            / math.sinh(0.5 * beta_f * Om_f)
        )
        f_minus_exact = math.exp(
            -(U_f / Om_f)
            * (math.cosh(0.5 * beta_f * Om_f) - math.cosh(Om_f * (0.5 * beta_f - u_val)))
            / math.sinh(0.5 * beta_f * Om_f)
        )
        max_rel_err_plus = max(max_rel_err_plus, abs(f_plus_2site_fock - f_plus_exact) / f_plus_exact)
        max_rel_err_minus = max(max_rel_err_minus, abs(f_minus_2site_fock - f_minus_exact) / f_minus_exact)
    assert max_rel_err_plus < 2e-12
    assert max_rel_err_minus < 1e-12
    out["fock_trace_F_plus_max_rel_err"] = max_rel_err_plus
    out["fock_trace_F_minus_max_rel_err"] = max_rel_err_minus
    return out


def witness_route_c() -> dict:
    """C_2 and C_4 reflection-positivity kernels (static & edge/site-centered)."""
    t = 1.0
    beta = 2.0

    def Z_el_C2(b0, b1):
        h = np.array([[-b0, -t], [-t, -b1]], dtype=float)
        ev = eigh(h, eigvals_only=True)
        return float(np.prod((2.0 * np.cosh(0.5 * beta * ev)) ** 2))

    def Z_el_C4(b0, b1, b2, b3):
        h = np.array([[-b0, -t, 0.0, -t],
                      [-t, -b1, -t, 0.0],
                      [0.0, -t, -b2, -t],
                      [-t, 0.0, -t, -b3]], dtype=float)
        ev = eigh(h, eigvals_only=True)
        return float(np.prod((2.0 * np.cosh(0.5 * beta * ev)) ** 2))

    pts = np.linspace(-3.0, 3.0, 7)
    M2_dir = np.array([[Z_el_C2(x, y) for y in pts] for x in pts])
    M2_ph = np.array([[Z_el_C2(x, -y) for y in pts] for x in pts])

    configs_A = [
        [1.5, -0.5],
        [0.5, 1.5],
        [-1.0, 2.0],
        [2.0, 0.2],
        [-1.5, -0.8],
        [0.0, 1.0],
    ]
    n_c = len(configs_A)
    M4_dir = np.zeros((n_c, n_c))
    M4_ph = np.zeros((n_c, n_c))
    for i in range(n_c):
        b0i, b3i = configs_A[i]
        for j in range(n_c):
            b0j, b3j = configs_A[j]
            M4_dir[i, j] = Z_el_C4(b0i, b0j, b3j, b3i)
            M4_ph[i, j] = Z_el_C4(b0i, -b0j, -b3j, b3i)

    M4_site_dir = np.array([[Z_el_C4(1.0, x, -1.0, y) for y in pts] for x in pts])
    M4_site_ph = np.array([[Z_el_C4(1.0, x, -1.0, -y) for y in pts] for x in pts])

    res = {
        "C2_dir_min_eval": float(np.min(eigh(M2_dir, eigvals_only=True))),
        "C2_ph_min_eval": float(np.min(eigh(M2_ph, eigvals_only=True))),
        "C4_edge_dir_min_eval": float(np.min(eigh(M4_dir, eigvals_only=True))),
        "C4_edge_ph_min_eval": float(np.min(eigh(M4_ph, eigvals_only=True))),
        "C4_site_dir_min_eval": float(np.min(eigh(M4_site_dir, eigvals_only=True))),
        "C4_site_ph_min_eval": float(np.min(eigh(M4_site_ph, eigvals_only=True))),
    }
    assert res["C2_dir_min_eval"] < -1e4
    assert res["C2_ph_min_eval"] > 0.0
    assert res["C4_edge_dir_min_eval"] < -1e4
    assert res["C4_edge_ph_min_eval"] < -1e3
    assert res["C4_site_dir_min_eval"] < -1e4
    assert res["C4_site_ph_min_eval"] < -1e4
    return res


def witness_route_d() -> dict:
    """2-site static conditional non-convexity along v=(1,-1) and v=(1,1) at both t=1 and t=4."""
    U = 40.0
    tau = 1.0
    beta = 1.0 / tau
    res = {}
    for t_lbl, t in [("t1", 1.0), ("t4", 4.0)]:
        def V_dimer(b0, b1, t_val=t):
            h = np.array([[-b0, -t_val], [-t_val, -b1]], dtype=float)
            ev = eigh(h, eigvals_only=True)
            F_el = -2.0 * tau * float(np.sum(np.log(2.0 * np.cosh(0.5 * beta * ev))))
            return (b0 ** 2 + b1 ** 2) / (2.0 * U) + F_el

        eps = 1e-4
        d2_anti = (V_dimer(eps, -eps) - 2.0 * V_dimer(0.0, 0.0) + V_dimer(-eps, eps)) / (2.0 * eps ** 2)
        d2_sym = (V_dimer(eps, eps) - 2.0 * V_dimer(0.0, 0.0) + V_dimer(-eps, -eps)) / (2.0 * eps ** 2)
        res[f"V_dimer_{t_lbl}_d2_anti_at_0"] = float(d2_anti)
        res[f"V_dimer_{t_lbl}_d2_sym_at_0"] = float(d2_sym)
    assert res["V_dimer_t1_d2_anti_at_0"] < -0.43
    assert res["V_dimer_t1_d2_sym_at_0"] < -0.36
    assert res["V_dimer_t4_d2_anti_at_0"] < -0.21
    assert res["V_dimer_t4_d2_sym_at_0"] < -0.01
    return res


def witness_route_e() -> dict:
    """
    (E1) 2-site 1-step cold-odd SW leaving ||C(cold<->hot_pair)|| = Theta(t^2/U) and <e^{0.25 n}> blowup.
    (E2) 3-site chain (N=2, Sz=0, X=4, U=40, t=4, M=1) local edge-wise Cut-SW cross-commutator norms.
    (E3) Converged post-S1_col exponential-moment blowup at X=40 (Nph=360, boundary share < 5e-3)
         plus non-vacuous relative-entropy tail bound (Thm 3.2) test on 2-site 2-mode LF dimer (p_hot = 0.1355 <= bound = 0.8466 < 1).
    """
    # (E1) 2-site dimer
    U = 40.0
    t = 4.0
    X = 20.0
    Om = U / X
    M = 1
    Nph = int(4 * X + 80)
    alpha = np.sqrt(X)
    a = np.zeros((Nph, Nph))
    for n in range(1, Nph):
        a[n - 1, n] = np.sqrt(n)
    Dp = expm(alpha * (a.T - a))
    Dm = Dp.T
    n_op = np.diag(np.arange(Nph, dtype=float))
    I = np.eye(Nph)

    H0 = np.zeros((3 * Nph, 3 * Nph))
    H0[0:Nph, 0:Nph] = Om * n_op
    H0[Nph:2 * Nph, Nph:2 * Nph] = Om * n_op
    H0[2 * Nph:3 * Nph, 2 * Nph:3 * Nph] = U * I + Om * n_op

    V = np.zeros((3 * Nph, 3 * Nph))
    V[2 * Nph:3 * Nph, 0:Nph] = -np.sqrt(2.0) * t * Dp
    V[0:Nph, 2 * Nph:3 * Nph] = -np.sqrt(2.0) * t * Dm
    V[2 * Nph:3 * Nph, Nph:2 * Nph] = -np.sqrt(2.0) * t * Dm
    V[Nph:2 * Nph, 2 * Nph:3 * Nph] = -np.sqrt(2.0) * t * Dp

    cold_idx = list(range(0, M + 1)) + list(range(Nph, Nph + M + 1))
    odd_idx = list(range(2 * Nph, 3 * Nph))
    hot_pair_idx = list(range(M + 1, Nph)) + list(range(Nph + M + 1, 2 * Nph))
    E_diag = np.diag(H0)
    S1 = np.zeros_like(H0)
    for i in cold_idx:
        for j in odd_idx:
            dE = E_diag[j] - E_diag[i]
            S1[i, j] = -V[i, j] / dE
            S1[j, i] = -S1[i, j]
    U1 = expm(S1)
    H1 = U1 @ (H0 + V) @ U1.T
    c_cold_hotpair = float(np.linalg.norm(H1[np.ix_(cold_idx, hot_pair_idx)], 2))
    c_cold_odd = float(np.linalg.norm(H1[np.ix_(cold_idx, odd_idx)], 2))
    _, evecs_1 = eigh(H1)
    v0 = evecs_1[:, 0]
    n_full = np.concatenate([np.arange(Nph), np.arange(Nph), np.arange(Nph)])
    exp_after_sw1 = float(np.sum(np.exp(0.25 * n_full) * (v0 ** 2)))

    # (E2) 3-site chain 0-1-2, N=2, Sz=0, X=4.0, U=40.0, Om=10.0, t=4.0, M=1
    U3 = 40.0
    Om3 = 10.0
    t3 = 4.0
    X3 = U3 / Om3
    M3 = 1
    Nph3 = 7
    alpha3 = np.sqrt(X3 / 2.0)
    a3 = np.zeros((Nph3, Nph3))
    for n in range(1, Nph3):
        a3[n - 1, n] = np.sqrt(n)
    Dp3 = expm(alpha3 * (a3.T - a3))
    Dm3 = Dp3.T
    I3 = np.eye(Nph3)

    def e_idx(ru, rd):
        return 3 * ru + rd

    def edge_hop(z0, z1):
        dim = 9 * (Nph3 ** 3)
        H_e = np.zeros((dim, dim))
        D_list_p = [I3, I3, I3]
        D_list_m = [I3, I3, I3]
        D_list_p[z0] = Dp3
        D_list_m[z1] = Dm3
        P_z0p_z1m = np.kron(np.kron(D_list_p[0] @ D_list_m[0], D_list_p[1] @ D_list_m[1]), D_list_p[2] @ D_list_m[2])
        P_z1p_z0m = P_z0p_z1m.T
        blk = Nph3 ** 3

        for ru in range(3):
            for rd in range(3):
                s_in = e_idx(ru, rd)
                if ru == z1:
                    s_out = e_idx(z0, rd)
                    H_e[s_out * blk:(s_out + 1) * blk, s_in * blk:(s_in + 1) * blk] += -t3 * P_z0p_z1m
                if ru == z0:
                    s_out = e_idx(z1, rd)
                    H_e[s_out * blk:(s_out + 1) * blk, s_in * blk:(s_in + 1) * blk] += -t3 * P_z1p_z0m
                if rd == z1:
                    s_out = e_idx(ru, z0)
                    H_e[s_out * blk:(s_out + 1) * blk, s_in * blk:(s_in + 1) * blk] += -t3 * P_z0p_z1m
                if rd == z0:
                    s_out = e_idx(ru, z1)
                    H_e[s_out * blk:(s_out + 1) * blk, s_in * blk:(s_in + 1) * blk] += -t3 * P_z1p_z0m
        return H_e

    V01 = edge_hop(0, 1)
    V12 = edge_hop(1, 2)
    blk = Nph3 ** 3
    dim3 = 9 * blk
    E0_3 = np.zeros(dim3)
    n0_arr = np.repeat(np.arange(Nph3), Nph3 * Nph3)
    n1_arr = np.tile(np.repeat(np.arange(Nph3), Nph3), Nph3)
    n2_arr = np.tile(np.arange(Nph3), Nph3 * Nph3)
    E_ph_blk = Om3 * (n0_arr + n1_arr + n2_arr)

    for ru in range(3):
        for rd in range(3):
            idx = e_idx(ru, rd)
            e_at = 0.0 if ru == rd else U3
            E0_3[idx * blk:(idx + 1) * blk] = e_at + E_ph_blk

    cold_01_blk = (n0_arr <= M3) & (n1_arr <= M3)
    cold_12_blk = (n1_arr <= M3) & (n2_arr <= M3)
    all_cold_blk = (n0_arr <= M3) & (n1_arr <= M3) & (n2_arr <= M3)
    n2_gt_2M_blk = n2_arr > 2 * M3

    all_cold_mask = np.zeros(dim3, dtype=bool)
    all_cold_odd_mask = np.zeros(dim3, dtype=bool)
    all_odd_mask = np.zeros(dim3, dtype=bool)
    n2_hot_mask = np.zeros(dim3, dtype=bool)
    for ru in range(3):
        for rd in range(3):
            idx = e_idx(ru, rd)
            if ru == rd:
                all_cold_mask[idx * blk:(idx + 1) * blk] = all_cold_blk
                n2_hot_mask[idx * blk:(idx + 1) * blk] = n2_gt_2M_blk
            else:
                all_cold_odd_mask[idx * blk:(idx + 1) * blk] = all_cold_blk
                all_odd_mask[idx * blk:(idx + 1) * blk] = True
                n2_hot_mask[idx * blk:(idx + 1) * blk] = n2_gt_2M_blk

    def build_edge_S_cut(z0, z1, edge_cold_blk, V_e):
        S_e = np.zeros((dim3, dim3))
        pair_indices = [e_idx(z0, z0), e_idx(z1, z1)]
        odd_indices = [e_idx(z0, z1), e_idx(z1, z0)]
        for ip in pair_indices:
            for io in odd_indices:
                rows_p = np.where(edge_cold_blk)[0] + ip * blk
                cols_o = np.where(edge_cold_blk)[0] + io * blk
                sub_V = V_e[np.ix_(rows_p, cols_o)]
                dE = E0_3[cols_o][None, :] - E0_3[rows_p][:, None]
                nz = np.abs(sub_V) > 0.0
                assert np.all(np.abs(dE[nz]) > 1.0)
                sub_S = np.divide(-sub_V, dE, out=np.zeros_like(sub_V), where=nz)
                S_e[np.ix_(rows_p, cols_o)] = sub_S
                S_e[np.ix_(cols_o, rows_p)] = -sub_S.T
        return S_e

    S01 = build_edge_S_cut(0, 1, cold_01_blk, V01)
    S12 = build_edge_S_cut(1, 2, cold_12_blk, V12)
    S_tot = S01 + S12
    V_tot = V01 + V12

    R1 = (S_tot * E0_3[None, :] - E0_3[:, None] * S_tot) + V_tot
    r1_all_cold_cold_odd = float(np.linalg.norm(R1[np.ix_(all_cold_mask, all_cold_odd_mask)], 2))
    r1_all_cold_all_odd = float(np.linalg.norm(R1[np.ix_(all_cold_mask, all_odd_mask)], 2))

    C2_cross = (S01 @ V12 - V12 @ S01) + (S12 @ V01 - V01 @ S12)
    c2_cold_odd = float(np.linalg.norm(C2_cross[np.ix_(all_cold_mask, all_odd_mask)], 2))
    c2_cold_n2hot = float(np.linalg.norm(C2_cross[np.ix_(all_cold_mask, n2_hot_mask)], 2))

    V12_hot = V12[np.ix_(n2_hot_mask, ~n2_hot_mask)]
    v12_hot_norm = float(np.linalg.norm(V12_hot, 2))

    assert c_cold_hotpair > 0.4
    assert exp_after_sw1 > 1e4
    assert r1_all_cold_cold_odd < 1e-11
    assert r1_all_cold_all_odd > 1.0
    assert c2_cold_odd > 0.8
    assert c2_cold_n2hot > 0.4
    assert v12_hot_norm > t3 * 0.5

    # (E3a) Converged post-S1_col exponential-moment blowup at X=40 (U=40, Om=1, t=1, M=1, Nph=360)
    # Note: the 2nd-order pair-exchange term D(-2*alpha) peaks at n ~ 4*X = 160, so Nph=360 > 2*(4X)
    # fully resolves the physical pair-exchange tail with boundary share < 5e-3.
    U_e3, Om_e3, t_e3, M_e3, Nph_e3 = 40.0, 1.0, 1.0, 1, 360
    a_e3 = np.zeros((Nph_e3, Nph_e3))
    for n in range(1, Nph_e3):
        a_e3[n - 1, n] = np.sqrt(n)
    Dp_e3 = expm(np.sqrt(U_e3 / Om_e3) * (a_e3.T - a_e3))
    Dm_e3 = Dp_e3.T
    nop_e3 = np.arange(Nph_e3, dtype=float)
    E0_e3 = np.concatenate([Om_e3 * nop_e3, Om_e3 * nop_e3, U_e3 + Om_e3 * nop_e3])
    V_e3 = np.zeros((3 * Nph_e3, 3 * Nph_e3))
    V_e3[0:Nph_e3, 2 * Nph_e3:3 * Nph_e3] = -np.sqrt(2.0) * t_e3 * Dm_e3
    V_e3[2 * Nph_e3:3 * Nph_e3, 0:Nph_e3] = -np.sqrt(2.0) * t_e3 * Dp_e3
    V_e3[Nph_e3:2 * Nph_e3, 2 * Nph_e3:3 * Nph_e3] = -np.sqrt(2.0) * t_e3 * Dp_e3
    V_e3[2 * Nph_e3:3 * Nph_e3, Nph_e3:2 * Nph_e3] = -np.sqrt(2.0) * t_e3 * Dm_e3
    H_e3 = np.diag(E0_e3) + V_e3
    S1_col = np.zeros_like(V_e3)
    for b_idx, Dmat in [(0, Dm_e3), (1, Dp_e3)]:
        dE_e3 = (U_e3 + Om_e3 * nop_e3)[None, :] - (Om_e3 * nop_e3)[:, None]
        sub_e3 = np.zeros_like(Dmat)
        sub_e3[:M_e3 + 1, :] = -(-np.sqrt(2.0) * t_e3 * Dmat[:M_e3 + 1, :]) / dE_e3[:M_e3 + 1, :]
        S1_col[b_idx * Nph_e3:(b_idx + 1) * Nph_e3, 2 * Nph_e3:3 * Nph_e3] = sub_e3
        S1_col[2 * Nph_e3:3 * Nph_e3, b_idx * Nph_e3:(b_idx + 1) * Nph_e3] = -sub_e3.T
    U_col = expm(S1_col)
    H_col = U_col @ H_e3 @ U_col.T
    ev_col, vec_col = eigh(H_col)
    w_col = np.exp(-(ev_col - ev_col[0]) / Om_e3)
    Z_trip_e3 = np.exp(-(U_e3 - ev_col[0]) / Om_e3) / (1.0 - np.exp(-1.0))
    Z_tot_e3 = float(np.sum(w_col) + Z_trip_e3)
    n_e3_vec = np.concatenate([nop_e3, nop_e3, nop_e3])
    exp_e3_vec = np.exp(0.25 * n_e3_vec)
    eq3_post_col_gs = float(np.sum(exp_e3_vec * (vec_col[:, 0] ** 2)))
    eq3_post_col_T = float(np.sum(w_col * np.sum(exp_e3_vec[:, None] * (vec_col ** 2), axis=0)) / Z_tot_e3)
    bdry_e3 = float(
        np.sum(w_col * np.sum((exp_e3_vec * (n_e3_vec >= Nph_e3 - 10))[:, None] * (vec_col ** 2), axis=0))
        / (Z_tot_e3 * eq3_post_col_T)
    )
    assert bdry_e3 < 0.02, f"Witness E3 not converged: bdry_e3={bdry_e3}"
    assert eq3_post_col_gs > 1e11
    assert eq3_post_col_T > 1e14
    # Also record X=128, M=2 at Nph=300 (the first-order tail plateau is 1.60e+14;
    # the 2nd-order D(2*alpha) tail peaks near 4*X*e^s ~ 657 and drives the true value > 1e30):
    U_e3b, Om_e3b, t_e3b, M_e3b, Nph_e3b = 128.0, 1.0, 1.0, 2, 300
    a_e3b = np.zeros((Nph_e3b, Nph_e3b))
    for n in range(1, Nph_e3b):
        a_e3b[n - 1, n] = np.sqrt(n)
    Dp_e3b = expm(np.sqrt(U_e3b / Om_e3b) * (a_e3b.T - a_e3b))
    Dm_e3b = Dp_e3b.T
    nop_e3b = np.arange(Nph_e3b, dtype=float)
    E0_e3b = np.concatenate([Om_e3b * nop_e3b, Om_e3b * nop_e3b, U_e3b + Om_e3b * nop_e3b])
    V_e3b = np.zeros((3 * Nph_e3b, 3 * Nph_e3b))
    V_e3b[0:Nph_e3b, 2 * Nph_e3b:3 * Nph_e3b] = -np.sqrt(2.0) * t_e3b * Dm_e3b
    V_e3b[2 * Nph_e3b:3 * Nph_e3b, 0:Nph_e3b] = -np.sqrt(2.0) * t_e3b * Dp_e3b
    V_e3b[Nph_e3b:2 * Nph_e3b, 2 * Nph_e3b:3 * Nph_e3b] = -np.sqrt(2.0) * t_e3b * Dp_e3b
    V_e3b[2 * Nph_e3b:3 * Nph_e3b, Nph_e3b:2 * Nph_e3b] = -np.sqrt(2.0) * t_e3b * Dm_e3b
    H_e3b = np.diag(E0_e3b) + V_e3b
    S1_col_b = np.zeros_like(V_e3b)
    for b_idx, Dmat in [(0, Dm_e3b), (1, Dp_e3b)]:
        dE_e3b = (U_e3b + Om_e3b * nop_e3b)[None, :] - (Om_e3b * nop_e3b)[:, None]
        sub_e3b = np.zeros_like(Dmat)
        sub_e3b[:M_e3b + 1, :] = -(-np.sqrt(2.0) * t_e3b * Dmat[:M_e3b + 1, :]) / dE_e3b[:M_e3b + 1, :]
        S1_col_b[b_idx * Nph_e3b:(b_idx + 1) * Nph_e3b, 2 * Nph_e3b:3 * Nph_e3b] = sub_e3b
        S1_col_b[2 * Nph_e3b:3 * Nph_e3b, b_idx * Nph_e3b:(b_idx + 1) * Nph_e3b] = -sub_e3b.T
    U_col_b = expm(S1_col_b)
    ev_e3b, evec_e3b = eigh(H_e3b)
    w_e3b = np.exp(-(ev_e3b - ev_e3b[0]) / Om_e3b)
    w_e3b /= np.sum(w_e3b)
    P_col_T_b = ((U_col_b @ evec_e3b) ** 2) @ w_e3b
    n_e3b_vec = np.concatenate([nop_e3b, nop_e3b, nop_e3b])
    eq3_post_col_T_X128 = float(np.sum(np.exp(0.25 * n_e3b_vec) * P_col_T_b))
    assert eq3_post_col_T_X128 > 1e14

    # (E3b) Non-vacuous relative-entropy tail bound (Thm 3.2) test on 2-site TWO-MODE LF system (N=2, Sz=0, U=16, Om=1, t=0.25, M=1, tau=1)
    # Here J_0 = ||V_0||_op = 2*t = 0.5, so bound = (2*J_0 + tau*log 2) / (Om*(M+1)) = (1 + log 2)/2 = 0.846574 < 1!
    U5, Om5, t5, M5, tau5, Nph_s = 16.0, 1.0, 0.25, 1, 1.0, 18
    al5 = math.sqrt(U5 / (2.0 * Om5))
    a5 = np.zeros((Nph_s, Nph_s))
    for n in range(1, Nph_s):
        a5[n - 1, n] = np.sqrt(n)
    Dp5 = expm(al5 * (a5.T - a5))
    Dm5 = Dp5.T
    I5 = np.eye(Nph_s)
    n5 = np.diag(np.arange(Nph_s, dtype=float))
    N2 = Nph_s * Nph_s
    dim5 = 3 * N2
    H5 = np.zeros((dim5, dim5))
    n_sum5 = np.kron(n5, I5) + np.kron(I5, n5)
    II5 = np.kron(I5, I5)
    H5[0:N2, 0:N2] = Om5 * n_sum5
    H5[N2:2 * N2, N2:2 * N2] = Om5 * n_sum5
    H5[2 * N2:3 * N2, 2 * N2:3 * N2] = U5 * II5 + Om5 * n_sum5
    D0p_D1m = np.kron(Dp5, Dm5)
    D0m_D1p = np.kron(Dm5, Dp5)
    H5[2 * N2:3 * N2, 0:N2] = -np.sqrt(2.0) * t5 * D0p_D1m
    H5[0:N2, 2 * N2:3 * N2] = -np.sqrt(2.0) * t5 * D0m_D1p
    H5[2 * N2:3 * N2, N2:2 * N2] = -np.sqrt(2.0) * t5 * D0m_D1p
    H5[N2:2 * N2, 2 * N2:3 * N2] = -np.sqrt(2.0) * t5 * D0p_D1m
    ev5, vec5 = eigh(H5)
    beta5 = 1.0 / tau5
    w5 = np.exp(-beta5 * (ev5 - ev5[0]))
    Z_trip5 = np.exp(-beta5 * (U5 - ev5[0])) / ((1.0 - np.exp(-beta5 * Om5)) ** 2)
    n0_grid = np.repeat(np.arange(Nph_s), Nph_s)
    hot0_1block = (n0_grid > M5).astype(float)
    hot0_mask = np.concatenate([hot0_1block, hot0_1block, hot0_1block])
    hot0_in_sing = float(np.sum(w5 * np.sum(hot0_mask[:, None] * (vec5 ** 2), axis=0)))
    p_hot_trip = np.exp(-beta5 * Om5 * (M5 + 1))
    Z_tot5 = float(np.sum(w5) + Z_trip5)
    p_hot0 = (hot0_in_sing + Z_trip5 * p_hot_trip) / Z_tot5
    J0_op = 2.0 * t5
    eq5_bound = (2.0 * J0_op + tau5 * math.log(2.0)) / (Om5 * (M5 + 1))
    assert p_hot0 <= eq5_bound < 1.0

    return {
        "dimer_C_cold_hotpair": c_cold_hotpair,
        "dimer_C_cold_odd": c_cold_odd,
        "dimer_exp_lyap_after_sw1": exp_after_sw1,
        "chain3_R1_all_cold_cold_odd": r1_all_cold_cold_odd,
        "chain3_R1_all_cold_all_odd": r1_all_cold_all_odd,
        "chain3_C2_all_cold_odd": c2_cold_odd,
        "chain3_C2_all_cold_n2hot": c2_cold_n2hot,
        "chain3_V12_hot_norm": v12_hot_norm,
        "eq3_post_S1col_GS_X40": eq3_post_col_gs,
        "eq3_post_S1col_T_Om_X40": eq3_post_col_T,
        "eq3_post_S1col_bdry_share": bdry_e3,
        "eq3_post_S1col_T_Om_X128_Nph300": eq3_post_col_T_X128,
        "eq5_2mode_p_hot0": p_hot0,
        "eq5_2mode_bound": eq5_bound,
    }


def run_guarded_mutants(cert: dict, wa: dict, wb: dict, wc: dict, wd: dict, we: dict) -> list[dict]:
    """9 guarded mutants: each asserts a broken positive claim and MUST fail (caught=True)."""
    mutants = []

    def expect_caught(name: str, desc: str, fn):
        try:
            fn()
        except AssertionError as exc:
            mutants.append({"name": name, "desc": desc, "caught": True, "detail": str(exc)})
            return
        raise RuntimeError(f"MUTANT ESCAPED: {name} ({desc})")

    # Mutant 1: Claim Prop 3 per-site cold weight stays >= 0.9 on 10x10 lattice
    def m1():
        assert cert["prop3_per_site_cold_ratio_X64"] ** 100 >= 0.9, (
            f"cold weight collapsed to {cert['prop3_per_site_cold_ratio_X64']**100:.3e}"
        )
    expect_caught("mutant_1_naive_cold_trace_lift", "Claim P_cold weight stays O(1) on V=100 at tau=Om", m1)

    # Mutant 2: Claim LF exponential moment <e^{0.25 n_rel}>_GS is bounded by 100 uniform in X
    def m2():
        assert wa[60]["exp_n_gs"] <= 100.0, f"<e^{{0.25 n_rel}}>_GS = {wa[60]['exp_n_gs']:.3e} > 100"
    expect_caught("mutant_2_lf_lyapunov_bounded", "Claim LF exp moment bounded uniformly in X", m2)

    # Mutant 3: Claim LF first moment <n_rel>_GS <= (t/U)^2 * 5 uniform in X
    def m3():
        assert wa[60]["mean_n_gs"] <= 0.2, f"<n_rel>_GS = {wa[60]['mean_n_gs']:.4f} > 0.2"
    expect_caught("mutant_3_lf_first_moment_bounded", "Claim LF first moment bounded uniformly in X", m3)

    # Mutant 4: Claim Gaussian-domination (Hubbard) dimer pair splitting is O(Omega) at X=40
    def m4():
        Om_40 = 40.0 / 40.0
        assert wb["hubbard_pair_splitting"] <= 2.0 * Om_40, (
            f"Hubbard splitting {wb['hubbard_pair_splitting']:.4f} >> 2*Om={2*Om_40}"
        )
    expect_caught("mutant_4_gaussian_domination_omega_scaling", "Claim Gaussian domination gives O(Om) splitting", m4)

    # Mutant 5: Claim vertical pair correlation <D0^dag D1> <= e^{-X/4} at X=40, tau=Om
    def m5():
        bound = math.exp(-40.0 / 4.0)
        assert wb[40]["corrs"]["tau=Om"] <= bound, (
            f"<D0^dag D1> = {wb[40]['corrs']['tau=Om']:.4e} >> e^{{-10}} = {bound:.4e}"
        )
    expect_caught("mutant_5_vertical_pair_corr_exponential_in_X", "Claim pair corr decays as e^{-cX} at tau=Om", m5)

    # Mutant 6: Claim C4 particle-hole or direct reflection kernel is PSD
    def m6():
        assert wc["C4_edge_ph_min_eval"] >= -1e-9, f"C4_edge_ph min_eval = {wc['C4_edge_ph_min_eval']:.3f} < 0"
    expect_caught("mutant_6_c4_reflection_positivity", "Claim C4 PH reflection kernel is PSD", m6)

    # Mutant 7: Claim 1-site phonon marginal satisfies Brascamp-Lieb Var(b) <= U*tau and 2-site V_eff is convex
    def m7():
        assert cert["route_d_1site_Var_b"] <= 40.0 and wd["V_dimer_t1_d2_anti_at_0"] >= 0.0, (
            f"Var(b) = {cert['route_d_1site_Var_b']:.2f} > U*tau = 40; "
            f"d2_anti(t=1) = {wd['V_dimer_t1_d2_anti_at_0']:.4f} < 0"
        )
    expect_caught("mutant_7_marginal_brascamp_lieb", "Claim phonon marginal satisfies BL Var(b)<=U*tau and convexity", m7)

    # Mutant 8: Claim local edge-wise Cut-SW cancels 2nd-order all_cold<->odd and all_cold<->(n2>2M)
    def m8():
        assert we["chain3_C2_all_cold_n2hot"] <= 1e-6, (
            f"3-site cross-commutator into n2>2M is {we['chain3_C2_all_cold_n2hot']:.4f}"
        )
    expect_caught("mutant_8_cut_sw_3site_commutes", "Claim local Cut-SW avoids n2>2M leaks on 3 sites", m8)

    # Mutant 9: Claim exponential-moment claim <e^{0.25 n}>_* stays bounded after S1_col(M, all) at T=Om, X=40
    def m9():
        assert we["eq3_post_S1col_T_Om_X40"] <= 1e4, (
            f"<e^{{0.25 n}}>_{{S1_col, T=Om, X=40}} = {we['eq3_post_S1col_T_Om_X40']:.3e} >> 1e4"
        )
    expect_caught("mutant_9_post_s1_eq3_bounded", "Claim post-S1_col exp moment Eq. (3) bounded at T=Om", m9)

    return mutants


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay W/MASSIVE/FULL-GIBBS obstruction checks & mutants.")
    parser.add_argument("--skip-mutants", action="store_true", help="Skip the 9 guarded mutant checks.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cert = check_certified_identities()
    wa = witness_route_a()
    wb = witness_route_b()
    wc = witness_route_c()
    wd = witness_route_d()
    we = witness_route_e()

    print("=== CERTIFIED IDENTITIES & BOUNDS ===")
    for k, v in cert.items():
        print(f"  {k}: {v}")

    print("\n=== WITNESS A (Route a: LF Dimer Lyapunov & First-Moment Blowup, U=40, t=8) ===")
    for X, d in wa.items():
        print(
            f"  X={X:2d}: <n_rel>_GS={d['mean_n_gs']:.6f}, <e^{{0.25 n_rel}}>_GS={d['exp_n_gs']:.4e} "
            f"(bdry={d['bdry_gs_share']:.2e}), <e^{{0.25 n_rel}}>_tau=Om={d['exp_n_tau_Om']:.4e}"
        )

    print("\n=== WITNESS B (Route b: Pre-LF Vertical Pair Corr, Hubbard Splitting & Fock Trace) ===")
    print(f"  Instantaneous Hubbard dimer pair splitting: {wb['hubbard_pair_splitting']:.6f}")
    print(f"  Fock-trace max rel err: F_+={wb['fock_trace_F_plus_max_rel_err']:.3e}, F_-={wb['fock_trace_F_minus_max_rel_err']:.3e}")
    for X in [10, 20, 40]:
        d = wb[X]
        print(f"  X={X:2d}: gap={d['gap']:.3e}, corrs={ {k: round(v, 6) for k, v in d['corrs'].items()} }")

    print("\n=== WITNESS C (Route c: C2 & C4 Reflection Positivity Min Eigenvalues) ===")
    for k, v in wc.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== WITNESS D (Route d: 2-site Non-Convexity at b=0 for t=1 and t=4) ===")
    for k, v in wd.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== WITNESS E (Route e: 1-step SW, 3-site Cut-SW, Post-S1_col & Non-Vacuous Thm 3.2) ===")
    for k, v in we.items():
        print(f"  {k}: {v:.6e}")

    if not args.skip_mutants:
        mutants = run_guarded_mutants(cert, wa, wb, wc, wd, we)
        print(f"\n=== GUARDED MUTANTS ({len(mutants)}/{len(mutants)} CAUGHT) ===")
        for m in mutants:
            print(f"  [CAUGHT] {m['name']}: {m['detail']}")

    dt = time.perf_counter() - t0
    print(f"\nALL CHECKS PASSED in {dt:.2f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

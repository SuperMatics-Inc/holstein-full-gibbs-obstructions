#!/usr/bin/env python3
"""
Replay and mutation verification for W/MASSIVE/DISTORTION-VALLEY/ (Strategy III).
Usage:
    python replay.py
    python replay.py --mutants
"""
import argparse
import math
import sys
import time
import numpy as np
import mpmath as mp
import sympy as sp
from scipy.linalg import eigh
from scipy.integrate import quad

mp.mp.dps = 35
_MP_ED_CACHE = {}


def c_wkb(x: float, kap: float) -> float:
    if kap >= 1.0:
        return 0.0
    rhs = 1.0 - math.sqrt(kap) * (1.0 - x)
    if rhs <= x:
        return 0.0
    s_star = math.sqrt(rhs * rhs - x * x)
    val, _ = quad(
        lambda s: math.sqrt(max(0.0, (1.0 - math.sqrt(s * s + x * x)) ** 2 - kap * (1.0 - x) ** 2)),
        0.0,
        s_star,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    return 4.0 * val


def c_wkb_zero_closed(x: float) -> float:
    rt = math.sqrt(1.0 - x * x)
    return 2.0 * rt - 2.0 * x * x * math.log((1.0 + rt) / x)


def thm1_ground_thresholds(x: float):
    kap_p = 1.0 / 8.0
    rhs = 1.0 - math.sqrt(kap_p) * (1.0 - x)
    s_star = math.sqrt(rhs * rhs - x * x)
    cw = c_wkb(x, kap_p)
    X0 = max(8.0 / ((1.0 - x) ** 2), 4.0 / (s_star ** 2), 64.0 * ((1.0 - x) ** 2) / (cw ** 2))
    return X0, 0.5 * cw, 1.0


def thm1_general_thresholds(x: float, kap: float):
    kap_p = 0.5 * (1.0 + kap)
    rhs = 1.0 - math.sqrt(kap_p) * (1.0 - x)
    s_star = math.sqrt(rhs * rhs - x * x)
    cw = c_wkb(x, kap_p)
    A_pref = 352.0 / (7.0 * (1.0 - kap) * ((1.0 - x) ** 2))
    X0 = max(
        2.0 / ((1.0 - kap) * ((1.0 - x) ** 2)),
        4.0 / (s_star ** 2),
        64.0 * ((1.0 - x) ** 2) / (cw ** 2),
        (32.0 / (9.0 * cw)) * math.log(max(1.0, 128.0 * A_pref / (9.0 * math.e * cw))),
    )
    C_val = 160.0 * A_pref / (math.e * cw)
    return X0, 0.5 * cw, C_val


def build_ho_dimer_hr(U: float, t: float, Om: float, n_ho: int = 120):
    """
    Build H_r in the harmonic oscillator basis of H_0 = -(U*Om^2/4) d^2/dr^2 + r^2/U - Om/2.
    Oscillator length scale: r = ell * q_dimless with q_dimless = (a + a^dag)/sqrt(2),
    ell = sqrt(U * Om / 2). Note sqrt(k_eff/m_eff) = sqrt((2/U)/(2/(U*Om^2))) = Om.
    So H_r = Om * diag(0, 1, ..., n_ho-1) x I_3 + H_el_sing(r_op).
    """
    ell = math.sqrt(U * Om / 2.0)
    idx = np.arange(1, n_ho, dtype=float)
    off = np.sqrt(idx / 2.0)
    q = np.diag(off, 1) + np.diag(off, -1)
    r_op = ell * q

    h_diag = Om * np.diag(np.arange(n_ho, dtype=float))
    I3 = np.eye(3)
    I_ho = np.eye(n_ho)

    # Basis {|+>, |s>, |->}
    T_el = np.array([[0.0, -2.0 * t, 0.0], [-2.0 * t, 0.0, 0.0], [0.0, 0.0, 0.0]])
    D_el = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    H_r = np.kron(h_diag, I3) + np.kron(I_ho, T_el) + np.kron(r_op, D_el)
    return H_r, r_op


def _build_parity_block(U: float, t: float, Om: float, n_ho: int, target_parity: int):
    """Build float64 parity block (-1)^n * S = target_parity of H_r."""
    ell = math.sqrt(U * Om / 2.0)
    basis = [
        (n, a)
        for n in range(n_ho)
        for a, s_a in [(0, +1), (1, +1), (2, -1)]
        if (1 if n % 2 == 0 else -1) * s_a == target_parity
    ]
    idx_map = {b: i for i, b in enumerate(basis)}
    dim = len(basis)
    Hf = np.zeros((dim, dim), dtype=float)
    for n in range(n_ho):
        if (n, 0) in idx_map and (n, 1) in idx_map:
            i0, i1 = idx_map[(n, 0)], idx_map[(n, 1)]
            Hf[i0, i0] = Hf[i1, i1] = Om * n
            Hf[i0, i1] = Hf[i1, i0] = -2.0 * t
        if (n, 2) in idx_map:
            Hf[idx_map[(n, 2)], idx_map[(n, 2)]] = Om * n
    for n in range(n_ho - 1):
        ro = ell * math.sqrt((n + 1) / 2.0)
        for (a, b) in [(0, 2), (2, 0)]:
            if (n, a) in idx_map and (n + 1, b) in idx_map:
                ia, ib = idx_map[(n, a)], idx_map[(n + 1, b)]
                Hf[ia, ib] = Hf[ib, ia] = 2.0 * ro
    return Hf, idx_map, dim


def _solve_parity_kth_mp(U: float, t: float, Om: float, n_ho: int, target_parity: int, shift_f: float, v0_f: np.ndarray, n_steps: int = 5):
    """Banded mpmath shifted inverse iteration for a single parity eigenstate (O(n_ho) cost)."""
    U_m, t_m, Om_m = mp.mpf(U), mp.mpf(t), mp.mpf(Om)
    ell = mp.sqrt(U_m * Om_m / 2)
    basis = [
        (n, a)
        for n in range(n_ho)
        for a, s_a in [(0, +1), (1, +1), (2, -1)]
        if (1 if n % 2 == 0 else -1) * s_a == target_parity
    ]
    idx_map = {b: i for i, b in enumerate(basis)}
    dim = len(basis)
    rows = [{} for _ in range(dim)]
    for n in range(n_ho):
        if (n, 0) in idx_map and (n, 1) in idx_map:
            i0, i1 = idx_map[(n, 0)], idx_map[(n, 1)]
            rows[i0][i0] = rows[i1][i1] = Om_m * n
            rows[i0][i1] = rows[i1][i0] = -2 * t_m
        if (n, 2) in idx_map:
            rows[idx_map[(n, 2)]][idx_map[(n, 2)]] = Om_m * n
    for n in range(n_ho - 1):
        ro = ell * mp.sqrt(mp.mpf(n + 1) / 2)
        for (a, b) in [(0, 2), (2, 0)]:
            if (n, a) in idx_map and (n + 1, b) in idx_map:
                ia, ib = idx_map[(n, a)], idx_map[(n + 1, b)]
                rows[ia][ib] = rows[ib][ia] = 2 * ro
    bw = max(abs(j - i) for i in range(dim) for j in rows[i])
    mu = mp.mpf(shift_f)
    piv_rows = [dict((j, rows[i][j] - (mu if i == j else mp.mpf(0))) for j in rows[i]) for i in range(dim)]
    for i in range(dim):
        if i not in piv_rows[i]:
            piv_rows[i][i] = -mu
    L_ops = []
    for k in range(dim - 1):
        p_row = k
        best = abs(piv_rows[k].get(k, mp.mpf(0)))
        for r in range(k + 1, min(dim, k + bw + 1)):
            val = abs(piv_rows[r].get(k, mp.mpf(0)))
            if val > best:
                best = val
                p_row = r
        if p_row != k:
            piv_rows[k], piv_rows[p_row] = piv_rows[p_row], piv_rows[k]
            L_ops.append(("swap", k, p_row))
        akk = piv_rows[k][k]
        for r in range(k + 1, min(dim, k + 2 * bw + 1)):
            ark = piv_rows[r].get(k, mp.mpf(0))
            if ark != 0:
                fac = ark / akk
                L_ops.append(("elim", r, k, fac))
                for c, akc in piv_rows[k].items():
                    if c > k:
                        piv_rows[r][c] = piv_rows[r].get(c, mp.mpf(0)) - fac * akc
                piv_rows[r].pop(k, None)
    v = [mp.mpf(float(val)) for val in v0_f]
    for _ in range(n_steps):
        w = list(v)
        for op in L_ops:
            if op[0] == "swap":
                w[op[1]], w[op[2]] = w[op[2]], w[op[1]]
            else:
                w[op[1]] -= op[3] * w[op[2]]
        x_vec = [mp.mpf(0)] * dim
        for k in range(dim - 1, -1, -1):
            s = w[k]
            for c, akc in piv_rows[k].items():
                if c > k:
                    s -= akc * x_vec[c]
            x_vec[k] = s / piv_rows[k][k]
        nrm = mp.sqrt(sum(xi * xi for xi in x_vec))
        v = [xi / nrm for xi in x_vec]
    Hv = [sum(val * v[j] for j, val in rows[i].items()) for i in range(dim)]
    rq = sum(v[i] * Hv[i] for i in range(dim))
    return rq


def build_ho_dimer_hr_mp(U: float, t: float, Om: float, n_ho: int = 120):
    """High-precision lowest doublet of H_r via float64 seeding + banded mpmath inverse iteration."""
    key = (U, t, Om, n_ho)
    if key in _MP_ED_CACHE:
        return _MP_ED_CACHE[key]
    low_mp = []
    all_f = []
    for p in (+1, -1):
        Hf, _, _ = _build_parity_block(U, t, Om, n_ho, p)
        ev_f, evec_f = eigh(Hf)
        all_f.extend(ev_f.tolist())
        rq = _solve_parity_kth_mp(U, t, Om, n_ho, p, float(ev_f[0]) - 1e-6, evec_f[:, 0])
        low_mp.append(rq)
    low_mp.sort()
    all_f.sort()
    res = (all_f, low_mp)
    _MP_ED_CACHE[key] = res
    return res


def check_1_certified_conventions_and_identities(mutant: str = "none"):
    print("[Check 1] Exact conventions, BO surfaces, identities, and explicit Theorem 1 thresholds...")
    U, Om, t, r, c = sp.symbols("U Om t r c", positive=True, real=True)
    # 1. Bipolaron binding energy U = 2 g^2 / Om
    g_expr = sp.sqrt(U * Om / 2)
    if mutant == "mutant_1_sign_g":
        g_expr = sp.sqrt(U * Om)  # missing factor 1/2
    E_20 = -2 * (g_expr ** 2) / Om
    E_11 = sp.Integer(0)
    assert sp.simplify(E_11 - E_20 - U) == 0, f"Bipolaron binding energy mismatch: {E_11 - E_20}"

    # 2. Electronic matrix and Delta_1^dag Delta_2 in {|+>, |s>, |->}
    H_el = sp.Matrix([[0, -2 * t, 2 * r], [-2 * t, 0, 0], [2 * r, 0, 0]])
    lam = sp.Symbol("lam")
    char_poly = sp.factor(H_el.charpoly(lam).as_expr())
    assert sp.simplify(char_poly - lam * (lam ** 2 - 4 * (r ** 2 + t ** 2))) == 0

    # Pair-transfer operator |2,0><0,2| in {|+>, |s>, |->}
    P20_02 = sp.Rational(1, 2) * sp.Matrix([[1, 0, -1], [0, 0, 0], [1, 0, -1]])
    if mutant == "mutant_5_delta1_delta2_diagonal":
        P20_02 = sp.Rational(1, 2) * sp.Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    ket_plus = sp.Matrix([1, 0, 0])
    ket_minus = sp.Matrix([0, 0, 1])
    ket_20 = (ket_plus + ket_minus) / sp.sqrt(2)
    ket_02 = (ket_plus - ket_minus) / sp.sqrt(2)
    assert P20_02 * ket_02 == ket_20, "Delta_1^dag Delta_2 failed to map |0,2> -> |2,0>"
    assert P20_02 * ket_20 == sp.zeros(3, 1), "Delta_1^dag Delta_2 failed to annihilate |2,0>"

    # 3. BO barrier B_dimer = (U-t)^2 / U
    _ = (c ** 2 + r ** 2) / U - 2 * sp.sqrt(r ** 2 + t ** 2)
    r0_sq = U ** 2 - t ** 2
    V_min = r0_sq / U - 2 * U
    V_bar = -2 * t
    B_calc = sp.simplify(V_bar - V_min)
    B_target = (U - t) ** 2 / U
    if mutant == "mutant_2_barrier_missing_t2":
        B_target = U - 2 * t
    assert sp.simplify(B_calc - B_target) == 0, f"Barrier mismatch: {B_calc} vs {B_target}"

    # 4. Closed-form WKB c_WKB(x, 0) vs quadrature, and global quadratic lower bound (3.9)
    for x in [0.01, 0.05, 0.1, 0.2, 0.25]:
        cw_num = c_wkb(x, 0.0)
        cw_cls = c_wkb_zero_closed(x)
        assert abs(cw_num - cw_cls) < 1e-11, f"WKB closed-form mismatch at x={x}: {cw_num} vs {cw_cls}"
        assert cw_cls >= 1.678562, f"c_WKB({x}, 0) below minimum"
        if mutant == "mutant_3_lf_wrong_exponent":
            assert cw_cls >= 2.0, f"False claim: c_WKB({x}, 0) = {cw_cls} < 2"

        # Check global quadratic lower bound (3.9)
        r0 = math.sqrt(1.0 - x * x)
        rs = np.linspace(-3.0, 3.0, 20001)
        lhs = rs ** 2 - 2.0 * np.sqrt(rs ** 2 + x * x) + (1.0 + x * x)
        rhs = ((1.0 - x) / (1.0 + x)) * np.minimum((rs - r0) ** 2, (rs + r0) ** 2)
        assert np.min(lhs - rhs) >= -1e-14

        # Verify explicit Theorem 1 thresholds
        X0_g, c0_g, C0_g = thm1_ground_thresholds(x)
        assert X0_g <= 31.0 and c0_g >= 0.655
        # Verify C_exact <= C0_g at X0_g
        cw_p = 2.0 * c0_g
        M_pref = 64.0 / (7.0 * ((1.0 - x) ** 2) * X0_g)
        two_M = 2.0 * M_pref * math.exp(-(9.0 / 16.0) * cw_p * X0_g)
        assert two_M < 1e-5
        C_ex = 4.5 * M_pref / (X0_g * (1.0 - two_M))
        assert C_ex <= C0_g

        for kap in [0.0, 0.25, 0.5, 0.75]:
            X0_k, ck, _Ck = thm1_general_thresholds(x, kap)
            cw_k = 2.0 * ck
            A_pref = 352.0 / (7.0 * (1.0 - kap) * ((1.0 - x) ** 2))
            two_Mk = 2.0 * A_pref * X0_k * math.exp(-(9.0 / 16.0) * cw_k * X0_k)
            assert two_Mk <= 0.5, f"2M_eta={two_Mk} > 0.5 at x={x}, kap={kap}"
            # Verify Golden-Thompson sub-barrier counting bound (3.10): k+1 <= 11*X
            for X_test in [9.0, X0_k]:
                bound_2kp1 = 6.0 * math.e * math.sqrt((1.0 + x) / (1.0 - x)) * (kap * ((1.0 - x) ** 2) * X_test + 0.5)
                assert 0.5 * (bound_2kp1 + 1.0) <= 11.0 * X_test, f"GT count exceeded 11X at x={x}, kap={kap}, X={X_test}"

    # 5. Check N=1,3 BO minimum formula for U > 2t and t < U <= 2t
    for U_val, t_val in [(20.0, 2.0), (20.0, 9.0), (20.0, 10.0), (20.0, 15.0)]:
        rs_test = np.linspace(-2.0 * U_val, 2.0 * U_val, 40001)
        v1_min_num = float(np.min(-U_val / 4.0 + rs_test ** 2 / U_val - np.sqrt(rs_test ** 2 + t_val ** 2)))
        v1_min_exact = (-U_val / 2.0 - (t_val ** 2) / U_val) if U_val > 2.0 * t_val else (-U_val / 4.0 - t_val)
        assert abs(v1_min_num - v1_min_exact) < 1e-6, f"N=1,3 BO min mismatch: {v1_min_num} vs {v1_min_exact}"
    print("  -> PASS (all symbolic, WKB, quadratic, N=1,3 BO, and threshold checks verified)")


def check_2_dimer_ed_splitting(mutant: str = "none"):
    print("[Check 2] High-precision HO-basis ED vs Theorem 1 sub-barrier splitting...")
    # Test ground doublet across X = U/Om in [10, 30] and x = t/U in [0.1, 0.25]
    # Note: for X <= 30, n_ho = 110 in mpmath converges E_1 - E_0 to < 1e-30!
    cases = [
        (20.0, 2.0, 2.0),   # X=10, x=0.1
        (20.0, 2.0, 1.0),   # X=20, x=0.1
        (20.0, 5.0, 1.0),   # X=20, x=0.25
        (30.0, 3.0, 1.0),   # X=30, x=0.1
        (30.0, 7.5, 1.0),   # X=30, x=0.25
    ]
    for U, t, Om in cases:
        x = t / U
        X = U / Om
        V_min = -U - (t * t) / U
        n_ho = max(100, int(4.5 * X) + 25)
        evals_f, evals_mp = build_ho_dimer_hr_mp(U, t, Om, n_ho=n_ho)
        E0, E1 = evals_mp[0], evals_mp[1]
        dE0 = float(E1 - E0)
        assert float(E0) <= V_min + 1e-12, f"E0={float(E0)} > V_min={V_min}"

        # Compare asymptotic WKB exponent vs true splitting
        cw0 = c_wkb_zero_closed(x)
        if X >= 20:
            ratio_wkb = dE0 / (U * math.exp(-cw0 * X))
            assert 1e-3 < ratio_wkb < 10.0, f"WKB scaling check failed: ratio={ratio_wkb}"

        # Check Theorem 1 ground bound when X >= X0(x)
        X0_g, c0_g, C0_g = thm1_ground_thresholds(x)
        if X >= X0_g:
            assert dE0 <= C0_g * U * math.exp(-c0_g * X)

        print(f"  U={U:.0f}, t={t:.1f}, Om={Om:.1f} (X={X:.0f}, x={x:.2f}): "
              f"E0-V_min={float(E0)-V_min:.3e}, dE0={dE0:.3e}, "
              f"dE0/(U*e^(-c_WKB*X))={dE0/(U*math.exp(-cw0*X)):.3f}")

    # Check excited doublets below vs above barrier at U=30, t=7.5, Om=1.0 (B=16.875)
    U, t, Om = 30.0, 7.5, 1.0
    V_min = -U - (t * t) / U
    B_dim = ((U - t) ** 2) / U
    evals_f, _ = build_ho_dimer_hr_mp(U, t, Om, n_ho=160)
    sub_splits = []
    above_splits = []
    for k in range(25):
        E_even, E_odd = evals_f[2 * k], evals_f[2 * k + 1]
        kap_k = (E_even - V_min) / B_dim
        if kap_k <= 0.5:
            sub_splits.append((k, kap_k, E_odd - E_even))
            assert E_odd - E_even < 1e-4 * U
        elif kap_k >= 1.1:
            above_splits.append((k, kap_k, E_odd - E_even))
    assert len(sub_splits) >= 8 and len(above_splits) >= 2
    if mutant == "mutant_4_no_pigeonhole_naive_bound":
        # Falsely assert above-barrier doublets still obey exponential suppression
        for k, kap_k, dEk in above_splits:
            assert dEk <= 1e-3 * U, f"Above-barrier splitting dE={dEk} is NOT small!"
    print(f"  Sub-barrier k=0..{sub_splits[-1][0]} max dE={max(s[2] for s in sub_splits):.3e}; "
          f"Above-barrier k={above_splits[0][0]} (kap={above_splits[0][1]:.2f}) dE={above_splits[0][2]:.3f}")
    print("  -> PASS")


def lf_second_order_splitting(U: float, t: float, Om: float) -> float:
    X = U / Om
    integrand = lambda s: math.exp(-U * s - X * (1.0 + math.exp(-Om * s)))
    val, _ = quad(integrand, 0.0, 50.0 / U, epsabs=1e-14, epsrel=1e-14)
    return 4.0 * (t ** 2) * val


def check_3_lang_firsov_crossover(mutant: str = "none"):
    print("[Check 3] Lang-Firsov small-polaron exponent (2) vs BO WKB crossover...")
    U, Om = 10.0, 1.0  # X = 10
    # Antiadiabatic regime: t^2/(U*Om) << 1
    ratios_anti = []
    for t in [0.02, 0.05, 0.1, 0.25]:
        evals, _ = build_ho_dimer_hr_mp(U, t, Om, n_ho=70)
        dE_ed = float(evals[1] - evals[0])
        dE_lf = lf_second_order_splitting(U, t, Om)
        dE_saddle = 4.0 * math.sqrt(math.pi / (2.0 * U * Om)) * (t ** 2) * math.exp(-2.0 * U / Om)
        ratio = dE_ed / dE_lf
        ratios_anti.append(ratio)
        assert 0.95 <= ratio <= 1.10, f"Antiadiabatic mismatch at t={t}: {ratio}"
        assert 0.90 <= dE_lf / dE_saddle <= 1.25

    # Adiabatic crossover: t/U = 0.25 (t = 2.5 at U=10, Om=1 -> t^2/(U*Om)=0.625)
    t_ad = 2.5
    evals_ad, _ = build_ho_dimer_hr_mp(U, t_ad, Om, n_ho=70)
    dE_ed_ad = float(evals_ad[1] - evals_ad[0])
    dE_lf_ad = lf_second_order_splitting(U, t_ad, Om)
    ratio_ad = dE_ed_ad / dE_lf_ad
    assert ratio_ad > 5.0, f"Expected adiabatic enhancement > 5, got {ratio_ad}"
    if mutant == "mutant_3_lf_wrong_exponent":
        assert dE_ed_ad <= 1.1 * dE_lf_ad
    print(f"  X=10 antiadiabatic ratios ED/LF(2nd) = {[round(r, 4) for r in ratios_anti]}")
    print(f"  X=10 adiabatic t/U=0.25: ED/LF(2nd) = {ratio_ad:.3f} (c_WKB={c_wkb_zero_closed(0.25):.4f} < 2)")
    print("  -> PASS")


def check_4_thermal_communication_and_excursion():
    print("[Check 4] Theorem 2: inter-valley communication & all-T excursion probability...")
    U, t, Om = 20.0, 2.0, 1.0  # X = 20, x = 0.1
    x = t / U
    n_ho = 120
    H_r, r_op = build_ho_dimer_hr(U, t, Om, n_ho=n_ho)
    evals, evecs = eigh(H_r)
    V_min = -U - (t * t) / U
    B_dim = ((U - t) ** 2) / U

    # Evaluate wavefunctions on Gauss-Hermite grid to compute real-space overlaps with chi_L, chi_R
    # Better: use position-space quadrature directly via Hermite polynomials or r_op eigendecomposition!
    r_vals, r_vecs = eigh(r_op)
    # Transform evecs (shape 3*n_ho, 3*n_ho) into (n_ho, 3, n_states) -> r-grid (n_r, 3, n_states)
    evecs_3 = evecs.reshape(n_ho, 3, -1)
    psi_r = np.einsum("nr,nas->ras", r_vecs, evecs_3)  # quadrature weights already absorbed in r_vecs!

    # 1. Smooth C^2 quintic Hermite partition chi_L, chi_R with eta = 0.5 * sqrt(U*Om)
    eta = 0.5 * math.sqrt(U * Om)
    s_ramp = np.clip((r_vals + eta) / (2.0 * eta), 0.0, 1.0)
    chi_R = (s_ramp ** 3) * (10.0 - 15.0 * s_ramp + 6.0 * (s_ramp ** 2))
    chi_L = 1.0 - chi_R
    # Pair-transfer matrix P20_02 in {|+>, |s>, |->}
    P20_02 = 0.5 * np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])

    # Sub-barrier states below kappa=0.25 (E <= V_min + 0.25 * B_dim = -16.15)
    kap = 0.25
    E_star = V_min + kap * B_dim
    sub_idx = np.where(evals <= E_star)[0]
    assert len(sub_idx) >= 6
    max_comm = 0.0
    max_hr_comm = 0.0
    for m in sub_idx:
        for k in sub_idx:
            psi_mL = chi_L[:, None] * psi_r[:, :, m]
            psi_kR = chi_R[:, None] * psi_r[:, :, k]
            mat_el = np.sum(psi_mL * (psi_kR @ P20_02.T))
            max_comm = max(max_comm, abs(mat_el))
        # Commutator [H_r, chi_R] psi_k in HO basis: H_r (chi_R psi_k) - E_k (chi_R psi_k)
        psi_kR_ho = np.einsum("nr,ra->na", r_vecs, chi_R[:, None] * psi_r[:, :, k]).reshape(-1)
        comm_vec = H_r @ psi_kR_ho - evals[k] * psi_kR_ho
        max_hr_comm = max(max_hr_comm, float(np.linalg.norm(comm_vec)))
    c_half = 0.25 * c_wkb(x, 0.5 * (1.0 + kap))
    assert max_comm < 1e-8, f"Inter-valley communication too large: {max_comm}"
    assert max_hr_comm <= 10.0 * U * math.exp(-c_half * (U / Om)), f"Commutator [H_r, chi_R] too large: {max_hr_comm}"
    # Singlet partition function: Z_sing_r = sum exp(-beta * evals)
    for beta_Om in [0.5, 1.0, 2.0, 5.0]:
        beta = beta_Om / Om
        w_sing = np.exp(-beta * (evals - V_min))
        Z_sing = np.sum(w_sing)
        Z_trip = 3.0 * math.exp(beta * V_min) / (1.0 - math.exp(-beta * Om))
        exc_sing = np.sum(w_sing[evals > E_star])
        prob_exact = (exc_sing + Z_trip) / (Z_sing + Z_trip)
        # Theorem 2 bound (5.3) minimized over gamma in (0, 1)
        gam_grid = np.linspace(0.05, 0.95, 19)
        rhs_sing = np.min((6.0 / gam_grid) * math.sqrt((1.0 + x) / (1.0 - x)) * np.exp(-beta * (1.0 - gam_grid) * kap * B_dim + 0.5 * beta * gam_grid * Om))
        rhs_bound = rhs_sing + 3.0 * math.exp(-beta * (U + t * t / U))
        assert prob_exact <= rhs_bound, f"Excursion bound violated at beta*Om={beta_Om}: {prob_exact} > {rhs_bound}"
        print(f"  beta*Om={beta_Om:3.1f}: P_exc={prob_exact:.3e} <= Thm2_RHS={rhs_bound:.3e}")
    print(f"  max|<psi_mL, Delta1^dag Delta2 psi_kR>| = {max_comm:.3e}, max||[H_r, chi_R] psi_k|| = {max_hr_comm:.3e}")
    print("  -> PASS")


def bo_potential(v: np.ndarray, adj: np.ndarray, U: float, t: float, mu: float = 0.0) -> float:
    h_v = -t * adj + np.diag(v - mu)
    evals = np.linalg.eigvalsh(h_v)
    return float(np.sum(v * v) / (2.0 * U) - np.sum(np.abs(evals)) - mu * len(v))


def check_5_manybody_continuation_and_locality(mutant: str = "none"):
    print("[Check 5] Many-body continuation: cold-cutoff avoidance, BO tensorization (Thm 3), and locality failure...")
    U, t, Om = 20.0, 2.0, 1.0
    # 5a. Physical distortion vs LF cold projection: virtual |s> weight vs barrier excursion in GS
    n_ho = 100
    H_r, r_op = build_ho_dimer_hr(U, t, Om, n_ho=n_ho)
    _evals, evecs = eigh(H_r)
    psi0 = evecs[:, 0].reshape(n_ho, 3)
    weight_s = float(np.sum(psi0[:, 1] ** 2))
    r_vals, r_vecs = eigh(r_op)
    psi0_r = r_vecs.T @ psi0
    r0 = math.sqrt(U * U - t * t)
    assert float(np.sum(psi0_r[np.abs(r_vals) <= 0.5 * r0] ** 2)) < 1e-5
    assert float(np.sum(psi0_r[np.abs(r_vals) <= 0.3 * r0] ** 2)) < 1e-10
    prob_bar_01 = float(np.sum(psi0_r[np.abs(r_vals) <= 0.1 * r0] ** 2))
    assert 0.5 * (t / U) ** 2 * 0.8 < weight_s < 1.5 * 0.5 * (t / U) ** 2
    assert prob_bar_01 < 1e-14

    # 5b. Theorem 3 BO barrier tensorization check on N=3 and N=4 chains/rings
    for N_sites, periodic, d in [(3, False, 1), (4, True, 1)]:
        adj = np.zeros((N_sites, N_sites))
        for i in range(N_sites - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        if periodic:
            adj[0, N_sites - 1] = adj[N_sites - 1, 0] = 1.0
        v_well = np.array([(-1) ** i * U for i in range(N_sites)], dtype=float)
        # Flip site 0 into barrier v_0 = 0
        v_bar1 = v_well.copy()
        v_bar1[0] = 0.0
        dV1 = bo_potential(v_bar1, adj, U, t) - bo_potential(v_well, adj, U, t)
        lb1 = 0.5 * U - 4 * d * t
        assert dV1 >= lb1, f"Theorem 3 failed on N={N_sites}: {dV1} < {lb1}"
        if N_sites == 4:
            # Flip two sites (0,1) -> 0 simultaneously (single-pair transfer)
            v_bar2 = np.array([0.0, 0.0, U, -U], dtype=float)
            dV2 = bo_potential(v_bar2, adj, U, t) - bo_potential(v_well, adj, U, t)
            lb2 = 2 * (0.5 * U - 4 * d * t)
            assert dV2 >= lb2
            # Flip all 4 sites -> 0 simultaneously (2-pair CDW reversal)
            v_bar4 = np.zeros(4, dtype=float)
            dV4 = bo_potential(v_bar4, adj, U, t) - bo_potential(v_well, adj, U, t)
            lb4 = 4 * (0.5 * U - 4 * d * t)
            assert dV4 >= lb4
            print(f"  N=4 2-site barrier={dV2:.3f} >= {lb2:.3f}; 4-site reversal={dV4:.3f} >= {lb4:.3f} (2*dimer={2*((U-t)**2/U):.3f})")

    # 5c. Locality failure witness: 3-site chain (sites 0,1,2) in N=2 singlet electronic sector
    # Show ||(I - P_01) V_12 |psi_0>|| is Theta(t^2/U), NOT e^{-c U/Om}.
    # Electronic N=2, S_z=0 singlet basis on 3 sites: 6 states (3 doubly occupied, 3 singlet bonds)
    def build_3site_singlet(v0, v1, v2):
        H = np.zeros((6, 6))
        occ = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        for idx, (n0, n1, n2) in enumerate(occ):
            H[idx, idx] = v0 * (n0 - 1) + v1 * (n1 - 1) + v2 * (n2 - 1)
        rt2 = math.sqrt(2.0)
        T01 = np.zeros((6, 6))
        T01[0, 3] = T01[3, 0] = -rt2 * t
        T01[1, 3] = T01[3, 1] = -rt2 * t
        T01[4, 5] = T01[5, 4] = -t
        T12 = np.zeros((6, 6))
        T12[1, 5] = T12[5, 1] = -rt2 * t
        T12[2, 5] = T12[5, 2] = -rt2 * t
        T12[3, 4] = T12[4, 3] = -t
        return H, T01, T12

    H_diag2, T01_2, T12_2 = build_3site_singlet(-U, +U, +U)
    # Local {0,1} dimer subspace with site 2 empty (n_2 = 0 -> basis indices [0, 1, 3])
    idx01 = [0, 1, 3]
    _, vecs_01_sub = eigh((H_diag2 + T01_2)[np.ix_(idx01, idx01)])
    psi_01_2 = np.zeros(6)
    psi_01_2[idx01] = vecs_01_sub[:, 0]
    # Local projector P_01 onto the 2 lowest states of the {0,1} dimer (n_2 = 0)
    P01 = np.zeros((6, 6))
    for col in range(2):
        u = np.zeros(6)
        u[idx01] = vecs_01_sub[:, col]
        P01 += np.outer(u, u)
    leak = float(np.linalg.norm((np.eye(6) - P01) @ T12_2 @ psi_01_2))
    # psi_01_2 has amplitude ~ sqrt(2)*t/(2U) in |1,1,0>_s; T12 maps |1,1,0>_s -> |1,0,1>_s (n_2=1),
    # giving leak ~ sqrt(2)*t^2/(2U) = 0.14089 at U=20, t=2!
    assert leak > 0.05, f"Expected O(t^2/U) leakage, got {leak}"
    if mutant == "mutant_6_locality_holds":
        assert leak <= math.exp(-0.5 * U / Om), f"Locality failed: leak={leak} >> exp(-0.5*X)"
    print(f"  Dimer GS virtual |s> weight={weight_s:.4e}, barrier P(|r|<=0.1 r0)={prob_bar_01:.2e}")
    print(f"  4-site BO tensorization N=4: dV1={dV1:.3f} >= {lb1:.3f}")
    print(f"  fixed-v 3-site locality leakage ||(I-P01) V12 |psi0>|| = {leak:.5f} (Theta(t^2/U), NOT e^(-cU/Om))")
    print("  -> PASS")


def run_all(mutant: str = "none"):
    check_1_certified_conventions_and_identities(mutant)
    check_2_dimer_ed_splitting(mutant)
    check_3_lang_firsov_crossover(mutant)
    check_4_thermal_communication_and_excursion()
    check_5_manybody_continuation_and_locality(mutant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants", action="store_true", help="Run adversarial mutation suite")
    args = parser.parse_args()

    t0 = time.time()
    if not args.mutants:
        run_all("none")
        print(f"ALL REPLAY CHECKS PASSED in {time.time() - t0:.2f}s")
        return 0

    mutants = [
        "mutant_1_sign_g",
        "mutant_2_barrier_missing_t2",
        "mutant_3_lf_wrong_exponent",
        "mutant_4_no_pigeonhole_naive_bound",
        "mutant_5_delta1_delta2_diagonal",
        "mutant_6_locality_holds",
    ]
    print(f"Running {len(mutants)} mutants (every mutant MUST fail)...")
    for m in mutants:
        caught = False
        try:
            run_all(m)
        except AssertionError as exc:
            caught = True
            print(f"  [CAUGHT {m}]: {exc}")
        if not caught:
            print(f"  [FAILURE] Mutant {m} was NOT caught!")
            return 1
    print(f"ALL {len(mutants)}/{len(mutants)} MUTANTS CAUGHT in {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

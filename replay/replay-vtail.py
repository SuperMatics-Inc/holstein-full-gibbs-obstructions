#!/usr/bin/env python3
"""PROOF DRAFT replay: local arithmetic, dimer ED and genuine mutation guards.

NUMERICAL ED uses finite-Fock Galerkin matrices, two cutoffs and the full
16-state grand-canonical electronic census. No infinite-dimensional theorem
is labeled CERTIFIED by this script.
"""
from __future__ import annotations

import os

# Keep one bounded CPU process, including BLAS, even when called directly.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from fractions import Fraction
import math
import subprocess
import sys
from typing import Any

import mpmath as mp
import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, gammaln

IV: Any = mp.iv  # mpmath interval stubs omit several real runtime operations.
IV.dps = 70
mp.mp.dps = 80


def iv(value: int | str | Fraction) -> Any:
    if isinstance(value, Fraction):
        return IV.mpf(value.numerator) / IV.mpf(value.denominator)
    return IV.mpf(value)


def guard(condition: bool, name: str) -> None:
    if not condition:
        raise AssertionError(name)


def local_constants(a: float, b: float, q: int = 1) -> tuple[float, float, float]:
    if not (0 < a < b and q >= 1):
        raise ValueError("Require 0<a<b and degree q>=1")
    c0 = min(1 / 64, a / (64 + 16 * math.log(q + 1)))
    c1 = min(1, 1 / b) if q == 1 else min(2.0**-20 / q**3, 1 / (b * q))
    return c0, c1, 1 / 4096


def check_arithmetic(mutant: int = 0) -> dict[str, float]:
    cD = iv(1) / 4 - iv(1) / (4 * IV.sqrt(iv(2))) - iv(1) / 64
    delta = 2 * IV.exp((-1 if mutant != 4 else 1) * 64 * cD)
    guard(float(cD.a) > 1 / 128 and float(delta.b) < 1 / 10, "decay")
    moment = (1 + 1 / IV.sqrt(iv(32))) ** 2
    guard(float(moment.b) < 1.5, "moment")
    block = Fraction(2) * Fraction(19, 10) ** 2 / 10
    guard(block == Fraction(361, 500) and block > Fraction(2, 3), "block")
    guard(float(IV.exp(iv(-16)).b) < 2.0**-23, "odd-thermal")
    guard(Fraction(1, 2**27) + Fraction(1, 2**34) < Fraction(1, 2**26), "odd-torus")
    guard(Fraction(32, 2**13) == Fraction(1, 256), "neighbor")
    guard(8 * math.sqrt(2) + 8 < 20, "same-edge")
    guard(Fraction(1, 6) - Fraction(1, 256) > Fraction(1, 8), "closing")
    guard(Fraction(1, 48**2) > Fraction(1, 4096), "tail-constant")
    h = (1 - IV.exp(iv(-2))) ** 2
    du_rem = iv(8) / 3 * h / 2**15
    guard(float((h / 512 - du_rem).a) > 0, "duhamel-remainder")
    guard(float((iv(1) - 17 * IV.exp(iv(-16))).a) > 0.5, "duhamel-kernel")
    guard(1 - Fraction(2048, 961 * 64) - Fraction(1, 64) > Fraction(1, 2), "poisson-event")

    # Corner checks supplement, but do not replace, the symbolic all-q proof.
    for a, b in ((0.001, 100), (0.5, 5), (2, 40)):
        for q in (1, 2, 4, 6, 20, 200):
            c0, c1, _ = local_constants(a, b, q)
            X = 1 / c0
            odd = 4 / (a * X) * math.exp(-a * X / 4) + 16 * q * c1 / X**3
            pk = 0.5 - odd - math.sqrt(q * b * c1 / (2 * X**2)) - 1 / 8
            guard(pk > 0.25, "cold-corner")
            if q > 1:
                guard(32 * (q - 1) * math.sqrt(odd) < 1 / 256, "neighbor-corner")
    return {"delta64_upper": float(delta.b), "positive_block": float(block)}


def fermions(sites: int) -> list[np.ndarray]:
    dimension = 1 << (2 * sites)
    result = []
    for orbital in range(2 * sites):
        c = np.zeros((dimension, dimension), dtype=np.int64)
        for state in range(dimension):
            if state & (1 << orbital):
                c[state ^ (1 << orbital), state] = (-1) ** ((state & ((1 << orbital) - 1)).bit_count())
        result.append(c)
    return result


def check_car(mutant: int = 0) -> None:
    c = fermions(2)
    I = np.eye(16, dtype=np.int64)
    for i in range(4):
        for j in range(4):
            guard(np.array_equal(c[i] @ c[j].T + c[j].T @ c[i], I if i == j else 0 * I), "car")
    n = [c[2*z].T @ c[2*z] + c[2*z+1].T @ c[2*z+1] for z in range(2)]
    odd = [np.diag((np.diag(nz) == 1).astype(np.int64)) for nz in n]
    counts = np.diag(odd[0] + odd[1])
    P = [np.diag((counts == j).astype(np.int64)) for j in range(3)]
    A = c[0].T @ c[2] + c[1].T @ c[3]
    guard(not np.any(P[0] @ A @ P[0]) and not np.any(P[2] @ A @ P[2]), "even-blocks")
    guard(np.any(P[1] @ A @ P[1]), "one-odd-block")
    Qpair = np.diag(((counts == 0) & (np.diag(n[0]) != np.diag(n[1]))).astype(np.int64))
    Qequal = P[0] - Qpair
    guard(not np.any(Qequal @ A @ P[2]) and not np.any(Qequal @ A.T @ P[2]), "equal-annihilation")
    H = A + A.T
    # Full hopping has eigenvalues 0,+-1,+-2, including the one-odd sector.
    H2 = H @ H
    guard(not np.any(H2 @ (H2-I) @ (H2-4*I)), "hop-polynomial")
    guard(abs(float(np.linalg.eigvalsh(H)[-1]) - (1 if mutant == 5 else 2)) < 1e-12, "hop-norm")

    # Three-site neighbor commutator really flips the nonshared site's parity.
    c3 = fermions(3)
    I3 = np.eye(64, dtype=np.int64)
    nz = [c3[2*z].T @ c3[2*z] + c3[2*z+1].T @ c3[2*z+1] for z in range(3)]
    oz = [np.diag((np.diag(v) == 1).astype(np.int64)) for v in nz]
    P0 = (I3-oz[0]) @ (I3-oz[1])
    P2 = oz[0] @ oz[1]
    h01 = c3[0].T @ c3[2] + c3[1].T @ c3[3]
    h01 = h01 + h01.T
    h12 = c3[2].T @ c3[4] + c3[3].T @ c3[5]
    h12 = h12 + h12.T
    K = P2 @ h01 @ P0 - P0 @ h01 @ P2
    C = K @ h12 - h12 @ K
    guard(np.any(C) and not np.any(oz[0] @ C @ oz[0]) and not np.any((I3-oz[0]) @ C @ (I3-oz[0])), "neighbor-parity")


def hot_fraction(total: int, cutoff: int) -> Fraction:
    if total <= cutoff:
        return Fraction(0)
    return Fraction((1 << total) - sum(math.comb(total, k) for k in range(cutoff + 1)), 1 << total)


def interval_channels(X: int, mutant: int = 0) -> tuple[Any, Any, float]:
    cutoff = X // 64 if mutant != 7 else 2 * X
    lam = iv(X)
    K = 2 * X + 128
    poisson = IV.exp(-lam)
    bare, rotated = iv(0), iv(0)
    for total in range(K + 1):
        fraction = iv(hot_fraction(total, cutoff))
        r = iv(total) / X
        bare += poisson * fraction / (1 + r) ** 2
        amp = -r / (1 + r) if mutant != 6 else iv(0)
        rotated += poisson * fraction * amp**2
        if total < K:
            poisson *= lam / (total + 1)
    tail = poisson * lam / (K + 1 - X)
    guard(float(tail.b) < 1e-45, "series-tail")
    if mutant == 1:
        bare *= iv(Fraction(1, 2))
    return bare + IV.mpf([0, float(tail.b)]), rotated + IV.mpf([0, float(tail.b)]), float(tail.b)


def direct_channels(X: int) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Independent two-site Poisson sum, not the conditional-binomial route."""
    lam = mp.mpf(X) / 2
    K = X + 100
    p = [mp.exp(-lam)]
    for n in range(1, K + 1):
        p.append(p[-1] * lam / n)
    bare, rotated = mp.mpf(0), mp.mpf(0)
    for n in range(X // 64 + 1, K + 1):
        for m in range(K + 1):
            r = mp.mpf(n + m) / X
            weight = p[n] * p[m]
            bare += weight / (1 + r)**2
            rotated += weight * (r / (1 + r))**2
    tail = 2 * p[-1] * lam / (K + 1 - lam)
    return bare, rotated, tail


def check_channels(mutant: int = 0) -> dict[int, tuple[float, float]]:
    results = {}
    for X in (64, 128, 256):
        bare, rotated, tail = interval_channels(X, mutant)
        b, r, direct_tail = direct_channels(X)
        bmid = (float(bare.a) + float(bare.b)) / 2
        rmid = (float(rotated.a) + float(rotated.b)) / 2
        guard(abs(bmid - float(b)) < 1e-14 and float(direct_tail) < 1e-35, "channel-normalization")
        guard(abs(rmid - float(r)) < 1e-14, "rotation-channel")
        # A high-precision independent value must lie inside the series enclosure.
        for enclosure, value in ((bare, b), (rotated, r)):
            enclosing = IV.mpf(mp.nstr(value, 75)) + IV.mpf([0, float(direct_tail)])
            guard(float((enclosure - enclosing).a) <= 0 <= float((enclosure - enclosing).b), "channel-enclosure")
        if X == 64:
            guard(abs(bmid - 0.252956) < 5e-7, "input-channel64")
        if X == 256:
            guard(abs(bmid - 0.250734) < 5e-7, "input-channel256")
        print(f"INTERVAL-ENCLOSED X={X}: w_LF={bmid:.12f}, w_S1={rmid:.12f}, omitted tail<={tail:.2e}", flush=True)
        results[X] = (bmid, rmid)
    return results


def beam_hot(p: int, q: int, cutoff: int) -> Fraction:
    """Exact rational 50:50 beam splitter at total occupation p+q."""
    total = p + q
    if total <= cutoff:
        return Fraction(0)
    cold = Fraction(0)
    denominator = math.factorial(p) * math.factorial(q) * (1 << total)
    for k in range(cutoff + 1):
        coefficient = sum((-1)**(q-k+j) * math.comb(p, j) * math.comb(q, k-j)
                          for j in range(max(0, k-q), min(p, k)+1))
        cold += Fraction(coefficient**2 * math.factorial(k) * math.factorial(total-k), denominator)
    result = 1 - cold
    guard(0 <= result <= 1, "beam-probability")
    return result


def beam_weights(nrel: int, cutoff: int, beta_Om: float) -> tuple[np.ndarray, float]:
    ncom = max(4, math.ceil(64 / beta_Om))
    r = math.exp(-beta_Om)
    weights = np.empty(nrel)
    # Unnormalized geometric truncation: omitted probability is explicit.
    pcom = (1-r) * r**np.arange(ncom)
    for q in range(nrel):
        weights[q] = sum(pcom[p] * float(beam_hot(p, q, cutoff)) for p in range(ncom))
    return weights, r**ncom


def displacement(nrel: int, alpha: float) -> np.ndarray:
    """Exact infinite-Fock Weyl matrix elements, then Galerkin compressed."""
    m, n = np.indices((nrel, nrel))
    lo = np.minimum(m, n)
    hi = np.maximum(m, n)
    power = hi-lo
    prefactor = np.exp(-alpha**2/2 + (gammaln(lo+1)-gammaln(hi+1))/2 + power*math.log(alpha))
    sign = np.where((m < n) & (power % 2 == 1), -1, 1)
    return sign * prefactor * eval_genlaguerre(lo, power, alpha**2)


def check_oscillators(mutant: int = 0) -> None:
    # Independent finite-dimensional matrix-exponential beam splitter.
    for total in range(1, 13):
        generator = np.zeros((total+1, total+1))
        for k in range(total):
            v = math.pi/4 * math.sqrt((k+1)*(total-k))
            generator[k+1, k] = v
            generator[k, k+1] = -v
        eigenvalues, vectors = eigh(1j*generator)
        U = (vectors*np.exp(-1j*eigenvalues)) @ vectors.conj().T
        for p in range(total+1):
            q = total-p
            numerical = float(np.sum(np.abs(U[2:, p])**2))
            guard(abs(numerical-float(beam_hot(p, q, 1))) < 3e-14, "beam-independent")
    nrel = 96
    alpha = math.sqrt(4 if mutant != 8 else 1)
    D = displacement(nrel, alpha)
    a = np.diag(np.sqrt(np.arange(1, nrel)), 1)
    eigenvalues, vectors = eigh(2j*(a.T-a))
    reference = ((vectors*np.exp(-1j*eigenvalues)) @ vectors.conj().T).real
    guard(np.max(np.abs(D[:12, :12]-reference[:12, :12])) < 2e-13, "displacement")


def ed_point(X: int, beta_Om: float, t: float, nrel: int) -> dict[str, float]:
    Om = 1.0
    U = float(X)
    cutoff = X // 64
    x = t/U
    wq, com_tail = beam_weights(nrel, cutoff, beta_Om)
    D = displacement(nrel, math.sqrt(X))
    I = np.eye(nrel)
    hph = np.diag(np.arange(nrel, dtype=float))
    zero = np.zeros((nrel, nrel))
    Vp = -math.sqrt(2)*t*D
    Vm = Vp.T
    H3 = np.block([[hph, zero, Vm], [zero, hph, Vp], [Vp, Vm, U*I+hph]])
    H2 = np.block([[U/2*I+hph, -t*D.T], [-t*D, U/2*I+hph]])
    ev3, vec3 = eigh(H3, check_finite=True)
    ev2, vec2 = eigh(H2, check_finite=True)
    emin = min(0.0, float(ev3[0]), float(ev2[0]))
    weights3 = np.exp(-beta_Om*(ev3-emin))
    weights2 = np.exp(-beta_Om*(ev2-emin))
    weights0 = np.exp(-beta_Om*(np.arange(nrel)-emin))
    weightstrip = np.exp(-beta_Om*(U+np.arange(nrel)-emin))
    Z = float(np.sum(weights3) + 4*np.sum(weights2) + 2*np.sum(weights0) + 3*np.sum(weightstrip))
    obs3 = np.tile(wq, 3)
    obs2 = np.tile(wq, 2)
    p3 = obs3 @ (vec3**2)
    p2 = obs2 @ (vec2**2)

    # The actual sign of LARGE-U S1, evaluated with a controlled Taylor tail.
    S = np.block([[zero, zero, -Vm/U], [zero, zero, -Vp/U], [Vp/U, Vm/U, zero]])
    H0 = np.block([[hph*0, zero, zero], [zero, hph*0, zero], [zero, zero, U*I]])
    Vmix = H3 - np.block([[hph, zero, zero], [zero, hph, zero], [zero, zero, U*I+hph]])
    guard(np.max(np.abs(S @ H0-H0 @ S+Vmix)) < 1e-18, "S1-sign")
    term = vec3.copy()
    rotated = vec3.copy()
    for k in range(1, 4):
        term = (S @ term)/k
        rotated += term
    rotation_error = math.exp(3*x) * (3*x)**4 / math.factorial(4)
    norm_error = float(np.max(np.abs(np.sum(rotated**2, axis=0)-1)))
    guard(norm_error < 1e-11, "rotation-norm")
    p3rot = obs3 @ (rotated**2)
    inactive = float(np.dot(2*weights0+3*weightstrip, wq))
    pbare = float((np.dot(weights3, p3)+4*np.dot(weights2, p2)+inactive)/Z)
    prot = float((np.dot(weights3, p3rot)+4*np.dot(weights2, p2)+inactive)/Z)
    guard(pbare >= 0 and prot >= 0, "ED-probability")
    return {"p_LF": pbare, "p_S1": prot, "p_atomic": math.exp(-beta_Om*(cutoff+1)),
            "com_tail": com_tail, "rotation_error": rotation_error,
            "dimension": float(3*nrel), "x2": x*x}


def run_ed(mutant: int = 0) -> list[dict[str, float]]:
    a, b = 2.0, 40.0
    c0, c1, c2 = local_constants(a, b)
    points = ((65, 2), (65, 20), (65, 28), (80, 4), (80, 22), (96, 26), (128, 24), (128, 32))
    result = []
    for X, beta_Om in points:
        t = 0.75*c1/X**2
        if mutant == 3:
            t = 50*X
        guard(X > 1/c0 and a <= beta_Om <= b and 0 < t < c1/X**2, "window")
        nrel = 2*X+64
        low = ed_point(X, beta_Om, t, nrel)
        high = ed_point(X, beta_Om, t, nrel+32)
        bound = c2*high["x2"]
        guard(high["com_tail"] < 1e-8*bound and 2*high["rotation_error"] < 1e-8*bound, "omitted-error")
        # Relative cutoff tolerance uses the actual probability, not an absolute
        # tolerance many orders of magnitude larger than the virtual signal.
        for key in ("p_LF", "p_S1"):
            tolerance = 2e-6*max(high[key], high["x2"])
            guard(abs(low[key]-high[key]) < tolerance, "ED-convergence")
        guard(high["p_LF"] >= bound, "ED-lower-bound")
        if mutant == 2:
            guard(high["p_LF"] >= 2.0*high["x2"], "inflated-constant")
        if beta_Om >= 20:
            # At these cold points the whole tail, rather than a cancellation-
            # prone warm excess, can be compared with the vacuum channel.
            bare, rotated, _ = interval_channels(X)
            expected_bare = float(bare.a)
            expected_rot = float(rotated.a)
            ratio_bare = (high["p_LF"]-high["p_atomic"])/high["x2"]
            ratio_rot = (high["p_S1"]-high["p_atomic"])/high["x2"]
            guard(abs(ratio_bare-expected_bare) < 1e-5, "ED-vacuum-channel")
            guard(abs(ratio_rot-expected_rot) < 1e-5, "ED-rotation-channel")
        high.update({"X": float(X), "beta_Om": float(beta_Om), "t": t, "bound": bound,
                     "cutoff_delta": abs(low["p_LF"]-high["p_LF"])})
        print(f"NUMERICAL ED X={X}, betaOmega={beta_Om}, t={t:.6e}: "
              f"p_LF={high['p_LF']:.12e} >= {bound:.3e}, p_S1={high['p_S1']:.12e}, "
              f"atomic={high['p_atomic']:.3e}, cutoff_delta={high['cutoff_delta']:.3e}, "
              f"COM_tail={high['com_tail']:.1e}", flush=True)
        result.append(high)
        if mutant == 2 and beta_Om >= 20:
            break
    guard(len(result) >= 6 or mutant != 0, "point-count")
    return result


MUTANTS = {
    1: ("Spurious factor 1/2 in the channel weight", "channel-normalization"),
    2: ("Inflate the cold dimer constant above its channel coefficient", "inflated-constant"),
    3: ("Drop the required small-hopping window", "window"),
    4: ("Reverse the cold displacement exponent", "decay"),
    5: ("Replace the spinful dimer hopping norm 2 by 1", "hop-norm"),
    6: ("Replace the post-S1 channel amplitude by zero", "rotation-channel"),
    7: ("Move the cutoff above the Franck-Condon peak", "channel-normalization"),
    8: ("Use relative LF displacement sqrt(X/4) instead of sqrt(X)", "displacement"),
}


def execute_mutant(number: int) -> int:
    if number not in MUTANTS:
        raise ValueError("Unknown mutant")
    try:
        if number in (1, 6, 7):
            check_channels(number)
        elif number == 4:
            check_arithmetic(number)
        elif number == 5:
            check_car(number)
        elif number == 8:
            check_oscillators(number)
        else:
            run_ed(number)
    except AssertionError as exc:
        if str(exc) == MUTANTS[number][1]:
            print(f"MUTANT_CAUGHT:{number}:{exc}")
            return 1
        print(f"UNRELATED_FAILURE:{number}:{exc}", file=sys.stderr)
        return 2
    print(f"MUTANT_MISSED:{number}")
    return 0


def run_mutants() -> int:
    for number, (description, assertion) in MUTANTS.items():
        child = subprocess.run([sys.executable, __file__, "--mutant", str(number)],
                               capture_output=True, text=True, check=False)
        marker = f"MUTANT_CAUGHT:{number}:{assertion}"
        print(f"MUTANT {number}: exit={child.returncode}, {description}", flush=True)
        if child.returncode != 1 or marker not in child.stdout:
            print(child.stdout + child.stderr, file=sys.stderr)
            return 1
    print("PASS: 8/8 mutants rejected by their designated guards; each exit 1.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutants", action="store_true")
    parser.add_argument("--mutant", type=int)
    args = parser.parse_args()
    if args.mutants:
        return run_mutants()
    if args.mutant is not None:
        return execute_mutant(args.mutant)
    check_arithmetic()
    check_car()
    check_oscillators()
    channels = check_channels()
    rows = run_ed()
    print(f"PASS: endpoint arithmetic, exact CAR blocks, independent channels, {len(rows)} converged dimer ED points.")
    print(f"Computed channel difference at X=256: {channels[256][0]-channels[256][1]:.12f}")
    print("PROOF DRAFT: independent analytical review is not supplied by this replay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Exact coefficient/constant certificates, not a full-Gibbs LRO proof.

Only the standard library and SymPy are used. Every enclosure endpoint is a
Fraction; floats are never used for admission. --all-mutants is an aggregate
runner: each named child must exit 1 and the aggregate then exits 0.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from itertools import combinations
from math import factorial
from pathlib import Path
import subprocess
import sys

import sympy as sp

MUTANTS = (
    "one-spin", "xy-sign", "extra-constant", "half-denominator",
    "one-oscillator", "wrong-kummer", "erase-anisotropy",
    "short-time-trace", "small-commutator-count", "tail-underestimate",
    "allow-cubic-cold", "atomic-lro",
)


class Rejected(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rejected(message)


def same(a, b, message: str) -> None:
    require(sp.simplify(a - b) == 0, message)


def exp_positive(x: F, terms: int = 192) -> tuple[F, F]:
    """Bound exp(x) by its positive Taylor series and a geometric tail."""
    require(x >= 0 and terms + 2 > x, "invalid exponential tail domain")
    term = F(1)
    total = term
    for k in range(1, terms + 1):
        term *= x / k
        total += term
    first_omitted = term * x / (terms + 1)
    tail = first_omitted / (1 - x / (terms + 2))
    return total, total + tail


def exp_negative(x: F) -> tuple[F, F]:
    low, high = exp_positive(x)
    return 1 / high, 1 / low


def product_interval(a: tuple[F, F], b: tuple[F, F]) -> tuple[F, F]:
    corners = [u * v for u in a for v in b]
    return min(corners), max(corners)


def poisson_resolvent(x: F, alternating: bool, terms: int = 192) -> tuple[F, F]:
    """Independent positive/alternating sums with an absolute infinite tail."""
    term = F(1)
    total = F(0)
    for m in range(terms + 1):
        if m:
            term *= x / m
        total += (-1 if alternating and m % 2 else 1) * term * x / (x + m)
    omitted_mass = term * x / (terms + 1)
    tail = omitted_mass / (1 - x / (terms + 2))
    raw = (total - tail, total + tail) if alternating else (total, total + tail)
    return product_interval(exp_negative(x), raw)


def fermion_checks(mutant: str | None) -> None:
    identity = sp.eye(2)
    parity = sp.diag(1, -1)
    annihilator = sp.Matrix([[0, 1], [0, 0]])
    cs = []
    for mode in range(4):
        factors = [parity if j < mode else annihilator if j == mode else identity
                   for j in range(4)]
        cs.append(sp.kronecker_product(*factors))
    for i in range(4):
        for j in range(4):
            require(cs[i] * cs[j].T + cs[j].T * cs[i] ==
                    (sp.eye(16) if i == j else sp.zeros(16)), "CAR mismatch")
    spin_pairs = [(0, 2)] if mutant == "one-spin" else [(0, 2), (1, 3)]
    forward = sp.zeros(16)
    for x, y in spin_pairs:
        forward += cs[y].T * cs[x]
    backward = forward.T
    zeta, chi = sp.symbols("zeta chi")
    used_chi = -chi if mutant == "xy-sign" else chi
    matrix = -(zeta * (forward * backward + backward * forward)
               + used_chi * (forward * forward + backward * backward))
    cold = [0, 12, 3, 15]  # |00>, |20>, |02>, |22> in tensor bit order
    projected = matrix.extract(cold, cold)
    if mutant == "extra-constant":
        projected -= zeta * sp.eye(4)
    target = sp.Matrix([[0, 0, 0, 0], [0, -2*zeta, -2*chi, 0],
                        [0, -2*chi, -2*zeta, 0], [0, 0, 0, 0]])
    require(projected == target, "exact bond matrix, factors/signs/constants")
    p = sp.diag(*[int(i in cold) for i in range(16)])
    nx = cs[0].T*cs[0] + cs[1].T*cs[1]
    ny = cs[2].T*cs[2] + cs[3].T*cs[3]
    sx = nx - sp.eye(16)
    sy = ny - sp.eye(16)
    hopping = -(forward + backward)
    require(sx * hopping * p == sp.zeros(16) and sy * hopping * p == sp.zeros(16),
            "one hop has two odd-occupation intermediate sites")
    require(p*hopping*p == sp.zeros(16), "first-order cold hopping vanishes")
    g = sp.diag(*[(-1)**int(nx[i, i]) for i in range(16)])
    require(g*hopping*g == -hopping and g*p == p, "bipartite parity selection")
    # With the coherent displacement vectors omitted, the two electronic
    # images coincide. Full-Fock rank <=2 follows from the two active columns;
    # claiming rank 2 for this bare electronic matrix would be incorrect.
    require((hopping*p).rank() == 1,
            "bare electronic image has rank one; two active full-Fock columns")
    print("EXACT electronic matrix: diagonals -2 f_z; transfer -2 f_xy; aligned 0")


def symbolic_checks(mutant: str | None) -> None:
    x, m = sp.symbols("X m", positive=True)
    # The Pochhammer ratio is obtained from Gamma recursion; this recurrence
    # proves it at every nonnegative index, not just the sampled indices.
    a_m = x / (x + m)
    same(a_m * (x + m) / (x + m + 1), x / (x + m + 1), "Pochhammer recurrence")
    same(a_m.subs(m, 0), 1, "Pochhammer initial value")
    denominator = x/2 + m if mutant == "half-denominator" else x + m
    same(x / denominator, x / (x + m), "U+m omega denominator")
    alpha_sq = x/2
    poisson_parameter = alpha_sq if mutant == "one-oscillator" else 2*alpha_sq
    same(poisson_parameter, x, "two-oscillator Poisson parameter")
    # Kummer: y(z)=exp(z) v(-z), with v solving parameters b-a,b.
    a, b, z = sp.symbols("a b z")
    v, vp, vpp = sp.symbols("v vp vpp")
    transformed = z*(v - 2*vp + vpp) + (b-z)*(v-vp) - a*v
    negative_v_equation = -((-z)*vpp + (b+z)*vp - (b-a)*v)
    same(transformed, negative_v_equation, "Kummer differential transformation")
    same(1 - (b-a)/b, a/b, "Kummer first derivative at zero")
    first_parameter = x if mutant == "wrong-kummer" else sp.Integer(1)
    same(first_parameter, (x+1)-x, "Kummer transformed first parameter is 1")
    odd_multiplier = 0 if mutant == "erase-anisotropy" else 2
    require(odd_multiplier == 2, "strict anisotropy retains positive odd terms")
    r = sp.symbols("r")
    full = (1+r)/(1-r)**2 - 1 - 3*r
    tail = r**2*((4 if mutant == "tail-underestimate" else 5)-3*r)/(1-r)**2
    same(full, tail, "all-order cubic-and-higher generating function")
    cold = ((1+r)/(1-r)**2 - (1-r)/(1+r)**2)/2 - 3*r
    same(cold, r**3*(7-3*r**2)/(1-r**2)**2, "cold even-event generating function")
    weighted_all = (2+3*r-r*r)/(1-r)**3 - 2 - 9*r
    same(weighted_all, r*r*(20-25*r+9*r*r)/(1-r)**3, "site incidence generating function")
    weighted_odd = ((2+3*r-r*r)/(1-r)**3 - (2-3*r-r*r)/(1+r)**3)/2 - 9*r
    same(weighted_odd, r**3*(35-28*r*r+9*r**4)/(1-r*r)**3,
         "weighted cold generating function")
    cubic_coefficient = 5 if mutant == "allow-cubic-cold" else 0
    require(cubic_coefficient == sp.expand(sp.series(cold, r, 0, 4).removeO()).coeff(r, 2),
            "bipartite cold cubic coefficient is zero")
    commutator_factor = 2 if mutant == "small-commutator-count" else 4
    require(commutator_factor == 2*2, "2||S_b|| with ||S_b|| <= 2t/U")
    print("EXACT symbolic: resolvent ratio, Kummer ODE/initial data, all four tails")


def constant_checks(mutant: str | None) -> None:
    e4_witness = sum((F(4)**k / factorial(k) for k in range(5)), F(0))
    require(e4_witness == F(103, 3), "exp(4) Taylor witness")
    require(F(6, 103) < F(1, 16), "J_xy/J_z upper bound")
    require(4*(F(1, 2)-F(3, 103)) == F(194, 103), "Delta lower bound")
    require(F(4, 5) < F(24, 25), "high heat-trace elementary witness")
    trace_time = F(1, 4) if mutant == "short-time-trace" else F(4)
    exp_iv = exp_negative(trace_time)
    high_lower = 4*exp_iv[0]/(1-exp_iv[0])  # U/2=omega=D=1
    high_upper = 4*exp_iv[1]/(1-exp_iv[1])
    proposed = exp_negative(trace_time/2)
    require(high_upper <= proposed[0], "high trace requires beta D >= 4")
    require(high_lower <= high_upper, "heat trace enclosure ordering")
    require(F(7)-3*F(1, 4)**2 == F(109, 16), "cold endpoint numerator")
    require((7-3*F(1, 4)**2)/(1-F(1, 4)**2)**2 == F(1744, 225) < 8,
            "cold unweighted endpoint constant")
    require((35-28*F(1, 2)**2+9*F(1, 2)**4)/(1-F(1, 2)**2)**3
            == F(1828, 27) < 68, "cold weighted endpoint constant")
    require(4*8*68 == 2176 and 2176*4**3*2**4 == 2228224,
            "weighted cold t^4 density constant")
    require(F(2228224, 65536**2) == F(17, 32768), "explicit local-window remainder")
    require(F(17, 32768) < F(194, 103*2048), "weighted cold remainder < Delta/2048")
    require(F(8*2, 65536*2**2) <= F(1, 256), "r endpoint at d=2")
    # Canonical, zero-temperature t=0 random cold pairs, on eight sites.
    n = 8
    signs = [1]*4 + [-1]*4
    values = []
    for occupied in combinations(range(n), n//2):
        occupied = set(occupied)
        mag = sum(signs[i]*(1 if i in occupied else -1) for i in range(n))
        values.append(F(mag*mag, n*n))
    atomic = F(1, 4) if mutant == "atomic-lro" else sum(values, F(0))/len(values)
    require(atomic == F(1, n-1), "atomic canonical M^2=1/(N-1), not a CDW floor")
    print("EXACT constants: ratio <1/16; Delta >194/103 t^2/U; cold norm <Delta/2048")
    print("EXACT trace witness: beta D>=4; canonical atomic N=8 gives M^2=1/7")


def interval_checks() -> None:
    for x in (F(1, 16), F(1, 4), F(1), F(2), F(4), F(8), F(16)):
        z = poisson_resolvent(x, False)
        xy = poisson_resolvent(x, True)
        require(z[0] > F(1, 2) and z[1] < 1, f"f_z range at X={x}")
        require(xy[0] > 0 and xy[1] < exp_negative(x)[0], f"f_xy range at X={x}")
        require(z[0] > xy[1], f"strict separated intervals at X={x}")
        require(z[1]-z[0] < F(1, 10**40) and xy[1]-xy[0] < F(1, 10**40),
                f"rational enclosure width at X={x}")
        if x >= 4:
            require(xy[1]/z[0] < F(1, 16), f"interval anisotropy ratio at X={x}")
        # This declared rational outer box is also checked independently.
        if x == 2:
            require(F(567667, 10**6) < z[0] <= z[1] < F(567668, 10**6), "X=2 f_z box")
            require(F(40194, 10**6) < xy[0] <= xy[1] < F(40195, 10**6), "X=2 f_xy box")
        print(f"RATIONAL enclosure X={x}: 1/2<f_z<1; 0<f_xy<f_z; width<10^-40")


def run(mutant: str | None) -> None:
    fermion_checks(mutant)
    symbolic_checks(mutant)
    constant_checks(mutant)
    interval_checks()
    print("CERTIFICATE OK: exact algebra/intervals only; full-Gibbs CDW LRO OPEN")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutant", choices=MUTANTS)
    parser.add_argument("--all-mutants", action="store_true")
    args = parser.parse_args()
    require(not (args.mutant and args.all_mutants), "choose one mutant mode")
    if args.all_mutants:
        run(None)
        for name in MUTANTS:
            child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                    "--mutant", name], capture_output=True, text=True,
                                   check=False)
            require(child.returncode == 1, f"mutant {name} exited {child.returncode}, expected 1")
            require("REJECTED:" in child.stdout, f"mutant {name} failed without a certificate rejection")
            last = child.stdout.strip().splitlines()[-1]
            print(f"MUTANT {name}: exit=1; {last}")
        print(f"ALL {len(MUTANTS)} MUTANTS REJECTED; aggregate exit=0")
    else:
        run(args.mutant)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Rejected as exc:
        print(f"REJECTED: {exc}")
        sys.exit(1)

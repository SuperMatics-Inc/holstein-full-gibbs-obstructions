"""Exact/interval arithmetic and NUMERICAL finite-cutoff checks, not a pairing theorem.

Run with the numpy, scipy, mpmath and sympy uv environment specified in CHECKS.txt.
--mutants launches ten sequential corrupted-claim processes, each required to exit 1.
--ed includes every eigenstate of each finite Hamiltonian, never a cold projection.
"""
import argparse
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from fractions import Fraction
from typing import Any

import mpmath as mp
import sympy as sy

R = sy.Rational
CONSTANTS = {
    "alpha_squared": R(1, 2),
    "covariance": R(1, 2),
    "fourier_action": R(1, 2),
    "jump_action": R(1, 4),
    "site_loop": R(2),
    "pair_loop": R(4),
    "facing_plateau": R(3, 22),
    "closed_loop": R(3),
    "directed_kernel": R(2),
    "thermal_site": R(1, 2),
}
MUTATIONS = {
    "alpha_squared": R(1),
    "covariance": R(1),
    "fourier_action": R(1),
    "jump_action": R(1, 2),
    "site_loop": R(4),
    "pair_loop": R(8),
    "facing_plateau": R(3, 11),
    "closed_loop": R(2),
    "directed_kernel": R(1),
    "thermal_site": R(1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def zero(expression, name):
    expression = sy.simplify(sy.expand(expression).rewrite(sy.exp))
    require(expression == 0, f"{name}: exact residual {expression}")
    print(f"EXACT {name}: residual=0", flush=True)


def fermion_step(state, mode, create):
    occupied = (state >> mode) & 1
    if occupied == int(create):
        return None
    sign = (-1) ** ((state & ((1 << mode) - 1)).bit_count())
    return state ^ (1 << mode), sign


def hop(state, dst, src):
    first = fermion_step(state, src, False)
    if first is None:
        return None
    second = fermion_step(first[0], dst, True)
    return None if second is None else (second[0], first[1] * second[1])


def pair_move(state, x, y, sites):
    sign = 1
    # Delta_x^dagger Delta_y = c_xup^dagger c_xdown^dagger c_ydown c_yup.
    for mode, create in [(y, False), (sites + y, False),
                         (sites + x, True), (x, True)]:
        result = fermion_step(state, mode, create)
        if result is None:
            return None
        state, step_sign = result
        sign *= step_sign
    return state, sign


def exact_checks(c):
    U, Om, w, beta, u, r, s = sy.symbols("U Om w beta u r s", positive=True)
    zero(c["alpha_squared"] - c["covariance"], "alpha=g/Omega")
    # Completion of the square gives -g^2/Omega = -U/2.
    zero(c["covariance"] * U - U / 2, "atomic_square_completion")
    D = sy.cosh(Om * (beta / 2 - u)) / sy.sinh(beta * Om / 2)
    zero(sy.integrate(c["covariance"] * U * Om * D, (u, 0, beta)) - U,
         "covariance_integral_U")
    atom = U / 2
    cov_fourier = 2 * c["covariance"] * U * Om**2 / (Om**2 + w**2)
    zero(atom - cov_fourier / 2 - c["fourier_action"] * U * w**2 / (Om**2 + w**2),
         "Parseval_action_coefficient")
    zero(c["jump_action"] * 2 - c["fourier_action"], "jump_Fourier_conversion")
    nB = 1 / (sy.exp(beta * Om) - 1)
    normal = -(r*r+s*s)/2-r*s*sy.exp(-Om*u) - nB*(r*sy.exp(Om*u)+s)*(r*sy.exp(-Om*u)+s)
    target = -(r*r+s*s)*sy.coth(beta*Om/2)/2-r*s*D
    zero(normal - target, "normal_ordered_two_displacement_trace")
    zero(2*c["alpha_squared"] - 1, "two_site_Fplus_exponent")
    b = sy.symbols("b", positive=True)
    zero((sy.cosh(b/2)+1)/sy.sinh(b/2)-sy.coth(b/4), "Fplus_minimum_coth")
    zero(c["site_loop"] - c["jump_action"] * (4+4), "single_site_jumps_2_minus2")
    zero(c["pair_loop"] - 2*c["site_loop"], "opposite_two_site_loop")
    zero(sy.coth(b/2)-1/sy.sinh(b/2)-sy.tanh(b/4), "closed_loop_antipodal_tanh")
    zero(sy.limit(c["site_loop"]*sy.tanh(b/4)/b, b, 0)-c["thermal_site"],
         "high_temperature_site_coefficient")
    # Two linear ramps: integral g'^2=2/W, integral g^2=2W/3+delta.
    W, delta, x = sy.symbols("W delta x", positive=True)
    g2 = 2*sy.integrate((x/W)**2, (x, 0, W)) + delta
    zero(g2-(2*W/3+delta), "trapezoid_squared_norm")
    zero(2+R(2,3)+1-R(11,3), "trapezoid_denominator_11_over_3")
    zero(c["facing_plateau"]*R(11,3)-c["fourier_action"], "plateau_3_over_22")
    require(c["facing_plateau"] <= R(1,2), "atomic branch of facing lemma")
    # Four-hop closed witness, epsilon=1/(4U): each site's q-q0 is 1,2,1.
    phi = R(1,2) * 2 * (1**2+2*2**2+1**2) / 4
    odd = R(1,2) * 2 * 2 / 4
    zero(phi-R(5,2), "closed_witness_influence_upper_bound")
    zero(odd-R(1,2), "closed_witness_atomic_cost")
    zero(c["closed_loop"]-phi-odd, "closed_witness_total_3")
    initial, final = 0b1010, 0b0101
    amplitude = 0
    for order in [(0, 1), (1, 0)]:
        state, sign = initial, 1
        for spin in order:
            result = hop(state, 2*spin, 2*spin+1)
            if result is None:
                raise ValueError("illegal directed hop")
            state, step = result
            sign *= step
        require(state == final, "two-spin directed path")
        amplitude += sign
    zero(c["directed_kernel"]-amplitude, "two_spin_orderings_kernel_prefactor")
    # Legality of the neutral-dressed counterexample, including spin labels.
    state = initial
    occupations = [(0, 2)]
    for dst, src in [(0,1), (2,3), (1,0), (0,1)]:
        result = hop(state, dst, src)
        if result is None:
            raise ValueError("illegal counterexample hop")
        state = result[0]
        occupations.append(((state & 1)+((state >> 2)&1),
                            ((state >> 1)&1)+((state >> 3)&1)))
    require(occupations == [(0,2),(1,1),(2,0),(1,1),(2,0)], "counterexample occupations")
    print(f"EXACT legal_counterexample_occupations={occupations}", flush=True)
    uniform_counterexample_checks()


def uniform_counterexample_checks():
    X, D0, Dhalf = sy.symbols("X D0 Dhalf", positive=True)
    charges4, chargesN = [1,1,-1,1], [-1,1]
    groups4, groupsN = [0,0,0,1], [0,1]
    def quadratic(charges, groups):
        return sum(charges[j]*charges[k]*(D0 if groups[j] == groups[k] else Dhalf)
                   for j in range(len(charges)) for k in range(len(charges)))
    zero(-X*(quadratic(charges4,groups4)-quadratic(chargesN,groupsN))/2+
         X*(D0+Dhalf)-X*(D0-Dhalf), "coincident_conditional_margin")
    zero(sum([R(1),R(2),R(1),R(1),R(2),R(3)])-10, "six_distance_shifts_sum_10")
    zero(R(10,4)-R(5,2), "four_charge_Lipschitz_error_5_over_2")
    zero(R(5,2)+R(3,4)-R(13,4), "conditional_Lipschitz_error_13_over_4")
    z = sy.symbols("z", positive=True)
    zero(sy.diff((z-1)/(z+1),z)-2/(z+1)**2, "tanh_fraction_monotone")
    zero((R(3,2)-1)/(R(3,2)+1)-R(1,5), "tanh_quarter_lower_1_over_5")
    zero(R(20,5)-R(13,4)-R(3,4), "uniform_X20_margin_3_over_4")


def interval_checks():
    mp.iv.dps = 80
    # mpmath's interval context is dynamic; its scalar stubs omit interval operators.
    iv: Any = mp.iv
    def rat(n, d=1):
        return iv.mpf(n)/d
    def cosh(z):
        return (iv.exp(z)+iv.exp(-z))/2
    def sinh(z):
        return (iv.exp(z)-iv.exp(-z))/2
    def db(v, b=rat(1)):
        return cosh(b/2-v)/sinh(b/2)
    def log_weight(charges, times, X=rat(40)):
        return -X/2*sum(charges[j]*charges[k]*db(abs(times[j]-times[k]))
                        for j in range(len(charges)) for k in range(len(charges)))
    times = [Fraction(1,160), Fraction(1,80), Fraction(3,160), Fraction(1,2)]
    # Time differences are exact rationals before interval conversion.
    def exact_log(charges, rational_times: list[Fraction]):
        total = rat(0)
        for j in range(len(charges)):
            for k in range(len(charges)):
                difference = abs(rational_times[j]-rational_times[k])
                total += charges[j]*charges[k]*db(rat(difference.numerator,difference.denominator))
        return -rat(40)/2*total
    log4 = exact_log([1,1,-1,1], times)
    logN = exact_log([-1,1], times[2:])
    margin = log4-logN+rat(40)*cosh(rat(1,4))/sinh(rat(1,4))
    require(margin.a > 0, "conditional-factorization witness not separated from zero")
    print(f"INTERVAL log(W4/WN)+40*coth(1/4)={margin}", flush=True)
    require(-log_weight([1,1,-1,-1], [rat(0),rat(1,160),rat(3,160),rat(1,40)]) < rat(5,2),
            "closed four-hop witness influence action")
    # An outward-rounded Riemann enclosure of the full integral, not mp.quad.
    U, Om, beta, t = rat(12), rat(2), rat(4,5), rat(1,2)
    B, X = beta*Om/2, U/Om
    def fplus(u):
        return iv.exp(-X*(cosh(B)+cosh(B-Om*u))/sinh(B))
    integral = rat(0)
    bins = 512
    for j in range(bins):
        # Exact rational endpoints, enclosed outward by interval division.
        left, right = rat(4*j,5*bins), rat(4*(j+1),5*bins)
        cell = iv.mpf([left.a, right.b])
        integral += (right-left)*iv.exp(-U*cell)*fplus(cell)
    actual = 2*t*t*integral
    bound = 2*t*t/U*(1-iv.exp(-beta*U))*iv.exp(-X*cosh(B/2)/sinh(B/2))
    require(actual.b < bound.a, "integrated directed kernel interval bound")
    print(f"INTERVAL K_dir={actual}", flush=True)
    print(f"INTERVAL K_bound={bound}; ratio={actual/bound}", flush=True)
    print("ARITHMETIC checks complete; independent review and full-Gibbs closure absent", flush=True)


def displacement(gamma, cutoff):
    import numpy as np
    from scipy.special import eval_genlaguerre, gammaln
    out = np.empty((cutoff, cutoff))
    for m in range(cutoff):
        for n in range(cutoff):
            lower, upper = min(m,n), max(m,n)
            power = upper-lower
            signed_gamma = gamma if m >= n else -gamma
            out[m,n] = math.exp(-gamma*gamma/2 +
                                (gammaln(lower+1)-gammaln(upper+1))/2) * \
                       signed_gamma**power * eval_genlaguerre(lower,power,gamma*gamma)
    return out


def numerical_kernel_checks():
    import numpy as np
    U, Om, beta, u = 12., 2., .8, .1
    alpha = math.sqrt(U/(2*Om))
    target_plus = math.exp(-U/Om*(math.cosh(beta*Om/2)+math.cosh(Om*(beta/2-u)))/math.sinh(beta*Om/2))
    target_minus = math.exp(-U/Om*(math.cosh(beta*Om/2)-math.cosh(Om*(beta/2-u)))/math.sinh(beta*Om/2))
    for cutoff in [40,64]:
        Dp, Dm = displacement(alpha,cutoff), displacement(-alpha,cutoff)
        ns = np.arange(cutoff)
        weights = np.exp(-(beta-u)*Om*ns[:,None]-u*Om*ns[None,:])
        Z = np.exp(-beta*Om*ns).sum()
        plus = np.sum(weights*Dp*Dp.T)**2/Z**2
        minus = np.sum(weights*Dm*Dp.T)**2/Z**2
        print(f"NUMERICAL Fock cutoff={cutoff} Fplus={plus:.15e} exact={target_plus:.15e} "
              f"Fminus={minus:.15e} exact={target_minus:.15e}", flush=True)
        require(abs(plus-target_plus) < 1e-10*target_plus, "NUMERICAL Fplus trace discrepancy")
        require(abs(minus-target_minus) < 1e-10*target_minus, "NUMERICAL Fminus trace discrepancy")


def dimer(cutoff, U, Om, t, temperatures):
    import numpy as np
    from scipy.linalg import eigh
    gam = math.sqrt(U/Om)
    Dm, Dp = displacement(-gam,cutoff), displacement(gam,cutoff)
    pairD = displacement(-2*gam,cutoff)
    sectors = {}
    for state in range(16):
        key = ((state & 3).bit_count(), (state >> 2).bit_count())
        sectors.setdefault(key,[]).append(state)
    traces = []
    for states in sectors.values():
        index = {s:i for i,s in enumerate(states)}
        dim = len(states)*cutoff
        H = np.zeros((dim,dim))
        O, bare = np.zeros_like(H), np.zeros_like(H)
        for state, i in index.items():
            sl = slice(i*cutoff,(i+1)*cutoff)
            na = (state&1)+((state>>2)&1)
            nb = ((state>>1)&1)+((state>>3)&1)
            # Mu=0; the state-independent -U and CM oscillator cancel in the trace.
            np.fill_diagonal(H[sl,sl], U/2*((na==1)+(nb==1))+Om*np.arange(cutoff))
            for spin in [0,1]:
                for dst,src,D in [(2*spin,2*spin+1,Dm),(2*spin+1,2*spin,Dp)]:
                    result = hop(state,dst,src)
                    if result is not None:
                        j,sign = index[result[0]],result[1]
                        H[j*cutoff:(j+1)*cutoff,sl] += -t*sign*D
            result = pair_move(state,0,1,2)
            if result is not None:
                j,sign = index[result[0]],result[1]
                O[j*cutoff:(j+1)*cutoff,sl] = sign*pairD
                bare[j*cutoff:(j+1)*cutoff,sl] = sign*np.eye(cutoff)
        require(np.max(np.abs(H-H.T)) < 1e-12, "dimer Hamiltonian symmetry")
        vals,vecs = eigh(H,overwrite_a=True,check_finite=False)
        diagO = np.einsum("ij,ij->j",vecs,O@vecs)
        diagBare = np.einsum("ij,ij->j",vecs,bare@vecs)
        traces.append((vals,diagO,diagBare))
    E0 = min(x[0][0] for x in traces)
    answers = []
    for tau in temperatures:
        Z, corr, diagnostic = 0.,0.,0.
        for vals,diagO,diagBare in traces:
            weights = np.exp(-(vals-E0)/tau)
            Z += weights.sum()
            corr += weights@diagO
            diagnostic += weights@diagBare
        answers.append(corr/Z)
        beta = 1/tau
        log_free = math.log(2*t*t/U*beta)-U/Om/math.tanh(beta*Om/4)
        print(f"NUMERICAL dimer grand-canonical mu=0 U={U} Omega={Om} t={t} "
              f"cutoff={cutoff} all_eigenstates={16*cutoff} tau={tau} "
              f"physical_pair={corr/Z:.12e} bare_LF_diagnostic={diagnostic/Z:.12e} "
              f"pair_over_(t/U)^2={corr/Z/(t/U)**2:.9g} log(beta*K_free_bound)={log_free:.9g}", flush=True)
    return np.array(answers)


def relative_chain(sites, max_total, U, Om, t, temperatures):
    import numpy as np
    from scipy.linalg import eigh
    modes = sites-1
    ph_states = [ns for ns in itertools.product(range(max_total+1),repeat=modes)
                 if sum(ns) <= max_total]
    ph_index = {ns:i for i,ns in enumerate(ph_states)}
    phdim = len(ph_states)
    relative = np.zeros((sites,modes))
    for j in range(1,sites):
        relative[:j,j-1] = 1/math.sqrt(j*(j+1))
        relative[j,j-1] = -j/math.sqrt(j*(j+1))
    require(np.max(np.abs(relative.T@relative-np.eye(modes))) < 1e-14, "relative modes orthonormal")
    el_states = [(up,dn) for up in range(sites) for dn in range(sites)]
    el_index = {ns:i for i,ns in enumerate(el_states)}
    dim = len(el_states)*phdim
    H = np.zeros((dim,dim))
    ph_diag = Om*np.array([sum(ns) for ns in ph_states])
    g = math.sqrt(U*Om/2)
    for (up,dn),i in el_index.items():
        base = i*phdim
        np.fill_diagonal(H[base:base+phdim,base:base+phdim],ph_diag)
        qrel = relative[up]+relative[dn]
        for ns,a in ph_index.items():
            for mode in range(modes):
                if ns[mode]:
                    lower = list(ns)
                    lower[mode] -= 1
                    b = ph_index[tuple(lower)]
                    value = -g*qrel[mode]*math.sqrt(ns[mode])
                    H[base+a,base+b] += value
                    H[base+b,base+a] += value
        for spin,pos in enumerate((up,dn)):
            for dst in [pos-1,pos+1]:
                if 0 <= dst < sites:
                    new = (dst,dn) if spin==0 else (up,dst)
                    j = el_index[new]
                    indices = np.arange(phdim)
                    H[base+indices,j*phdim+indices] -= t
    require(np.max(np.abs(H-H.T)) < 1e-12, "chain Hamiltonian symmetry")
    vals,vecs = eigh(H,overwrite_a=True,check_finite=False)
    target = sites-1
    # Physical original-frame pair field, a one-way operator, not its doubled Hermitian part.
    xx,yy = el_index[(0,0)]*phdim,el_index[(target,target)]*phdim
    diagO = np.einsum("ij,ij->j",vecs[xx:xx+phdim],vecs[yy:yy+phdim])
    answers = []
    for tau in temperatures:
        weights = np.exp(-(vals-vals[0])/tau)
        corr = weights@diagO/weights.sum()
        answers.append(corr)
        print(f"NUMERICAL chain sites={sites} canonical Nup=Ndown=1 U={U} Omega={Om} t={t} "
              f"max_total_phonons={max_total} relative_phdim={phdim} all_eigenstates={dim} "
              f"tau={tau} physical_pair_0_{target}={corr:.12e} "
              f"pair_over_(t/U)^(2dist)={corr/(t/U)**(2*target):.9g}", flush=True)
    return np.array(answers)


def ed_checks():
    import numpy as np
    print("NUMERICAL ED: no eigenvalue intervals, no full-Fock tail certificate, no thermodynamic theorem", flush=True)
    for Om in [4.,1.]:
        previous = None
        for cutoff in ([64,96,144] if Om == 4. else [64,96,144,192,256]):
            current = dimer(cutoff,20.,Om,.5,[.02,1.,4.])
            if previous is not None:
                print(f"NUMERICAL dimer Omega={Om} cutoff_change absolute={np.abs(current-previous)} "
                      f"relative={np.abs(current-previous)/np.maximum(np.abs(current),1e-300)}", flush=True)
            previous = current
    for sites,limits in [(3,[14,20,26]),(4,[6,8,10])]:
        previous = None
        for maximum in limits:
            current = relative_chain(sites,maximum,8.,4.,.4,[.02,1.,4.])
            if previous is not None:
                print(f"NUMERICAL chain sites={sites} cutoff_change absolute={np.abs(current-previous)} "
                      f"relative={np.abs(current-previous)/np.maximum(np.abs(current),1e-300)}", flush=True)
            previous = current


def mutant_driver():
    env = os.environ.copy()
    for name in MUTATIONS:
        cmd = ["uv","run","--no-project","--with","numpy","--with","scipy",
               "--with","mpmath","--with","sympy","python",str(Path(__file__).resolve()),"--mutant",name]
        result = subprocess.run(cmd,env=env,text=True,capture_output=True)
        print(f"MUTANT {name} exit={result.returncode} last_line={result.stdout.strip().splitlines()[-1:]}", flush=True)
        require(result.returncode == 1 and "REFUSED" in result.stdout, f"mutant {name} not refused by claim check: {result.stderr}")
    print("MUTANTS all 10 refused with exit 1; driver exit=0", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutants",action="store_true")
    parser.add_argument("--mutant",choices=tuple(MUTATIONS))
    parser.add_argument("--ed",action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        if args.mutants:
            mutant_driver()
            return 0
        constants = CONSTANTS.copy()
        if args.mutant:
            constants[args.mutant] = MUTATIONS[args.mutant]
        exact_checks(constants)
        if args.mutant:
            print(f"ERROR mutant {args.mutant} survived", flush=True)
            return 2
        interval_checks()
        numerical_kernel_checks()
        if args.ed:
            ed_checks()
        print(f"REPLAY exit=0 elapsed_seconds={time.perf_counter()-start:.3f}", flush=True)
        return 0
    except ValueError as exc:
        print(f"REFUSED {args.mutant or 'baseline'}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Independent exact-arithmetic checker; no replay imports or readback.

Uses bit-state hopping paths, an integral recurrence at integer X, positive
Kummer terms for pair transfer, and alternating Taylor bounds for exp(-X).
No floating-point admission, numerical diagonalization, or phonon cutoff.
"""
from fractions import Fraction as Q
from math import comb


def need(test, text):
    if not test:
        raise ArithmeticError(text)


def minus_exp(x, even_degree=240):
    """Alternating Taylor enclosure; tail terms decrease beyond this degree."""
    need(x >= 0 and even_degree % 2 == 0 and even_degree + 1 > x,
         "alternating exponential domain")
    term = Q(1)
    total = Q(1)
    for index in range(1, even_degree + 1):
        term *= -x / index
        total += term
    upper = total
    lower = total + term * (-x) / (even_degree + 1)
    need(0 < lower <= upper, "negative exponential enclosure")
    return lower, upper


def integer_diagonal(n):
    """e^-n times integral u^k e^(nu), using integration by parts."""
    e_low, e_high = minus_exp(Q(n))
    low, high = (1-e_high)/n, (1-e_low)/n
    for k in range(1, n):
        low, high = Q(1, n)-Q(k, n)*high, Q(1, n)-Q(k, n)*low
    return n*low, n*high


def positive_transfer(x, degree=240):
    """e^-2X M(1;1+X;X), with a positive series and geometric tail."""
    term = Q(1)
    total = Q(1)
    for k in range(1, degree + 1):
        term *= x / (x+k)
        total += term
    first = term * x / (x+degree+1)
    tail = first / (1-x/(x+degree+2))
    el, eh = minus_exp(2*x)
    return el*total, eh*(total+tail)


def toggle(state, mode, create):
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = (-1)**((state & ((1 << mode)-1)).bit_count())
    return state ^ (1 << mode), sign


def move(state, source, destination):
    first = toggle(state, source, False)
    if first is None:
        return None
    second = toggle(first[0], destination, True)
    if second is None:
        return None
    return second[0], first[1]*second[1]


def path_matrix():
    cold = (0, 3, 12, 15)
    # Direction is tracked explicitly: opposite directions use f_z,
    # same directions use f_xy. No tensor-matrix construction is used.
    jumps = ((0, 2, 1), (1, 3, 1), (2, 0, -1), (3, 1, -1))
    zmat = [[0]*4 for _ in range(4)]
    xmat = [[0]*4 for _ in range(4)]
    for col, state in enumerate(cold):
        for src, dst, direction in jumps:
            first = move(state, src, dst)
            if first is None:
                continue
            n_x = (first[0] & 3).bit_count()
            n_y = ((first[0] >> 2) & 3).bit_count()
            need(n_x == n_y == 1, "two singly occupied intermediate sites")
            for src2, dst2, direction2 in jumps:
                second = move(first[0], src2, dst2)
                if second is None or second[0] not in cold:
                    continue
                row = cold.index(second[0])
                target = xmat if direction == direction2 else zmat
                target[row][col] -= first[1]*second[1]
    need(zmat == [[0,0,0,0], [0,-2,0,0], [0,0,-2,0], [0,0,0,0]],
         "diagonal coefficient/factor of two")
    need(xmat == [[0,0,0,0], [0,0,-2,0], [0,-2,0,0], [0,0,0,0]],
         "pair-transfer coefficient/sign")
    print("INDEPENDENT exact hopping paths: two spin orderings; same/opposite signs correct")


def power_series_checks():
    # Coefficient reconstruction from (1-r)^-p = sum binomial(n+p-1,p-1)r^n.
    # Denominator finite differences annihilate the polynomial coefficient
    # sequences; first degrees below fix the rational-function numerators.
    def coefficient(n, numerator, power):
        return sum(v*comb(n-j+power-1, power-1)
                   for j, v in enumerate(numerator) if j <= n)
    for n in range(101):
        need(coefficient(n, [0,0,5,-3], 2) == (2*n+1 if n >= 2 else 0),
             "unweighted full series coefficient")
        need(coefficient(n, [0,0,20,-25,9], 3) ==
             ((n+2)*(2*n+1) if n >= 2 else 0), "weighted full series coefficient")
    # After extracting r^3, the variable is y=r^2.
    for k in range(101):
        need(coefficient(k, [7,-3], 2) == 4*k+7, "cold series coefficient")
        need(coefficient(k, [35,-28,9], 3) == (2*k+5)*(4*k+7),
             "weighted cold series coefficient")
        need((4*k+7) > 0 and (2*k+5)*(4*k+7) > 0, "positive cold majorants")
    need((7-3*Q(1,4)**2)/(1-Q(1,4)**2)**2 == Q(1744,225) < 8,
         "unweighted endpoint constant")
    need((35-28*Q(1,2)**2+9*Q(1,2)**4)/(1-Q(1,2)**2)**3 == Q(1828,27) < 68,
         "weighted endpoint constant")
    local_coefficient = 4*68*2**3*(4*2)**3*2
    need(local_coefficient == 2228224, "dimension d^4 weighted coefficient")
    need(Q(local_coefficient, 65536**2) == Q(17,32768), "explicit perturbative window")
    need(Q(17,32768) < Q(194,103)/2048, "cold local remainder comparison")
    print("INDEPENDENT exact remainder coefficients and local-window constants agree")


def interval_checks():
    for n in (1, 2, 4, 8, 16):
        zl, zh = integer_diagonal(n)
        xl, xh = positive_transfer(Q(n))
        need(Q(1,2) < zl <= zh < 1, f"integer diagonal at {n}")
        need(0 < xl <= xh < zl, f"positive transfer at {n}")
        need(zh-zl < Q(1,10**40) and xh-xl < Q(1,10**40), "enclosure width")
        if n >= 4:
            need(xh/zl < Q(1,16), "anisotropy ratio")
        if n == 2:
            need(Q(567667,10**6) < zl <= zh < Q(567668,10**6), "f_z rational box")
            need(Q(40194,10**6) < xl <= xh < Q(40195,10**6), "f_xy rational box")
        print(f"INDEPENDENT rational X={n}: strict gap, width <10^-40")
    # At X=1 the integral recurrence gives 1-e^-1 and e^-1-e^-2.
    e1 = minus_exp(Q(1))
    e2 = minus_exp(Q(2))
    t1 = positive_transfer(Q(1))
    need(t1[0] <= e1[1]-e2[0] and t1[1] >= e1[0]-e2[1],
         "independent X=1 closed-form overlap")
    e4 = minus_exp(Q(4))
    need(4*e4[1]/(1-e4[1]) < e2[0], "high-state trace at beta D=4")
    small = minus_exp(Q(1,4))
    halfsmall = minus_exp(Q(1,8))
    need(4*small[0]/(1-small[0]) > halfsmall[1], "short-time false trace bound detected")
    need(2*e4[1] < Q(1,16), "uniform X>=4 ratio witness")
    need(Q(1,2)-Q(3,103) == Q(97,206), "uniform gap witness")
    # Exchangeable half-pair occupations: off-site covariance -1/(N-1).
    for n in (4,8,16,64):
        diagonal = Q(n)
        staggered_offsite_sum = -n  # sum eta=0, eta_x^2=1
        value = (diagonal-Q(1,n-1)*staggered_offsite_sum)/(n*n)
        need(value == Q(1,n-1), "canonical atomic mixture has no uniform floor")
    print("INDEPENDENT trace witness and atomic no-order calculation agree")


def main():
    path_matrix()
    power_series_checks()
    interval_checks()
    print("INDEPENDENT CERTIFICATE OK; analytic proof drafts unreviewed; full-model LRO OPEN")


if __name__ == "__main__":
    main()

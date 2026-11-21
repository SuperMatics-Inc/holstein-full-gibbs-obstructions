# Replay supplement

Six standalone scripts reproduce every finite and exact-arithmetic claim used in
the paper. Each is a byte copy of the script that was run during checking.
`SHA256SUMS` in this directory lists the SHA-256 of every other file in it, and
the hashes of the six scripts are repeated at the end of this file. None of the
scripts imports another, and none reads any file outside this directory.

| File | Paper section | What it checks |
|---|---|---|
| `replay-kernel.py` | The cutoff-free directed kernel | two-oscillator traces, the directed kernel bound, the influence-action weight of the four-charge counterexample history with its interval margin, the four-hop pair-return loop cost, the facing-plateau constant 3/22, and finite-Fock numerical values of the kernel functions |
| `replay-vtail.py` | The hot sector in the full Gibbs state | the three channel-coefficient limits, exact canonical-anticommutation blocks, the independent Duhamel route on the dimer with the corrected constants, and eight converged two-cutoff grand-canonical dimer exact diagonalizations |
| `replay-cdw.py` | Charge order | the second-order pseudospin coefficients and their signs, the Jacobi and Kummer identities, rational enclosures of the coefficient functions at seven values of the frequency ratio, the local window constants, and the fixed-history trace witnesses; `--all-mutants` additionally rejects twelve deliberately broken variants, each of which must exit 1 |
| `independent-cdw.py` | Charge order | the same coefficient enclosures with a different representation, importing neither `replay-cdw.py` nor its outputs |
| `replay-fullgibbs.py` | The five obstructed route families | the route witnesses: the exact resonance scan, the neutral and pair excursion actions, the one-sided rotation moments, the attractive Hubbard splitting, the pre-Lang-Firsov equal-time correlations, the four reflection kernels on $C_4$, the marginal curvature and variance, the cut-Schrieffer-Wolff leaks, and nine guarded mutants, each of which must exit 1; two of its printed dictionary keys are `hot_tail_U2_wM_over_t2` and `K_beta_ratio_to_bound` |
| `replay-valley.py` | The distortion-valley route | the distortion-coordinate conventions and Born-Oppenheimer surfaces, the explicit ground-doublet thresholds, oscillator-basis exact diagonalization of the dimer against the splitting bound, the Lang-Firsov versus Born-Oppenheimer exponent crossover, the all-temperature excursion bound, the many-body barrier tensorization, and the fixed-distortion three-site projector leakage; `--mutants` must report `ALL 6/6 MUTANTS CAUGHT` |

## How to run

From this directory, with a POSIX shell on any platform that has `uv`:

```
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-kernel.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-vtail.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-cdw.py --all-mutants
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python independent-cdw.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-fullgibbs.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-valley.py
```

Use any cache location you like; `uv` chooses its own by default.
`replay-kernel.py`, `replay-vtail.py`, `replay-fullgibbs.py` and
`replay-valley.py` need SciPy, mpmath and sympy; `replay-cdw.py` and
`independent-cdw.py` need only the standard library, and the extra packages are
harmless for them.

## Expected results

Each command exits with status 0. The last line printed by each, in the order of
the command block above, was:

```
REPLAY exit=0 elapsed_seconds=10.999
PROOF DRAFT: independent analytical review is not supplied by this replay.
ALL 12 MUTANTS REJECTED; aggregate exit=0
INDEPENDENT CERTIFICATE OK; analytic proof drafts unreviewed; full-model LRO OPEN
ALL CHECKS PASSED in 191.92 s.
ALL REPLAY CHECKS PASSED in 1.71s
```

The labels inside those lines are the scripts' own output.
`replay-vtail.py` also prints `PASS: endpoint arithmetic, exact CAR blocks,
independent channels, 8 converged dimer ED points.` and the channel difference
before its closing line; `replay-cdw.py` prints one `MUTANT ... REJECTED` line per
broken variant; `replay-fullgibbs.py` prints one `[CAUGHT] ...` line per mutant.
Running `replay-valley.py --mutants` instead ends with `ALL 6/6 MUTANTS CAUGHT in
1.96s` and also exits 0.

Timings vary widely with machine and load, by more than an order of magnitude on
a busy laptop; the reason strings do not. Two of the heavier scripts dominate:
`replay-vtail.py` spends most of its time in the eight converged dimer
diagonalizations, and `replay-fullgibbs.py` in its nine mutant children.

## What these scripts do not establish

The scripts check finite arithmetic: identities, signs, rational enclosures,
determinants and finite-dimensional diagonalizations. They are not certificates
for any thermodynamic statement, and passing them proves none of the open lemmas
stated in the paper.

## SHA-256 of the copies in this directory

```
9af5e1bed2179fddd360405b75135e65e8f52c8afb6396f84f29d86dc4064254  replay-kernel.py
f6bdf8413c11578e30e4f2eb8fda3eab6c40ce498543c43659a3433d1284f121  replay-vtail.py
eb3ab217ca1804f0beaa1c91103871edbd36641a7849348092990ff429f03827  replay-cdw.py
157ec0e6af5ea8d2e095a9bf71cc56f2acbfb14f4ef1000d11d6690d56fb331a  independent-cdw.py
85d15542392bffd2d45f6bc9ee7480a7c082206fc1686b8c4fa5da8013d321b2  replay-fullgibbs.py
e412c7e3abd46e4da4cc528ee14d6feee8c0e4354795b1e7b094ca03ffd0f430  replay-valley.py
```

# Obstructions to lifting cold-phonon pairing bounds to the full Gibbs state in the Holstein model

Davis Rens, SuperMatics Inc., [ORCID 0000-0001-5839-5197](https://orcid.org/0000-0001-5839-5197)

| | |
| --- | --- |
| Preprint | [10.5281/zenodo.24419885](https://doi.org/10.5281/zenodo.24419885) |
| Replay supplement | [10.5281/zenodo.24419889](https://doi.org/10.5281/zenodo.24419889) |
| Version | 1.0.0, 21 November 2026 |
| Manuscript | [`paper/main.pdf`](paper/main.pdf), 28 pages |

This repository is the tagged release tree of that preprint. It holds the reviewed
manuscript source, the released PDF and the complete replay supplement, pinned byte for
byte through [`MANIFEST.sha256`](MANIFEST.sha256) and [`replay/SHA256SUMS`](replay/SHA256SUMS).

## What the paper does

For the spinful Holstein model at large `U/Omega`, a released argument bounds the polaronic
pair-decay temperature by `2^26 (t^2/U) exp(-7U/(8 Omega))`, but only on a cold manifold
that keeps the odd-site occupation zero and the phonon occupation below
`M = floor(U/(64 Omega))` at every site. Whether the same ceiling holds for the Gibbs state
of the unprojected Hamiltonian is open, and this note does not close it. It records why five
natural route families fail at the level of specified estimates, quantifies what any
surviving route must control, and isolates the remaining statement.

1. **Five obstructed route families.** Operator Schrieffer-Wolff with phonon-number
   Lyapunov bounds, Gaussian domination of the Lang-Firsov influence functional, reflection
   positivity in the phonon coordinate, log-concavity of the phonon marginal
   (Brascamp-Lieb), and cut Schrieffer-Wolff without a global projection are each obstructed
   by an explicit witness at the constants the released route would need. What survives is a
   volume-uniform single-site bound.
2. **The cutoff-free directed kernel.** The directed pair-transfer kernel obeys an exact
   bound carrying the same `exp(-(U/Omega) coth(beta Omega/4))` exponent, but the influence
   weight of an electron history does not factorize over disjoint time segments: an explicit
   legal history with four bond charges has conditional log weight exceeding the free
   directed maximum by at least `3/4` for all `U/Omega >= 20` at `beta Omega = 1`, and a
   closed four-hop pair-return loop costs action at most `3` independently of `U/Omega`. A
   facing-plateau lemma survives with the weaker constant `3/22` and does not add along a
   spatial skeleton.
3. **A sixth route in the distortion coordinate.** Distortion coordinates remove the cold
   projection at the level of the two-site problem and give a non-asymptotic splitting
   bound, an excursion bound at every temperature and a volume-uniform many-body barrier
   bound, with ground-doublet constant `c_0 >= 0.655327` verified in exact outward-interval
   arithmetic. The route stops at the same locality step, because local valley projectors
   leak at order `t^2/U` rather than exponentially.
4. **The hot sector in the full Gibbs state.** In the bare Lang-Firsov frame the population
   of the hot region `N_z > M` is at least `c_2 (t/U)^2` with `c_2 = 1/4096`, on the dimer
   and, with volume-uniform constants, on every torus and every finite vertex-transitive
   regular graph, at every temperature in a fixed window and for all hopping below an
   explicit power of `Omega^3/U^2`; a separate second-order Duhamel calculation on the dimer
   gives the same conclusion with a different constant. The channel weight of a first-order
   virtual transfer tends to `1/4` both in the bare frame and after the first
   Schrieffer-Wolff rotation, so that rotation removes the electronic gap from the
   denominator and leaves the virtual cloud intact.
5. **Charge order, from the other side.** The strong-coupling pseudospin theory of the same
   model is a known easy-axis model; this note adds the first full-Fock transform, an
   explicit local frequency window and a fixed-history trace bound. Positive-temperature
   charge order remains open at a decorated-contour lemma that the nearest rigorous theorem
   cannot supply.

The open problem for the ceiling is stated precisely in the paper as a normalized
charged-skeleton influence bound.

## What the paper does not claim

No pairing ceiling is proved or disproved, and no bound is proved or disproved on any
critical temperature. No material is claimed, no superconductivity is claimed, and no phase
transition is proved or excluded. The five obstructions are obstructions for the specified
estimates at the specified constants, not impossibility statements for the underlying
questions. The finite arithmetic witnesses are checked by the replay scripts; they verify
finite arithmetic, not analysis, and they are not certificates for any thermodynamic
statement.

## Contents

| Path | What it is |
| --- | --- |
| `paper/main.pdf` | The released manuscript, 28 pages. |
| `paper/main.tex` | The LaTeX source that produces that PDF. |
| `paper/refs.bib` | The bibliography. |
| `replay/README.md` | What each replay script checks, how to run it and its byte custody. |
| `replay/replay-kernel.py` | Kernel route: oscillator traces, directed kernel bound, the four-charge counterexample history, the four-hop loop, the facing-plateau constant `3/22`. |
| `replay/replay-vtail.py` | Hot sector: channel limits, exact CAR blocks, the Duhamel route, eight converged dimer diagonalizations. |
| `replay/replay-cdw.py` | Charge order: pseudospin coefficients, enclosures, window constants, trace witnesses, twelve mutants under `--all-mutants`. |
| `replay/independent-cdw.py` | The same enclosures in an independent representation. |
| `replay/replay-fullgibbs.py` | The five obstructed route families, with nine guarded mutants. |
| `replay/replay-valley.py` | The distortion-valley route, with six mutants under `--mutants`. |
| `replay/SHA256SUMS` | Pins for the seven other files of the replay directory. |
| `MANIFEST.sha256` | Pins for every other tracked file of this repository. |
| `CLAIMS.md` | The claim list, with each statement's status and the evidence that supports it. |
| `BUILD.md` | How the PDF is built, with the compiler version, the pinned inputs and the observed log counts. |
| `CITATION.cff` | Citation metadata for the preprint and the supplement. |
| `LICENSE` | Apache-2.0 for the code, CC-BY-4.0 for the manuscript, as stated in its first lines. |

## Reproduce

### The replay

Python 3.12 or later. `replay-cdw.py` and `independent-cdw.py` need only the standard
library; the other four scripts also need NumPy, SciPy, mpmath and sympy:

```sh
python -m pip install numpy scipy mpmath sympy
```

From the `replay/` directory of an unpacked release:

```sh
cd replay
python -B replay-kernel.py
python -B replay-kernel.py --mutants
python -B replay-vtail.py
python -B replay-vtail.py --mutants
python -B replay-cdw.py
python -B replay-cdw.py --all-mutants
python -B independent-cdw.py
python -B replay-fullgibbs.py
python -B replay-valley.py
python -B replay-valley.py --mutants
```

Each command exits `0` and prints nothing on standard error. The closing lines are
`REPLAY exit=0`, `MUTANTS all 10 refused with exit 1; driver exit=0`,
`PASS: endpoint arithmetic, exact CAR blocks, independent channels, 8 converged dimer ED
points.`, `PASS: 8/8 mutants rejected by their designated guards; each exit 1.`,
`CERTIFICATE OK: exact algebra/intervals only; full-Gibbs CDW LRO OPEN`,
`ALL 12 MUTANTS REJECTED; aggregate exit=0`,
`INDEPENDENT CERTIFICATE OK; analytic proof drafts unreviewed; full-model LRO OPEN`,
`ALL CHECKS PASSED in ... s.`, `ALL REPLAY CHECKS PASSED in ...s` and `ALL 6/6 MUTANTS
CAUGHT in ...s`. Timings vary by more than an order of magnitude with machine and load; the
reason strings do not. [`replay/README.md`](replay/README.md) maps every script to the paper
section it checks and lists the observed final lines.

### The paper

```sh
cd paper
SOURCE_DATE_EPOCH=1795132800 tectonic -X compile --keep-logs main.tex
```

`Tectonic 0.17.0` with that fixed epoch reproduces the released PDF; the log has no
overfull box, no underfull box, no undefined reference and no warning. [`BUILD.md`](BUILD.md)
records the compiler, the driver, the pinned epoch and the digest of every build input.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) pins the Tectonic release it
installs, rebuilds the manuscript, checks the rebuilt text against the released PDF,
verifies both manifests, builds the release archive from the commit, unpacks it and runs all
six scripts plus the mutant suites on the unpacked copy, with NumPy, SciPy, mpmath and
sympy installed by `pip`.

## Byte custody

Every released file is a copy of reviewed bytes, and `MANIFEST.sha256` plus
`replay/SHA256SUMS` pin all of them except the two manifests themselves:

```sh
sha256sum -c MANIFEST.sha256
(cd replay && sha256sum -c SHA256SUMS)
```

The deposited supplement archive is `git archive` of the `v1.0.0` tag, so its members are
the files listed by `git ls-files` at that commit, unchanged.

## Licences

The manuscript (`paper/main.tex`, `paper/refs.bib`, `paper/main.pdf`) is licensed under
Creative Commons Attribution 4.0 International (CC-BY-4.0). The replay scripts and all
remaining files are licensed under the Apache License 2.0. The first lines of
[`LICENSE`](LICENSE) state that split; the full Apache-2.0 text follows.

## Review and AI assistance

AI tools were used for drafting, calculation and checking under D. Rens' direction and
review, and separate AI systems carried out independent reviews of the argument and of the
replay scripts, whose comments led to several corrections. This preprint has not been peer
reviewed.

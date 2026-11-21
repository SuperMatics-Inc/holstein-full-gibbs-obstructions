# Claims of `paper/main.pdf`, with status

Each row names one statement of the note, its status and the evidence that supports it.
Status vocabulary:

* **proved** - the note contains a complete analytic proof; reviewed independently for
  the argument.
* **computed** - an exact identity, sign or rational enclosure, in exact or
  outward-interval arithmetic, replayed by a named script.
* **numerical evidence** - converged finite-dimensional diagonalization; never a
  hypothesis of any theorem.
* **open** - stated as an open problem.

| # | Statement | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Cold-projected ceiling of the released argument: `k_B T_pair^cold <= 2^26 (t^2/U) e^{-7U/(8 Omega)}` on the cold manifold | used as an input, not re-derived | cited by DOI, `Section 3` |
| 2 | Five route families are obstructed at the specified estimates: resonant operator Schrieffer-Wolff, Gaussian domination of the influence functional, reflection positivity, Brascamp-Lieb log-concavity, cut Schrieffer-Wolff with a volume-uniform single-site survival | proved, per family, with explicit witnesses | `replay/replay-fullgibbs.py` (nine guarded mutants) |
| 3 | Ground doublet: the distortion-coordinate two-site problem has a ground doublet with threshold constant `c_0 >= 0.655327` | proved, constant computed | `replay/replay-valley.py` (interval enclosures) |
| 4 | All sub-barrier doublets and the counting bound that follows | proved | `replay/replay-valley.py` |
| 5 | Excursion bound at every temperature for the distortion coordinate | proved, constant computed | `replay/replay-valley.py` |
| 6 | Barrier tensorization over sites, volume-uniform | proved, constant computed | `replay/replay-valley.py` |
| 7 | Directed kernel exact bound with exponent `e^{-(U/Omega) coth(beta Omega/4)}` | proved, constants computed | `replay/replay-kernel.py` |
| 8 | Non-factorization: an explicit legal history with four bond charges exceeds the free directed maximum by at least `3/4` for every `U/Omega >= 20` at `beta Omega = 1`, with an interval margin | proved, margin computed | `replay/replay-kernel.py` |
| 9 | Short closed loops: every closed four-hop pair-return loop costs action at most `3`, independently of `U/Omega` | proved, constant computed | `replay/replay-kernel.py` |
| 10 | Facing plateaus: local cost `3UW/22` per facing plateau, which does not add along a spatial skeleton | proved, constant computed | `replay/replay-kernel.py` |
| 11 | Hot sector on the dimer: the population of `N_z > M` is at least `c_2 (t/U)^2` with `c_2 = 1/4096`, at every temperature in a fixed window and for all hopping below an explicit power of `Omega^3/U^2`; the channel coefficient of first-order virtual transfer tends to `1/4` in the bare frame and after the first Schrieffer-Wolff rotation | proved, limits and blocks computed | `replay/replay-vtail.py` (eight converged dimer diagonalizations as numerical evidence) |
| 12 | Hot sector, volume-uniform: the same lower bound on every torus and every finite vertex-transitive regular graph, with constants independent of the volume | proved, by a Duhamel route independent of `Section 4` | `replay/replay-vtail.py` |
| 13 | Charge order: the full-Fock transform of the pseudospin theory, the local frequency window, the fixed-history trace bound, and the observation that the nearest rigorous theorem does not supply the decorated-contour lemma | lemmas proved; the charge-order conclusion is open | `replay/replay-cdw.py` and `replay/independent-cdw.py` |
| - | **Open problem:** a normalized charged-skeleton influence bound that would lift the cold-projected ceiling to the full Gibbs state | open | `Section 8` states the lemma precisely |

## What the scripts do and do not establish

The replay scripts check finite arithmetic: identities, signs, rational enclosures,
determinants and finite-dimensional diagonalizations, in exact rational or
outward-interval arithmetic with floats never used for admission. They are not
certificates for any thermodynamic statement, and passing them proves none of the
analytic steps. `replay/replay-fullgibbs.py`, `replay/replay-cdw.py` and
`replay/replay-valley.py` additionally reject deliberately broken variants of
themselves, nine, twelve and six respectively, each of which must exit with status 1.

## What the note does not claim

No pairing ceiling is proved or disproved for the Gibbs state of the unprojected
Hamiltonian, and no bound is proved or disproved for any critical temperature. No
material is claimed, no superconductivity is claimed, and no phase transition is proved
or excluded. The obstructions concern the specified estimates at the specified constants,
not the impossibility of the underlying questions. No released result changes status.

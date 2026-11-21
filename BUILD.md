# Build and verification record

Version 1.0.0, 21 November 2026. This record covers the released bytes of this repository: the
manuscript `paper/main.pdf`, its sources and the six replay scripts. Every digest below was computed
on the released files themselves, and `MANIFEST.sha256` plus `replay/SHA256SUMS` pin the same bytes.

## Released bytes

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `paper/main.tex` | 103676 | `33a4f4b7fcb6d5c1cb98cc6e83197f183f1cf31e6989a36669ad826431c904ac` |
| `paper/refs.bib` | 7322 | `a0259a202f20c81d9e990ea90911c6e7ac482a48b7fd000c39e9cbd3bd236ff9` |
| `paper/main.pdf` | 275948 | `665157947d8961217954cf75249254f247553a83d4f65597fd7dc9f4edd3fc8f` |
| `replay/replay-kernel.py` | 18931 | `9af5e1bed2179fddd360405b75135e65e8f52c8afb6396f84f29d86dc4064254` |
| `replay/replay-vtail.py` | 18226 | `f6bdf8413c11578e30e4f2eb8fda3eab6c40ce498543c43659a3433d1284f121` |
| `replay/replay-cdw.py` | 11616 | `eb3ab217ca1804f0beaa1c91103871edbd36641a7849348092990ff429f03827` |
| `replay/independent-cdw.py` | 6959 | `157ec0e6af5ea8d2e095a9bf71cc56f2acbfb14f4ef1000d11d6690d56fb331a` |
| `replay/replay-fullgibbs.py` | 43318 | `85d15542392bffd2d45f6bc9ee7480a7c082206fc1686b8c4fa5da8013d321b2` |
| `replay/replay-valley.py` | 26320 | `e412c7e3abd46e4da4cc528ee14d6feee8c0e4354795b1e7b094ca03ffd0f430` |
| `replay/README.md` | 5710 | `39f94fbca5db3cbe1a143c3f9aa2bc221055b8938be1492db8faa1c0ce7c78d2` |
| `replay/SHA256SUMS` | 575 | `4aff440767c456b67b8643c23e6f9e4a6c27839af1f6380931509fca3ee60a1e` |

The PDF has 28 pages. The MD5 of `paper/main.pdf` is `bf4fda2fa124b7581b7506a07e41e42a`. All files
are UTF-8 or binary as appropriate and use LF line endings.

## Manuscript toolchain

- Compiler: Tectonic 0.17.0 (release asset `tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz`,
  SHA-256 `8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`), driving its bundled
  XeTeX engine and `xdvipdfmx`; no local TeX installation is used.
- Document class `article`, 11 pt, margin 1 in, with `amsmath`, `amssymb`, `amsthm`, `mathtools`,
  `hyperref` (`hidelinks,pdfusetitle`), `microtype`, `enumitem`, `array`, `booktabs` and `geometry`,
  all from the bundled distribution.
- Bibliography: `plain` style, `refs.bib`, run by Tectonic's own BibTeX pass.
- Fixed build epoch: `SOURCE_DATE_EPOCH=1795132800` (2026-11-20T00:00:00Z), so that the PDF
  carries no build time; its declared creation date is `D:20261120000000-00'00'`.
- PDF metadata: title equal to the paper title and author `Davis Rens, SuperMatics Inc., ORCID
  0000-0001-5839-5197`, both set by `pdfusetitle` from the manuscript.

Recipe, from the `paper/` directory:

```sh
SOURCE_DATE_EPOCH=1795132800 tectonic -X compile --keep-logs main.tex
```

Observed result: exit code 0; the log holds 0 `Overfull` lines, 0 `Underfull` lines, 0
`undefined` references or citations and no package warning; a second run from the same inputs
reproduces the PDF byte for byte, digest
`665157947d8961217954cf75249254f247553a83d4f65597fd7dc9f4edd3fc8f`. The released PDF is the
reviewed build: the review's findings were applied to `paper/main.tex` before this build, and the
released source, bibliography and PDF are the reviewed bytes unchanged.

## Replay

The six scripts need an interpreter of Python 3.12 or later. `replay-cdw.py` and
`independent-cdw.py` use only the standard library; `replay-kernel.py`, `replay-vtail.py`,
`replay-fullgibbs.py` and `replay-valley.py` also need NumPy, SciPy, mpmath and sympy. From the
`replay/` directory of an unpacked release:

```sh
python -B replay-kernel.py                # exit 0
python -B replay-kernel.py --mutants      # exit 0, 10 mutants refused
python -B replay-vtail.py                 # exit 0
python -B replay-vtail.py --mutants       # exit 0, 8 mutants refused
python -B replay-cdw.py                   # exit 0
python -B replay-cdw.py --all-mutants     # exit 0, 12 mutants refused
python -B independent-cdw.py              # exit 0
python -B replay-fullgibbs.py             # exit 0, 9 guarded mutants caught
python -B replay-valley.py                # exit 0
python -B replay-valley.py --mutants      # exit 0, 6 mutants refused
```

Observed on the released bytes, each with an empty standard error and the following closing
lines: `REPLAY exit=0 elapsed_seconds=1.752`,
`MUTANTS all 10 refused with exit 1; driver exit=0`,
`PASS: endpoint arithmetic, exact CAR blocks, independent channels, 8 converged dimer ED points.`,
`PASS: 8/8 mutants rejected by their designated guards; each exit 1.`,
`CERTIFICATE OK: exact algebra/intervals only; full-Gibbs CDW LRO OPEN`,
`ALL 12 MUTANTS REJECTED; aggregate exit=0`,
`INDEPENDENT CERTIFICATE OK; analytic proof drafts unreviewed; full-model LRO OPEN`,
`ALL CHECKS PASSED in 18.89 s.`, `ALL REPLAY CHECKS PASSED in 0.67s` and
`ALL 6/6 MUTANTS CAUGHT in 0.74s`. Elapsed times vary by more than an order of magnitude with the
machine and its load; the reason strings do not. `replay/README.md` maps every script to the paper
section it checks.

## Continuous integration

`.github/workflows/ci.yml` has two jobs. The `paper` job installs the pinned Tectonic release after
checking the digest of its archive, checks the digests of the released sources and PDF, rebuilds the
manuscript with the epoch above, requires zero overfull boxes, zero underfull boxes, zero undefined
references and no warning in the log, compares the rebuilt text with the released PDF, and checks
the PDF metadata and the disclosure sentence against the manuscript. The `replay` job installs
NumPy, SciPy, mpmath, sympy and uv with `pip` (uv is what the kernel script's mutant driver uses to spawn
its ten child processes), verifies both pin files, checks that `MANIFEST.sha256`
lists every tracked file except itself, builds the release archive with `git archive` from the
commit, unpacks it and runs all six scripts plus the kernel, vtail, charge-order and valley mutant
suites on the unpacked copy. Both jobs run Python 3.12, and the replay job requires each script to
exit `0` with empty standard error and to print its documented closing line, so a script that
warns or exits early fails the run.

## Byte custody

```sh
sha256sum -c MANIFEST.sha256
(cd replay && sha256sum -c SHA256SUMS)
```

`MANIFEST.sha256` lists every tracked file except itself; `replay/SHA256SUMS` lists the seven other
files of the replay directory. Both lists were compared against the released tree with Python's
`hashlib` and report zero mismatches, and the continuous-integration job reruns `sha256sum -c` on
both. The deposited supplement archive is `git archive` of the release tag, so its members are
exactly the files listed by `git ls-files` at that commit.

## Custody notes

The released bytes are copies of the reviewed files and were not edited for packaging. Three
short labels of the internal working directory, each a directory name followed by a packet
name, appear in two hash-frozen scripts: in the module docstrings of `replay/replay-fullgibbs.py` and
`replay/replay-valley.py`, and in one `argparse` description of `replay/replay-fullgibbs.py`. They
are words in module and help text, never filesystem paths, and no released file reads them at run
time. No other released file carries a workspace path or an internal label, and no review, note,
fix or backup file is part of the release.

## Scope

The scripts check finite arithmetic, in exact rational or outward-interval arithmetic with floats
never used for admission. They are not certificates for any thermodynamic statement, and passing
them proves none of the analytic steps of the paper. No pairing ceiling and no bound on any critical
temperature is proved or disproved, and no released result changes status.

"""Check that a released PDF carries the manuscript's title, identifiers and constants.

Usage: python3 ci/verify_text.py <extracted-text-file>

The text file is the extracted text of `paper/main.pdf` (`pdftotext -raw`, or `pypdf`).
Every marker is
compared after folding case and dropping everything except letters and digits, so line
breaks, hyphenation and inter-word spacing in the extraction cannot hide a marker.
"""

import re
import sys

TITLE = "Obstructions to lifting cold-phonon pairing bounds to the full Gibbs state in the Holstein model"

MARKERS = [
    TITLE,
    "Davis Rens",
    "SuperMatics Inc.",
    "0000-0001-5839-5197",
    "This preprint has not been peer reviewed.",

    "Koteck",  # a cited work appears in the bibliography
    "Lang-Firsov",  # the frame in which the hot-sector theorem is stated
    # the ground-doublet constant and the hot-sector constant
    "0.655327",
    "1/4096",
    # the facing-plateau constant
    "3/22",
    # the released programs cited by DOI
    "10.5281/zenodo.24415833",
    "10.5281/zenodo.24414616",
    "10.5281/zenodo.24410350",
    "10.5281/zenodo.24384373",
    "10.5281/zenodo.24374043",
]


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_text.py <extracted-text-file>", file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
        text = norm(handle.read())
    missing = [marker for marker in MARKERS if norm(marker) not in text]
    for marker in MARKERS:
        print(f"{'ok  ' if norm(marker) in text else 'MISS'} {marker}")
    phrase = norm("This preprint has not been peer reviewed.")
    count = text.count(phrase)
    print(f"disclosure phrase occurrences: {count}")
    if count != 1:
        print("the disclosure phrase must appear exactly once", file=sys.stderr)
        return 1
    if missing:
        print(f"missing markers: {missing}", file=sys.stderr)
        return 1
    print(f"all {len(MARKERS)} markers present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

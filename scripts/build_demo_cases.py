"""Build the demo verification set: real arXiv papers plus contrived edge cases.

Real papers come from a late row group of the same pinned metadata revision, far
outside the 2007-2009 training window, so the demo is exercised on papers the
model never saw and on a taxonomy that has since moved. Metadata only: title,
abstract, primary category, identifier. No full text is read.

Edge cases are deliberately unfair. The point is to record what the model does
with them, not to make it look good.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arxiv_router_data as data

LATE_ROW_GROUP = 48  # a much later slice of the same file, outside the training window
REAL_COUNT = 5

EDGE_CASES = [
    {"kind": "edge", "id": "edge-very-short", "note": "Three-word abstract, no subject signal at all.",
     "title": "Notes on a problem",
     "abstract": "Very short. Results forthcoming."},
    {"kind": "edge", "id": "edge-ambiguous", "note": "Title that fits two of the eight areas equally well.",
     "title": "Stochastic estimates of a transport coefficient in a random medium",
     "abstract": "We derive a Markov chain estimator for a diffusion constant and analyse its "
                 "rate of convergence. No experimental data is used."},
    {"kind": "edge", "id": "edge-wrong-domain-1", "note": "Structural biology, nothing like any of the eight areas.",
     "title": "Ribosome profiling reveals a translation pause at a rare codon in yeast",
     "abstract": "Ribo-seq in Saccharomyces cerevisiae shows an uncharacterised pause site. "
                 "We report the codon-level profile and validate the pause with mass spectrometry."},
    {"kind": "edge", "id": "edge-wrong-domain-2", "note": "Climate chemistry, also outside the eight areas.",
     "title": "Aqueous phase chemistry of cloud droplets under marine aerosol conditions",
     "abstract": "We measure the uptake of dimethyl sulfide by sea salt aerosol in a flow tube "
                 "and fit a bulk phase-transfer parameterisation for climate models."},
    {"kind": "edge", "id": "edge-non-english", "note": "French abstract with an English title.",
     "title": "Propagation of gravitational waves in a perturbed spacetime",
     "abstract": "Nous etudions la propagation des ondes gravitationnelles dans un espace-temps "
                 "perturbe et calculons les corrections de second ordre."},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo/verification_cases.jsonl")
    parser.add_argument("--real-count", type=int, default=REAL_COUNT)
    args = parser.parse_args()

    rows = data.fetch_row_groups([LATE_ROW_GROUP], log=lambda *a: None)[LATE_ROW_GROUP]
    dates = sorted(str(row["first_version_date"])[:10] for row in rows if row["first_version_date"])
    print(f"row group {LATE_ROW_GROUP}: {len(rows)} rows, {dates[0]} to {dates[-1]}", file=sys.stderr)

    # Deterministic spread through the row group, with a usable abstract.
    step = max(1, len(rows) // (args.real_count * 40))
    candidates = [row for row in rows[::step]
                  if row["title"] and row["abstract"] and len(row["abstract"].split()) >= 40
                  and "/" not in (row["paper_id"] or "")]
    real = candidates[:args.real_count]

    cases = []
    for row in real:
        cases.append({"id": row["paper_id"], "title": data.strip_reference_cues(row["title"]),
                      "abstract": data.strip_reference_cues(row["abstract"]),
                      "gold_primary_category": row["primary_category"],
                      "first_version_date": str(row["first_version_date"])[:10],
                      "kind": "real"})
    cases.extend(EDGE_CASES)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {target} with {len(real)} real papers and {len(EDGE_CASES)} edge cases", file=sys.stderr)
    for case in cases:
        gold = case.get("gold_primary_category", "none (contrived)")
        print(f"  {case['id']}: {case['title'][:64]!r} gold={gold}", file=sys.stderr)


if __name__ == "__main__":
    main()

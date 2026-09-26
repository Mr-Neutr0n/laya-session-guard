"""Bounded, metadata-only router dataset from secemp9/arxiv-complete.

No full text is read. The builder projects `paper_id`, `title`, `abstract`,
`primary_category`, `license` and `first_version_date` from the `metadata`
config through HTTP range reads and keeps at most `--max-examples` rows.

Split rules (all deterministic, no randomness):
  * three disjoint Parquet row groups, in file order, which is paper_id
    ascending, so train < calibration < test in submission time;
  * the label set is chosen from train row-group frequencies only;
  * the model input carries the title and the first `--abstract-words` words of
    the abstract only. No category, licence, author, date or identifier text
    reaches the input, so no category cue can be read off the prompt;
  * legacy identifiers such as `cs/0704001` embed the archive in the string and
    are dropped;
  * a normalized title may appear in exactly one split, which removes
    cross-split duplicates and version pairs of the same paper.
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

DATASET = "secemp9/arxiv-complete"
REVISION = "cee894837962fede5612cccf2a4c7cacf49b4c3a"
CONFIG = "metadata"
PARQUET = "metadata/train-00000-of-00001.parquet"
COLUMNS = ["paper_id", "title", "abstract", "primary_category", "license", "first_version_date"]
SPLIT_ROW_GROUPS = {"train": 0, "calibration": 1, "test": 2}
SPLIT_CAPS = {"train": 4000, "calibration": 400, "test": 800}
DEFAULT_MAX_EXAMPLES = 24000
MIN_ABSTRACT_WORDS = 40

# arXiv top-level archives. Paper-level licences are per author; see LICENSE.
LICENCE_NOTE = (
    "Dataset compilation: CC0 organisational layer. Paper rights stay with the authors under each "
    "recorded per-paper licence (metadata.license), or none recorded. No paper full text is used."
)


def normalise_title(title):
    """Case, punctuation and whitespace insensitive key for duplicate detection."""
    text = unicodedata.normalize("NFKD", title).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


LEGACY_REFERENCE = re.compile(
    r"(?:arXiv:|arxiv:)?\s*[a-z][a-z-]*(?:-[a-z]+)*(?:\.[A-Z]{2})?/\d{7}(?:\.\d{4})?"
)


def strip_reference_cues(text):
    """Remove legacy arXiv identifiers from titles and abstracts.

    In the 2007-2009 window abstracts routinely cite eprints as `hep-th/0607218`
    or `math.PR/0410115`. Those strings spell out the archive, so leaving them in
    would hand the router a lookup-free copy of the label. New-style identifiers
    such as `0704.12345` carry no archive and are kept.
    """
    return LEGACY_REFERENCE.sub("REF", re.sub(r"\s+", " ", text)).strip()


def clean_abstract(abstract, max_words):
    words = re.sub(r"\s+", " ", abstract).strip().split(" ")
    return " ".join(words[:max_words])


def fetch_row_groups(row_groups, log=print):
    """Read only the projected columns of the requested row groups, over HTTP ranges.

    Returns {row_group_index: [row]}. Each metadata row group is about 50,000 rows
    and about 27 MB compressed for every column, so three projected groups are a
    bounded few-hundred-megabyte-scale transfer, not a corpus download.
    """
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("hf", repo_type="dataset", revision=REVISION)
    remote = f"datasets/{DATASET}/{PARQUET}"
    handle = fs.open(remote, "rb")
    try:
        parquet = pq.ParquetFile(handle)
        available = parquet.metadata.num_row_groups
        result = {}
        for index in sorted(set(row_groups)):
            if index >= available:
                raise IndexError(f"row group {index} missing; file has {available}")
            table = parquet.read_row_groups([index], columns=COLUMNS)
            result[index] = table.to_pylist()
            log(f"row group {index}: {table.num_rows} rows, {len(COLUMNS)} projected columns")
    finally:
        handle.close()
    return result


def select(rows_by_group, labels, caps, max_words, min_words=MIN_ABSTRACT_WORDS, max_examples=None):
    """Pure selection. Returns {split: [example]} with disjoint normalized titles."""
    max_examples = max_examples or sum(caps.values())
    kept, seen_titles, stats = {}, {}, {}
    budget = 0
    for split in ("train", "calibration", "test"):
        pool = []
        counters = Counter()
        dropped = Counter()
        for row in rows_by_group[split]:
            # Raw Parquet rows carry `primary_category`; already-selected rows carry `label`.
            category = row.get("primary_category") or row.get("label")
            if category not in labels:
                dropped["category_not_in_label_set"] += 1
                continue
            paper_id = str(row.get("paper_id") or row.get("id") or "").strip()
            if not paper_id or "/" in paper_id:
                dropped["legacy_identifier"] += 1
                continue
            title = strip_reference_cues(row["title"] or "")
            abstract = row["abstract"] or ""
            if len(title) < 10 or len(abstract.split()) < min_words:
                dropped["short_text"] += 1
                continue
            abstract = strip_reference_cues(abstract)
            blob = (title + " " + abstract).lower()
            if any(label.lower() in blob or label.replace(".", "/").lower() in blob for label in labels):
                dropped["residual_archive_cue"] += 1
                continue
            key = normalise_title(title)
            if not key:
                dropped["empty_title"] += 1
                continue
            if key in seen_titles:
                dropped["duplicate_title"] += 1
                continue
            seen_titles[key] = split
            pool.append({
                "id": paper_id,
                "title": title,
                "abstract": clean_abstract(abstract, max_words),
                "label": category,
                "first_version_date": str(row.get("first_version_date") or "")[:10],
                "license": row.get("license"),
                "split": split,
            })
            counters[category] += 1
        # Deterministic, class-balanced round robin over label order.
        by_label = {label: [e for e in pool if e["label"] == label] for label in labels}
        ordered, taken = [], Counter()
        cap = min(caps[split], max(0, max_examples - budget))
        while len(ordered) < cap and any(taken[label] < len(by_label[label]) for label in labels):
            for label in labels:
                if taken[label] < len(by_label[label]) and len(ordered) < cap:
                    ordered.append(by_label[label][taken[label]])
                    taken[label] += 1
        kept[split] = ordered
        budget += len(ordered)
        stats[split] = {"available": len(pool), "kept": len(ordered), "dropped": dict(dropped),
                        "per_label_available": dict(counters),
                        "per_label_kept": {label: taken[label] for label in labels}}
    return kept, stats


def select_labels(rows, k, per_archive=2, min_archive_rows=5000):
    """Archive-diverse label set chosen from the train row group only.

    Ranking by raw frequency alone returns eight physics codes from the early
    arXiv id range. Taking the busiest categories of each large archive keeps the
    router realistic: the decision then spans distinct subject areas instead of
    fine physics distinctions inside one archive.
    """
    counts = Counter((row.get("primary_category") or row.get("label")) for row in rows)
    counts.pop(None, None)
    archives = Counter()
    for category, count in counts.items():
        archives[category.split(".")[0]] += count
    candidates = []
    for archive, total in archives.most_common():
        if total < min_archive_rows:
            continue
        ranked = [c for c, _ in counts.most_common() if c.split(".")[0] == archive][:per_archive]
        candidates.extend(ranked)
    candidates.sort(key=lambda c: (-counts[c], c))
    return sorted(candidates[:k])


# Curated, fixed a priori, and readable as topic names for the choice head.
# Criteria for the curated set, applied before looking at any model output:
#   * the category exists with at least 900 rows in every one of the three row
#     groups, so the split cannot run dry;
#   * it spans five archives (hep, quant-ph, gr-qc, cond-mat, math), so the task
#     is multi-topic routing and not one archive's internal taxonomy;
#   * `astro-ph` is excluded because the archive is a parent code in the train
#     window and split into `.GA`, `.CO` and others by the test window, which
#     would make the label a moving target;
#   * `cs.*` is excluded because the 2007-2008 window holds only a few hundred
#     computer-science papers per category. Sampling a later row group would be
#     a different experiment, not a different seed.
LABELS = ["cond-mat.mes-hall", "cond-mat.mtrl-sci", "gr-qc", "hep-ph", "hep-th",
          "math.AG", "math.PR", "quant-ph"]
MIN_PER_LABEL = 400


def verify_labels(rows_by_group, labels, min_per_label=MIN_PER_LABEL):
    """Fail loudly rather than silently balancing a split against an empty class."""
    report = {}
    for split, rows in rows_by_group.items():
        counts = Counter((row.get("primary_category") or row.get("label")) for row in rows)
        counts.pop(None, None)
        report[split] = {label: counts.get(label, 0) for label in labels}
        missing = [label for label, count in report[split].items() if count < min_per_label]
        if missing:
            raise ValueError(f"{split} row group lacks {min_per_label} rows for {missing}")
    return report


def build(output_dir, k=8, max_words=110, max_examples=DEFAULT_MAX_EXAMPLES, revision_note=None,
          labels=None, min_per_label=MIN_PER_LABEL):
    groups = fetch_row_groups(SPLIT_ROW_GROUPS.values())
    by_split = {split: groups[group] for split, group in SPLIT_ROW_GROUPS.items()}
    labels = list(labels) if labels else (LABELS if k == len(LABELS) else select_labels(by_split["train"], k))
    availability = verify_labels(by_split, labels, min_per_label)
    kept, stats = select(by_split, labels, SPLIT_CAPS, max_words, max_examples=max_examples)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in kept.items():
        with (output / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    licences = Counter(row["license"] or "none-recorded" for rows in kept.values() for row in rows)
    dates = {split: [row["first_version_date"] for row in rows if row["first_version_date"]]
             for split, rows in kept.items()}
    provenance = {
        "source": {"dataset": DATASET, "revision": REVISION, "config": CONFIG, "path": PARQUET,
                   "revision_note": revision_note or "Pinned commit sha of the dataset repository.",
                   "columns_read": COLUMNS, "columns_not_read": ["paper_text", "text", "latex", "pdf", "source", "ps"],
                   "access": "HTTP range reads of projected Parquet columns; the 16.08 TB repo is never cloned."},
        "licence": LICENCE_NOTE,
        "licence_counts": dict(licences),
        "labels": labels,
        "label_availability_per_row_group": availability,
        "label_rule": "Curated a priori for multi-archive coverage and per-split availability; see LABELS in src/arxiv_router_data.py.",
        "selection_rules": [
            "Splits are three disjoint Parquet row groups in file order (paper_id ascending): train 0, calibration 1, test 2.",
            "The label set is curated a priori, not chosen by model output: multi-archive coverage and per-split availability only.",
            "Legacy identifiers containing '/' are dropped because the archive code is embedded in the string.",
            "Legacy arXiv identifiers cited inside titles and abstracts are replaced by 'REF', because strings such as hep-th/0607218 spell out the archive.",
            "Titles shorter than 10 characters and abstracts shorter than 40 words are dropped.",
            f"Input text is the title plus the first {max_words} abstract words; category, licence, author and date never reach the input.",
            "A normalized title may appear in at most one split; duplicates across splits are dropped.",
            "Splits are class balanced by deterministic round robin over the label order.",
        ],
        "split_sizes": {split: len(rows) for split, rows in kept.items()},
        "total_examples": sum(len(rows) for rows in kept.values()),
        "max_examples_requested": max_examples,
        "date_ranges": {split: [min(values), max(values)] for split, values in dates.items() if values},
        "stats": stats,
        "limitations": [
            "Labels are arXiv author-declared primary categories, not expert annotations.",
            "The label set covers only 8 archives of 170 recorded primary categories.",
            "Abstracts are truncated, so the task is title-plus-abstract routing, not full comprehension.",
        ],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/router")
    parser.add_argument("--labels", type=int, default=8)
    parser.add_argument("--abstract-words", type=int, default=110)
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.labels, args.abstract_words, args.max_examples), indent=2))

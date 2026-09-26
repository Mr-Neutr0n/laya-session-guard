"""Router task definition: title and abstract in, one arXiv subject area out.

Pure stdlib so it is testable without torch. The state handed to Laya contains
the title and the truncated abstract and nothing else. No category code, licence,
author, date or identifier is present, so the label cannot be read off the input.
"""
import json
import re
from pathlib import Path

DESCRIPTIONS = {
    "cond-mat.mes-hall": "Condensed matter: mesoscopic transport, Hall and magnetotransport, quantum dots.",
    "cond-mat.mtrl-sci": "Condensed matter materials: surfaces, thin films, nanostructures, growth.",
    "gr-qc": "General relativity and quantum cosmology: gravity, black holes, spacetime.",
    "hep-ph": "High energy phenomenology: colliders, the Standard Model, neutrino experiments.",
    "hep-th": "High energy theory: field theory, strings, supersymmetry, quantum gravity.",
    "math.AG": "Mathematics: algebraic and differential geometry, topology, algebra.",
    "math.PR": "Mathematics: probability, stochastic processes, statistics.",
    "quant-ph": "Quantum physics: quantum information, computing, optics, open systems.",
}

INSTRUCTIONS = (
    "Route the preprint to its primary arXiv subject area. Judge the title and abstract only, "
    "and choose exactly one area."
)


def label_description(label):
    return DESCRIPTIONS.get(label, f"Primary arXiv subject area {label}.")


def paper_state(row):
    """Model input. Deliberately excludes label, licence, date and identifier."""
    return {"title": row["title"], "abstract": row["abstract"]}


def router_question(labels):
    return {"t": "choice", "ins": INSTRUCTIONS,
            "crit": {label: label_description(label) for label in labels}}


def assert_no_cue(row, labels, state=None):
    """Fail if the input text could carry the answer directly."""
    state = paper_state(row) if state is None else state
    if set(state) != {"title", "abstract"}:
        raise ValueError(f"unexpected state fields: {sorted(state)}")
    blob = json.dumps(state, ensure_ascii=False).lower()
    for label in labels:
        if label.lower() in blob or label.replace(".", "/").lower() in blob:
            raise ValueError(f"input text contains the label cue {label!r}")
    # A legacy identifier embeds the archive, so its presence in the text is a
    # cue. A new-style identifier such as 0706.3050 carries no archive, and
    # papers citing their own eprint number leak nothing, so it is left alone.
    if "/" in row["id"] and row["id"].lower() in blob:
        raise ValueError("input text contains a legacy arXiv identifier")
    return True


def load_split(path, labels):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        assert_no_cue(row, labels)
    return rows


def build_items(tok, rows, labels, max_len=384, head_max_len=192):
    """Tokenize one choice sequence per paper and return training-ready items."""
    from laya.common import build_sequence

    question = router_question(labels)
    order = {label: index for index, label in enumerate(labels)}
    items = []
    for row in rows:
        assert_no_cue(row, labels)
        state = paper_state(row)
        ids, markers = build_sequence(tok, state, question, max_len, head_max_len)
        if len(markers) != len(labels):
            raise ValueError("question options were truncated; raise head_max_len")
        if len(ids) >= max_len:
            raise ValueError(f"sequence hit the {max_len} token budget; shorten the abstract")
        items.append({"ids": ids, "markers": markers, "id": row["id"], "label": order[row["label"]],
                      "category": row["label"]})
    return items


def normalise_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

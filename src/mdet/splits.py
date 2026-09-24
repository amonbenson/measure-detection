"""Creates dataset/meta.json and the document splits (guideline sections 5.1 and 5.7).

Splits are by document, stratified by show, with a fixed seed. test.txt is frozen: this script
refuses to overwrite existing split files unless --force is given.
Usage: uv run python -m mdet.splits
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np

from .data import DATASET, SPLITS, all_document_ids, load_document, read_split

# Show-level metadata established during the audit (docs/CONVENTIONS.md), keyed by the show id in
# source.json. Documents store no show names; `show` in meta.json is the show id. Shows added later
# (curation page) have no audit metadata.
SHOW_META = {
    # show A
    "ufdze5y4yx1yror": {
        "score_type": "piano-conductor",
        "publisher": "show-a",  # neutral keys used in docs/RESULTS.md
        "source": "flatbed-scan",
        "scan_dpi": 600,
        "numbering_style": "every-measure",
        "number_position": "below",
        "has_cue_staves": True,
        "max_vocal_staves": 1,
        "multi_song": False,
    },
    # show B
    "oyy8yysryp55587": {
        "score_type": "vocal",
        "publisher": "show-b",
        "source": "born-digital",
        "scan_dpi": None,
        "numbering_style": "every-measure",
        "number_position": "above",
        "has_cue_staves": False,
        "max_vocal_staves": 2,
        "multi_song": False,
    },
}


# Documents whose file source differs from the rest of their show.
DOC_OVERRIDES = {
    "5bu88s20mzndpax": {"source": "flatbed-scan-strips", "scan_dpi": None, "notes": "A5 page assembled from image strips"},
}


# A start boundary closer than this to the system's opening line is at the system edge, not after the header.
EDGE_START_S = 2.0


def edge_starts(doc_id: str) -> int | None:
    """Data check for the start convention (docs/CONVENTIONS.md: the first measure of a system starts
    after clef, key and time signature). Number of rows whose annotated start lies within EDGE_START_S
    of the detected system start; None if strips are not built yet."""
    from .targets import load_samples

    files = sorted((DATASET / "derived" / "strips" / doc_id).glob("*.npz"))
    off = [(s["boundaries"][0] - s["strip"]["system_u0"]) / 8.0 for f in files for s in load_samples(f)]
    if not off:
        return None
    return int(sum(o < EDGE_START_S for o in off))


def build_meta() -> dict[str, dict]:
    meta = {}
    labeled_ids = set(all_document_ids(labeled_only=True))
    for doc_id in all_document_ids(labeled_only=False):
        src = json.loads((DATASET / "documents" / doc_id / "source.json").read_text(encoding="utf-8"))
        show = src["show_id"]
        m = dict(SHOW_META.get(show, {}))
        m.update({"show": show, "edition_group": show, "number": src.get("number", "")})
        m.update(DOC_OVERRIDES.get(doc_id, {}))
        labeled = doc_id in labeled_ids
        m["labeled"] = labeled
        if labeled:
            doc = load_document(doc_id)
            m["has_alphanumeric_labels"] = any(not ms.label.isdigit() for ms in doc.measures)
            m["num_measures"] = len(doc.measures)
            m["num_pages"] = doc.num_pages
            m["edge_starts"] = edge_starts(doc_id)
        meta[doc_id] = m
    return meta


def build_splits(meta: dict[str, dict], seed: int = 20260923) -> dict[str, list[str]]:
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for show in sorted({m["show"] for m in meta.values()}):
        docs = sorted(d for d, m in meta.items() if m["show"] == show and m["labeled"])
        rng.shuffle(docs)
        n_eval = round(0.15 * len(docs))
        splits["test"] += docs[:n_eval]
        splits["val"] += docs[n_eval : 2 * n_eval]
        splits["train"] += docs[2 * n_eval :]
    return {k: sorted(v) for k, v in splits.items()}


def build_challenge() -> list[str]:
    """Pages of test documents selected by rule (guideline 5.7): alphanumeric labels, multi-measure
    rests (virtual boundaries), skew above 1 degree, or scan shear above 0.01 (0.6 degrees)."""
    from .geometry import PageGeometry
    from .targets import load_samples

    out = []
    for d in read_split("test"):
        doc = load_document(d)
        for p in range(doc.num_pages):
            rows = doc.rows_on_page(p)
            if not rows:
                continue
            reasons = []
            if any(not m.label.isdigit() for r in rows for m in r.measures):
                reasons.append("alphanumeric")
            samples = load_samples(DATASET / "derived" / "strips" / d / f"{p:03d}.npz")
            if any(np.asarray(s["virtual"]).any() for s in samples):
                reasons.append("multi-measure-rest")
            if abs(PageGeometry.load(d, p).info["deskew_deg"]) > 1.0:
                reasons.append("skew")
            if any(abs(s["strip"]["shear"]) > 0.01 for s in samples):
                reasons.append("shear")
            if reasons:
                out.append(f"{d} {p} {','.join(reasons)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--challenge", action="store_true", help="write challenge.txt (needs derived strips)")
    args = ap.parse_args()
    if args.challenge:
        lines = build_challenge()
        (SPLITS / "challenge.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"challenge: {len(lines)} pages")
        return
    meta = build_meta()
    (DATASET / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    SPLITS.mkdir(parents=True, exist_ok=True)
    if (SPLITS / "test.txt").exists() and not args.force:
        # New documents are added to train or val by the curation page, never here.
        assigned = {d for name in ("train", "val", "test") for d in read_split(name)}
        missing = sorted(d for d, m in meta.items() if m["labeled"] and d not in assigned)
        no_strips = sorted(d for d, m in meta.items() if m["labeled"] and m.get("edge_starts") is None)
        edge = sorted((d, m["edge_starts"]) for d, m in meta.items() if m["labeled"] and m.get("edge_starts"))
        print(f"meta.json: {len(meta)} documents, {len({m['show'] for m in meta.values()})} shows")
        if missing:
            print(f"{len(missing)} labeled documents are in no split: {' '.join(missing)}")
        if no_strips:
            print(f"{len(no_strips)} documents have no strips yet (run mdet.prepare and mdet.targets first): {' '.join(no_strips)}")
        if edge:
            print(f"{len(edge)} documents have rows that start at the system edge instead of after the header: " + " ".join(f"{d} ({n})" for d, n in edge))
        print("splits exist; test split is frozen (use --force to recreate)")
        return
    for name, ids in build_splits(meta).items():
        (SPLITS / f"{name}.txt").write_text("\n".join(ids) + "\n", newline="\n")
        n_meas = sum(meta[d]["num_measures"] for d in ids)
        shows = {s: sum(meta[d]["show"] == s for d in ids) for s in sorted({m["show"] for m in meta.values()})}
        print(f"{name}: {len(ids)} documents, {n_meas} measures, {shows}")


if __name__ == "__main__":
    main()

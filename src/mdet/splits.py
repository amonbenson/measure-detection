"""Creates dataset/meta.json and the document splits (guideline sections 5.1 and 5.7).

Splits are by document, stratified by show, with a fixed seed. test.txt is frozen: this script
refuses to overwrite existing split files unless --force is given.
Usage: uv run python -m mdet.splits
"""

from __future__ import annotations

import argparse
import json
import random

from .data import DATASET, SPLITS, all_document_ids, load_document

# Show-level metadata, established during the audit (docs/CONVENTIONS.md).
SHOW_META = {
    "show A": {
        "score_type": "piano-conductor",
        "publisher": "show-a",
        "source": "flatbed-scan",
        "scan_dpi": 600,
        "numbering_style": "every-measure",
        "number_position": "below",
        "has_cue_staves": True,
        "max_vocal_staves": 1,
        "multi_song": False,
    },
    "show B": {
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


def build_meta() -> dict[str, dict]:
    meta = {}
    for doc_id in all_document_ids(labeled_only=False):
        src = json.loads((DATASET / "documents" / doc_id / "source.json").read_text(encoding="utf-8"))
        m = dict(SHOW_META[src["show"]])
        m.update({"show": src["show"], "song_number": src["number"], "title": src["title"], "edition_group": src["show"]})
        m.update(DOC_OVERRIDES.get(doc_id, {}))
        labeled = doc_id in set(all_document_ids(labeled_only=True))
        m["labeled"] = labeled
        if labeled:
            doc = load_document(doc_id)
            m["has_alphanumeric_labels"] = any(not ms.label.isdigit() for ms in doc.measures)
            m["num_measures"] = len(doc.measures)
            m["num_pages"] = doc.num_pages
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    meta = build_meta()
    (DATASET / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    SPLITS.mkdir(parents=True, exist_ok=True)
    if (SPLITS / "test.txt").exists() and not args.force:
        print("splits exist; test split is frozen (use --force to recreate)")
        return
    for name, ids in build_splits(meta).items():
        (SPLITS / f"{name}.txt").write_text("\n".join(ids) + "\n", newline="\n")
        n_meas = sum(meta[d]["num_measures"] for d in ids)
        shows = {s: sum(meta[d]["show"] == s for d in ids) for s in SHOW_META}
        print(f"{name}: {len(ids)} documents, {n_meas} measures, {shows}")


if __name__ == "__main__":
    main()

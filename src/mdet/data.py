"""Dataset loading.

Converts the raw measure list of a document (a playback sequence) into printed measures
grouped into rows in reading order. See docs/CONVENTIONS.md for the rules applied here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
DOCUMENTS = DATASET / "documents"
DERIVED = DATASET / "derived"
SPLITS = DATASET / "splits"
RUNS = ROOT / "runs"

VALUE_RE = re.compile(r"^(?P<label>.*?)(?:-(?P<pass>\d+))?$")
LABEL_RE = re.compile(r"^(?P<number>\d*)(?P<suffix>[A-Za-z]*)$")


def parse_value(value: str) -> tuple[str, int]:
    """Split a JSON value into (label, pass). `23A-2` -> (`23A`, 2); `23` -> (`23`, 1)."""
    m = VALUE_RE.match(value.strip())
    assert m is not None
    return m["label"], int(m["pass"]) if m["pass"] else 1


def parse_label(label: str) -> tuple[int | None, str] | None:
    """Parse a label into (number, suffix). Returns None if the label is not in the grammar."""
    m = LABEL_RE.match(label)
    if m is None or (not m["number"] and not m["suffix"]):
        return None
    return (int(m["number"]) if m["number"] else None), m["suffix"]


@dataclass
class Measure:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    label: str
    value: str
    index: int  # position in the raw JSON array

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


@dataclass
class Row:
    page: int
    measures: list[Measure]

    @property
    def y0(self) -> float:
        return min(m.y0 for m in self.measures)

    @property
    def y1(self) -> float:
        return max(m.y1 for m in self.measures)

    @property
    def x0(self) -> float:
        return self.measures[0].x0

    @property
    def x1(self) -> float:
        return self.measures[-1].x1

    def boundaries(self) -> list[float]:
        """Start boundary followed by every barline boundary (page-relative x)."""
        return [self.measures[0].x0] + [m.x1 for m in self.measures]


@dataclass
class Document:
    doc_id: str
    source: dict
    num_pages: int
    rows: list[Row]
    dropped: dict = field(default_factory=dict)  # counts of entries removed by the loader

    @property
    def measures(self) -> list[Measure]:
        return [m for r in self.rows for m in r.measures]

    def rows_on_page(self, page: int) -> list[Row]:
        return [r for r in self.rows if r.page == page]

    @property
    def pdf(self) -> Path:
        return DOCUMENTS / self.doc_id / "score.pdf"


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-12, union)


def _vertical_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Intersection over the smaller of the two heights."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-9, min(a1 - a0, b1 - b0))


def load_raw(doc_id: str) -> list[dict]:
    return json.loads((DOCUMENTS / doc_id / "measures.json").read_text(encoding="utf-8"))


def page_count(doc_id: str) -> int:
    import pymupdf

    with pymupdf.open(DOCUMENTS / doc_id / "score.pdf") as pdf:
        return len(pdf)


def load_document(doc_id: str, num_pages: int | None = None) -> Document:
    raw = load_raw(doc_id)
    source = json.loads((DOCUMENTS / doc_id / "source.json").read_text(encoding="utf-8"))
    dropped = {"no_layout": 0, "duplicate_box": 0}

    # 1. Keep entries with a layout. Entries without a layout are later playback passes or the end marker.
    measures: list[Measure] = []
    for i, entry in enumerate(raw):
        lay = entry.get("layout")
        if not lay:
            dropped["no_layout"] += 1
            continue
        label, _ = parse_value(entry["value"])
        measures.append(
            Measure(
                page=int(lay["page"]),
                x0=lay["x"],
                y0=lay["y"],
                x1=lay["x"] + lay["width"],
                y1=lay["y"] + lay["height"],
                label=label,
                value=entry["value"],
                index=i,
            )
        )

    # 2. Duplicate boxes: the same printed measure annotated twice (a passage under two names).
    #    Boxes on one page with IoU >= 0.8 are duplicates. The later entry wins; it matches the print.
    kept: list[Measure] = []
    for m in measures:
        dup = next((k for k in kept if k.page == m.page and box_iou(k.box, m.box) >= 0.8), None)
        if dup is not None:
            dropped["duplicate_box"] += 1
            kept.remove(dup)
        kept.append(m)
    measures = sorted(kept, key=lambda m: m.index)

    # 3. Rows: measures on one page whose vertical extents overlap by more than half the smaller height.
    rows: list[Row] = []
    for m in sorted(measures, key=lambda m: (m.page, (m.y0 + m.y1) / 2)):
        for r in rows:
            if r.page == m.page and _vertical_overlap(r.y0, r.y1, m.y0, m.y1) > 0.5:
                r.measures.append(m)
                break
        else:
            rows.append(Row(page=m.page, measures=[m]))
    for r in rows:
        r.measures.sort(key=lambda m: m.x0)
    rows.sort(key=lambda r: (r.page, r.y0))

    if num_pages is None:
        num_pages = page_count(doc_id)
    return Document(doc_id=doc_id, source=source, num_pages=num_pages, rows=rows, dropped=dropped)


def all_document_ids(labeled_only: bool = True) -> list[str]:
    ids = sorted(p.name for p in DOCUMENTS.iterdir() if (p / "measures.json").exists())
    if labeled_only:
        ids = [d for d in ids if any(e.get("layout") for e in load_raw(d))]
    return ids


def read_split(name: str) -> list[str]:
    return [line.strip() for line in (SPLITS / f"{name}.txt").read_text().splitlines() if line.strip()]


def read_meta() -> dict[str, dict]:
    return json.loads((DATASET / "meta.json").read_text(encoding="utf-8"))

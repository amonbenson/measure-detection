# Annotation conventions

Established by the data audit (`uv run python -m mdet.audit`, report in `docs/AUDIT.md`). Terms are defined in `docs/PLAN.md`, section 1.

## Source format

Each document is one song from the Pocketbase `songs` collection: `score.pdf` plus `measures.json`, a JSON array of entries `{ beats, layout?, value }`. `beats` is ignored.

## Coordinates

- `layout.x`, `layout.y`, `layout.width`, `layout.height` are fractions of the displayed page: origin top-left, y down, after `/Rotate` and CropBox.
- show A pages are stored landscape with `/Rotate 270`. The coordinates refer to the rotated (portrait) page.
- Boxes are axis-aligned.

## From entries to measures

The loader (`src/mdet/data.py`) applies these rules in order.

| Rule | Reason |
|---|---|
| Drop entries without `layout` | They are later playback passes of a repeated passage or repeat (`24-2`) or one trailing end marker per document |
| Strip the pass suffix: label = `value` without a trailing `-k` | The pass counts playback repetitions and is never printed |
| Boxes on one page with IoU >= 0.8 are duplicates; the later array entry wins | A repeated passage annotated under two names (`9A..9D` and `10..13`). In all 3 cases the later entry matches the printed label |
| Rows: measures on one page whose vertical extents overlap by more than half the smaller height | Heights inside a row differ slightly in 9 rows |
| Reading order: page, then row top, then x | The JSON array is a playback order (jumps back for repeats and codas) and is not used for order |

## Measure boxes

| Question (guideline 5.3) | Answer |
|---|---|
| Vertical extent | Inconsistent. show A: top staff line to bottom staff line of the system. show B: top staff line to either the bottom staff line or below the lyrics. Predictions span top staff line to bottom staff line; evaluation matches systems by vertical intersection over the smaller height |
| Start of the first measure | show A: the system's opening line, or the start-repeat barline if the system begins with one. show B: after clef and key signature, or at the start-repeat barline |
| Right edge | The barline, including final and end-repeat barlines |
| Pickup | Labeled `0` (6 documents start with `0`) |
| Multi-measure rests | One printed rest with a range label (`12-15`, `30 A-30 D`) is annotated as N equal-width boxes, one per measure. The inner boundaries have no barline (virtual boundaries) |
| Repeats and endings | Each printed measure has one box. Repeat passes have no box |
| Pages without music | Labeled with no entries. All 7 such pages are "turn the page" pages: true negatives |
| Every measure labeled | Yes, except show B songs 11 and 14, which have no entries at all and are excluded |
| Tiling | Consecutive boxes in a row share the boundary within 1e-4 in all but 8 cases (max overlap 0.004 of page width) |

## Labels

- Grammar: optional integer followed by an optional letter suffix: `12`, `23A`, `8a`, `A`.
- Charset of labels: `0-9`, `A-N`, `a`.
- Printed range labels (multi-measure rests) add `-` and a space: `12-15`, `30 A-30 D`.
- Numbering: every measure is numbered in both shows. show A prints numbers below the system at the left barline. show B prints numbers above the staff at the left barline and additionally boxed rehearsal numbers at the system start.
- Transitions in reading order: 4,597 increments, 60 advance suffix, 25 leave suffix, 23 insert suffix, 4 letter sequence, 5 letter to number, about 50 free jumps. Free jumps are printed cuts (for example `10` -> `15` in show A 12a); they are real, not annotation errors.

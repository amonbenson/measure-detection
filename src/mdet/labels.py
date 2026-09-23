"""Label grammar and transition types (guideline section 4.9)."""

from __future__ import annotations

import re

from .data import parse_label

TRANSITIONS = (
    "increment",  # 23 -> 24
    "insert_suffix",  # 23 -> 23A
    "advance_suffix",  # 23A -> 23B
    "leave_suffix",  # 23B -> 24
    "letter_sequence",  # A -> B
    "letter_to_number",  # B -> 1
    "reset",  # 57 -> 1
    "free_jump",  # anything else
)

RANGE_RE = re.compile(r"^(?P<a>\d*[A-Za-z]*)-(?P<b>\d*[A-Za-z]*)$")


def next_letter(s: str) -> str | None:
    """Successor of a suffix within its case: A -> B, a -> b, Z -> None."""
    if len(s) != 1 or s in "Zz":
        return None
    return chr(ord(s) + 1)


def first_letter_like(s: str) -> str:
    return "a" if s.islower() else "A"


def transition_type(prev: str, cur: str) -> str:
    p, c = parse_label(prev), parse_label(cur)
    if p is None or c is None:
        return "free_jump"
    (pn, ps), (cn, cs) = p, c
    if pn is not None and cn is not None:
        if cn == pn + 1 and cs == "":
            return "leave_suffix" if ps else "increment"
        if cn == pn and ps == "" and cs in ("A", "a"):
            return "insert_suffix"
        if cn == pn and ps and cs == next_letter(ps):
            return "advance_suffix"
        if cn in (0, 1) and cs == "":
            return "reset"
        return "free_jump"
    if pn is None and cn is None:
        return "letter_sequence" if cs == next_letter(ps) else "free_jump"
    if pn is None and cn is not None:
        return "letter_to_number" if cs == "" else "free_jump"
    return "free_jump"


def successors(label: str) -> list[tuple[str, str]]:
    """Grammar successors of a label as (label, transition type). Reset and free jump are not enumerated."""
    p = parse_label(label)
    if p is None:
        return []
    n, s = p
    out: list[tuple[str, str]] = []
    if n is not None:
        out.append((f"{n + 1}", "leave_suffix" if s else "increment"))
        if s == "":
            out.append((f"{n}A", "insert_suffix"))
            out.append((f"{n}a", "insert_suffix"))
        elif next_letter(s):
            out.append((f"{n}{next_letter(s)}", "advance_suffix"))
    else:
        if next_letter(s):
            out.append((next_letter(s), "letter_sequence"))
        out.append(("1", "letter_to_number"))
    return out


def expand_range(text: str) -> list[str] | None:
    """Expand a range label into its member labels. `12-15` -> [12..15]; `30A-30D` -> [30A..30D]."""
    m = RANGE_RE.match(text.replace(" ", ""))
    if m is None:
        return None
    a, b = parse_label(m["a"]), parse_label(m["b"])
    if a is None or b is None:
        return None
    (an, as_), (bn, bs) = a, b
    if an is not None and bn is not None and as_ == "" and bs == "" and 1 <= bn - an <= 200:
        return [str(k) for k in range(an, bn + 1)]
    if an is not None and an == bn and as_ and bs and len(as_) == len(bs) == 1 and as_ < bs:
        return [f"{an}{chr(k)}" for k in range(ord(as_), ord(bs) + 1)]
    return None

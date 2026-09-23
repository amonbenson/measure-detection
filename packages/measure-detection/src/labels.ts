// Label grammar and transition types (mdet/labels.py, mdet/data.py).

export const TRANSITIONS = ["increment", "insert_suffix", "advance_suffix", "leave_suffix", "letter_sequence", "letter_to_number", "reset", "free_jump"] as const;
export type Transition = (typeof TRANSITIONS)[number];

export function parseLabel(label: string): [number | null, string] | null {
  const m = /^(\d*)([A-Za-z]*)$/.exec(label);
  if (!m || (!m[1] && !m[2])) return null;
  return [m[1] ? parseInt(m[1], 10) : null, m[2]];
}

function nextLetter(s: string): string | null {
  if (s.length !== 1 || s === "Z" || s === "z") return null;
  return String.fromCharCode(s.charCodeAt(0) + 1);
}

export function transitionType(prev: string, cur: string): Transition {
  const p = parseLabel(prev), c = parseLabel(cur);
  if (!p || !c) return "free_jump";
  const [pn, ps] = p, [cn, cs] = c;
  if (pn !== null && cn !== null) {
    if (cn === pn + 1 && cs === "") return ps ? "leave_suffix" : "increment";
    if (cn === pn && ps === "" && (cs === "A" || cs === "a")) return "insert_suffix";
    if (cn === pn && ps && cs === nextLetter(ps)) return "advance_suffix";
    if ((cn === 0 || cn === 1) && cs === "") return "reset";
    return "free_jump";
  }
  if (pn === null && cn === null) return cs === nextLetter(ps) ? "letter_sequence" : "free_jump";
  if (pn === null && cn !== null) return cs === "" ? "letter_to_number" : "free_jump";
  return "free_jump";
}

export function successors(label: string): string[] {
  const p = parseLabel(label);
  if (!p) return [];
  const [n, s] = p;
  const out: string[] = [];
  if (n !== null) {
    out.push(`${n + 1}`);
    if (s === "") out.push(`${n}A`, `${n}a`);
    else if (nextLetter(s)) out.push(`${n}${nextLetter(s)}`);
  } else {
    if (nextLetter(s)) out.push(nextLetter(s)!);
    out.push("1");
  }
  return out;
}

export function expandRange(text: string): string[] | null {
  const m = /^(\d*[A-Za-z]*)-(\d*[A-Za-z]*)$/.exec(text.replace(/ /g, ""));
  if (!m) return null;
  const a = parseLabel(m[1]), b = parseLabel(m[2]);
  if (!a || !b) return null;
  const [an, as] = a, [bn, bs] = b;
  if (an !== null && bn !== null && as === "" && bs === "" && bn - an >= 1 && bn - an <= 200) {
    const out: string[] = [];
    for (let k = an; k <= bn; k++) out.push(`${k}`);
    return out;
  }
  if (an !== null && an === bn && as && bs && as.length === 1 && bs.length === 1 && as < bs) {
    const out: string[] = [];
    for (let k = as.charCodeAt(0); k <= bs.charCodeAt(0); k++) out.push(`${an}${String.fromCharCode(k)}`);
    return out;
  }
  return null;
}

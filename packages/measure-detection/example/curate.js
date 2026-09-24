// Curation page: split a large PDF (a book) into songs, detect measures per song, correct boxes and values,
// export each song as a training document (dataset/documents/<id>/, same format as the original data).
// Exported and original documents can be opened again and edited in place.
// Boxes are axis-aligned fractions of the displayed page, like layout in measures.json.
// Only the song number and the show id are stored with a document; titles and show names never are.

import * as ort from "onnxruntime-web";
import * as pdfjs from "pdfjs-dist";
import { MeasureDetector } from "../dist/index.js";

ort.env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";
pdfjs.GlobalWorkerOptions.workerSrc = "/node_modules/pdfjs-dist/build/pdf.worker.min.mjs";
// pdf.js decodes JBIG2 and JPEG 2000 scans with WebAssembly; without it such pages render blank.
const WASM_URL = "/node_modules/pdfjs-dist/wasm/";

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const LOW_CONFIDENCE = 0.5;
const LABEL_RE = /^\d*[A-Za-z]*$/;
const NUMBER_RE = /^[0-9A-Za-z.-]{0,16}$/;
const MIN_SIZE = 0.004; // smallest box side, page fraction
const LINK_TOL = 0.004; // edges closer than this are one shared boundary
const SNAP_TOL = 0.012; // a drawn box snaps to a neighbor edge within this distance
const DRAG_PX = 3; // pointer travel before a press becomes a drag
// Training data uses one start convention: the first measure of a system starts after clef and key.
const START_CONVENTION = "after-header";

let pdf = null;
// kind "book": a PDF opened from disk, split into songs, saved as a session on the server.
// kind "document": one document from dataset/documents, one song covering all its pages, saved into the document.
let state = null;
let docs = []; // document list from the server
let view = { kind: "none" };
let selected = null; // measure id
let nextId = 1;
const pending = new Map(); // page -> unlabeled measures shown while the song is detected
const running = new Map(); // song -> progress text
let undoStack = [], redoStack = [];

const status = (text) => ($("status").textContent = text);
const songEnd = (song) => {
  const i = state.songs.indexOf(song);
  return i + 1 < state.songs.length ? state.songs[i + 1].start - 1 : state.numPages - 1;
};
const current = () => (view.kind === "song" ? view.song : null);
const findMeasure = (id) => current()?.measures?.find((m) => m.id === id) ?? null;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const songName = (s) => (s.number ? `No. ${s.number}` : "(no number)");
const isBook = () => state?.kind === "book";

function randomId() {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  return Array.from(crypto.getRandomValues(new Uint8Array(15)), (b) => chars[b % chars.length]).join("");
}

function newSong(start) {
  return { start, number: "", skip: false, split: "train", measures: null, docId: null, exported: false, modified: null };
}

// measures.json entries -> editor measures. Original documents list later playback passes without a box and
// values with a pass suffix (docs/CONVENTIONS.md); the editor keeps the boxes and the printed label.
function fromEntries(entries, start) {
  return entries
    .filter((e) => e.layout)
    .map((e) => ({ id: nextId++, page: start + e.layout.page, x: e.layout.x, y: e.layout.y, w: e.layout.width, h: e.layout.height, value: e.value.trim().replace(/-\d+$/, ""), conf: null, edited: true }));
}

async function loadDocs() {
  const r = await fetch("/api/documents");
  docs = r.ok ? await r.json() : [];
}

// ---------------------------------------------------------------- persistence

let saveTimer = null, saving = false, dirty = false;

function touch(song) {
  if (song) song.exported = false;
  dirty = true;
  if (!isBook()) {
    // A document is written only by Save: half-edited measure lists never reach the training data.
    $("saved").textContent = "Unsaved changes (Save writes the document)";
    renderSongs();
    return;
  }
  $("saved").textContent = "Unsaved changes";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 800);
}

async function save() {
  if (!isBook()) return;
  if (saving) return void (saveTimer = setTimeout(save, 300));
  saving = true;
  dirty = false;
  $("saved").textContent = "Saving...";
  try {
    const r = await fetch(`/api/session/${state.fingerprint}`, { method: "PUT", body: JSON.stringify(state) });
    if (!r.ok) throw new Error((await r.json()).error);
    if (!dirty) $("saved").textContent = `Saved ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    dirty = true;
    $("saved").textContent = `Save failed: ${err.message}`;
  } finally {
    saving = false;
  }
}

window.addEventListener("beforeunload", (ev) => {
  if (dirty || saving || running.size) ev.preventDefault();
});

function canLeave() {
  return !(dirty || running.size) || confirm("Changes are not saved yet or a detection is running. Continue anyway?");
}

function resetEditor() {
  nextId = 1;
  undoStack = [];
  redoStack = [];
  pending.clear();
  dirty = false;
}

function afterOpen() {
  for (const s of state.songs) for (const m of s.measures ?? []) nextId = Math.max(nextId, (m.id ?? 0) + 1);
  for (const s of state.songs) for (const m of s.measures ?? []) m.id ??= nextId++;
  $("show").disabled = false;
  $("overview").disabled = $("detectAll").disabled = !isBook();
  renderShows();
  renderSongs();
}

// ---------------------------------------------------------------- opening a book

$("file").addEventListener("change", async () => {
  const file = $("file").files[0];
  $("file").value = "";
  if (!file || !canLeave()) return;
  status("Loading PDF...");
  try {
    pdf = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()), wasmUrl: WASM_URL }).promise;
    const fingerprint = pdf.fingerprints[0];
    const r = await fetch(`/api/session/${fingerprint}`);
    let restored = null;
    if (r.ok) {
      restored = await r.json();
      if (restored.numPages !== pdf.numPages && !confirm(`A saved session with this fingerprint has ${restored.numPages} pages, this PDF has ${pdf.numPages}. Discard the saved session?`)) return;
      if (restored.numPages !== pdf.numPages) restored = null;
    }
    resetEditor();
    state = restored ?? { version: 2, kind: "book", fingerprint, numPages: pdf.numPages, showId: randomId(), songs: [newSong(0)] };
    // Sessions written before names were dropped.
    state.kind = "book";
    state.version = 2;
    delete state.fileName;
    delete state.show;
    delete state.startConvention;
    state.showId ??= randomId();
    for (const s of state.songs) delete s.title;
    await loadDocs();
    const note = await syncFromDocuments();
    afterOpen();
    status(`Book: ${pdf.numPages} pages, ${state.songs.length} songs${restored ? " (session restored)" : ""}.${note}`);
    $("saved").textContent = restored ? "Session restored" : "";
    await save();
    showOverview();
  } catch (err) {
    status(`Error: ${err.message ?? err}`);
    console.error(err);
  }
});

// A song exported from this book may have been edited later as a document. The document is the newer
// version, so its measures replace the session's copy, unless the song was also changed here since the export.
async function syncFromDocuments() {
  const byId = new Map(docs.map((d) => [d.id, d]));
  let synced = 0, conflicts = 0, missing = 0;
  for (const song of state.songs) {
    if (!song.docId) continue;
    const d = byId.get(song.docId);
    if (!d) {
      song.docId = null;
      song.exported = false;
      missing++;
      continue;
    }
    if (!d.modified || d.modified === song.modified) continue;
    const doc = await (await fetch(`/api/documents/${song.docId}`)).json();
    const n = songEnd(song) - song.start + 1;
    if (!song.exported || doc.measures.some((e) => e.layout && e.layout.page >= n)) {
      conflicts++;
      continue;
    }
    song.measures = fromEntries(doc.measures, song.start);
    song.number = doc.source.number ?? "";
    song.split = doc.split ?? "none";
    song.modified = d.modified;
    song.exported = true;
    synced++;
  }
  const parts = [];
  if (synced) parts.push(`${synced} songs updated from their documents`);
  if (conflicts) parts.push(`${conflicts} songs changed both here and as a document (this session's version is kept; export overwrites the document)`);
  if (missing) parts.push(`${missing} exported documents no longer exist`);
  return parts.length ? " " + parts.join("; ") + "." : "";
}

// ---------------------------------------------------------------- documents

async function showDocuments() {
  if (!canLeave()) return;
  generation++;
  view = { kind: "documents" };
  selected = null;
  await loadDocs();
  const main = $("main");
  main.replaceChildren();
  // main is the scroll container and outlives the view; a new view starts at the top.
  main.scrollTop = 0;
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `<div class="row"><span>Documents in <code>dataset/documents</code>. Click one to open and edit it; Save writes it back.</span>
    <label>Show <select id="docShow"><option value="">all</option></select></label></div>`;
  main.append(bar);
  for (const [id, n] of showCounts()) $("docShow").add(new Option(`${id} (${n})`, id));
  const table = document.createElement("table");
  table.className = "docs";
  main.append(table);
  const draw = () => {
    const show = $("docShow").value;
    const rows = docs.filter((d) => !show || d.show_id === show);
    rows.sort((a, b) => (a.show_id ?? "").localeCompare(b.show_id ?? "") || a.number.localeCompare(b.number, undefined, { numeric: true }) || a.id.localeCompare(b.id));
    table.innerHTML = `<thead><tr><th>Document</th><th>Show</th><th>No.</th><th>Split</th><th>Kind</th><th>Modified</th></tr></thead><tbody></tbody>`;
    for (const d of rows) {
      const tr = document.createElement("tr");
      for (const v of [d.id, d.show_id ?? "", d.number, d.split ?? "none", d.curated ? "curated" : "original", d.modified ? new Date(d.modified).toLocaleString() : ""]) tr.insertCell().textContent = v;
      tr.addEventListener("click", () => openDocument(d.id));
      table.tBodies[0].append(tr);
    }
  };
  $("docShow").addEventListener("change", draw);
  draw();
  renderSongs();
}

async function openDocument(id) {
  status(`Loading document ${id}...`);
  try {
    const doc = await (await fetch(`/api/documents/${id}`)).json();
    pdf = await pdfjs.getDocument({ url: `/dataset/documents/${id}/score.pdf`, wasmUrl: WASM_URL }).promise;
    resetEditor();
    const song = { ...newSong(0), number: doc.source.number ?? "", split: doc.split ?? "none", measures: fromEntries(doc.measures, 0), docId: id, exported: true, modified: doc.source.modified ?? null };
    state = { version: 2, kind: "document", docId: id, original: !doc.source.curated, numPages: pdf.numPages, showId: doc.source.show_id, songs: [song] };
    afterOpen();
    $("saved").textContent = "";
    status(`Document ${id}: ${pdf.numPages} pages, ${song.measures.length} measures${state.original ? " (original document)" : ""}.`);
    showSong(song);
  } catch (err) {
    status(`Error: ${err.message ?? err}`);
    console.error(err);
  }
}

$("documents").addEventListener("click", showDocuments);

// ---------------------------------------------------------------- show

function showCounts() {
  const counts = new Map();
  for (const d of docs) if (d.show_id) counts.set(d.show_id, (counts.get(d.show_id) ?? 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1]);
}

// Shows are identified by an id only. A book starts as a new show; pick an existing one if the book belongs to it.
function renderShows() {
  const sel = $("show");
  sel.replaceChildren();
  const counts = showCounts();
  if (!counts.some(([id]) => id === state.showId)) sel.add(new Option(`${state.showId} (new)`, state.showId));
  for (const [id, n] of counts) sel.add(new Option(`${id} (${n} documents)`, id));
  sel.add(new Option("New show", "__new"));
  sel.value = state.showId;
}

$("show").addEventListener("change", () => {
  state.showId = $("show").value === "__new" ? randomId() : $("show").value;
  state.songs.forEach((s) => (s.exported = false));
  touch();
  renderShows();
  renderSongs();
});

// ---------------------------------------------------------------- song list

function songBadges(song) {
  const b = [];
  if (running.has(song)) b.push(["run", running.get(song)]);
  if (song.skip) return [...b, ["", "skipped"]];
  if (song.measures === null) b.push(["", "not detected"]);
  else {
    const review = song.measures.filter((m) => !m.edited && m.conf !== null && m.conf < LOW_CONFIDENCE).length;
    b.push(["", `${song.measures.length} measures`]);
    if (review) b.push(["warn", `${review} to review`]);
  }
  if (!isBook()) b.push(song.exported ? ["ok", "saved"] : ["warn", "unsaved"]);
  else if (song.exported) b.push(["ok", "exported"]);
  else if (song.docId) b.push(["warn", "changed since export"]);
  return b;
}

function renderSongs() {
  const list = $("songs");
  list.replaceChildren();
  if (!state) return;
  state.songs.forEach((song, i) => {
    const el = document.createElement("div");
    el.className = "song" + (current() === song ? " active" : "") + (song.skip ? " skip" : "");
    const end = songEnd(song);
    el.innerHTML = `<div><b></b></div><div class="sub"></div>`;
    el.querySelector("b").textContent = isBook() ? `${i + 1}. ${songName(song)}` : `${state.docId}, ${songName(song)}`;
    const sub = el.querySelector(".sub");
    sub.textContent = `pages ${song.start + 1}-${end + 1}`;
    for (const [cls, text] of songBadges(song)) {
      const span = document.createElement("span");
      span.className = "badge " + cls;
      span.textContent = text;
      sub.append(span);
    }
    el.addEventListener("click", () => showSong(song));
    list.append(el);
  });
}

// Split the song containing `page` so that a new song starts there, or merge the song starting there into the previous one.
function toggleStart(page) {
  if (page === 0 || !isBook()) return;
  const i = state.songs.findIndex((s) => s.start === page);
  if (i > 0) {
    const [gone] = state.songs.splice(i, 1);
    const prev = state.songs[i - 1];
    if (gone.docId && !confirm(`Song "${songName(gone)}" was exported as ${gone.docId}. Its files stay in dataset/documents; delete them by hand if needed. Merge anyway?`)) {
      state.songs.splice(i, 0, gone);
      return;
    }
    if (gone.measures) prev.measures = [...(prev.measures ?? []), ...gone.measures];
    touch(prev);
  } else {
    const j = state.songs.findLastIndex((s) => s.start < page);
    const parent = state.songs[j];
    const song = newSong(page);
    song.split = parent.split;
    if (parent.measures) {
      song.measures = parent.measures.filter((m) => m.page >= page);
      parent.measures = parent.measures.filter((m) => m.page < page);
    }
    state.songs.splice(j + 1, 0, song);
    touch(parent);
    touch(song);
  }
  // Undo entries refer to song objects whose page ranges just changed.
  undoStack = [];
  redoStack = [];
  renderSongs();
}

// ---------------------------------------------------------------- rendering pages

// Renders run one at a time: large scans take long to decode and parallel renders only add memory.
let renderChain = Promise.resolve();
let generation = 0;

function renderInto(holder, page, cssWidth, gen) {
  renderChain = renderChain.then(async () => {
    if (gen !== generation || !holder.isConnected) return;
    const p = await pdf.getPage(page + 1);
    const vp1 = p.getViewport({ scale: 1 });
    const scale = Math.min(3, (cssWidth * devicePixelRatio) / vp1.width);
    const vp = p.getViewport({ scale });
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(vp.width);
    canvas.height = Math.round(vp.height);
    await p.render({ canvasContext: canvas.getContext("2d"), canvas, viewport: vp }).promise;
    p.cleanup();
    if (gen !== generation || !holder.isConnected) return;
    holder.querySelector(".ph, canvas")?.replaceWith(canvas);
    holder.dispatchEvent(new CustomEvent("rendered", { detail: vp1.height / vp1.width }));
  }).catch((err) => console.error(err));
}

function lazyRender(container, items, width) {
  const gen = generation;
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      renderInto(e.target, Number(e.target.dataset.page), width(), gen);
    }
  }, { root: $("main"), rootMargin: "600px 0px" });
  items.forEach((el) => io.observe(el));
}

// ---------------------------------------------------------------- page overview

function showOverview() {
  generation++;
  view = { kind: "overview" };
  selected = null;
  const main = $("main");
  main.replaceChildren();
  main.scrollTop = 0;
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `<div class="row"><span>Click a page to start a new song there (click a song's first page to merge it into the previous song). Double-click to open the song.</span>
    <label>Size <input type="range" min="90" max="360" value="150" id="thumbSize"></label></div>`;
  main.append(bar);
  const grid = document.createElement("div");
  grid.className = "grid";
  main.append(grid);
  const thumbs = [];
  for (let p = 0; p < state.numPages; p++) {
    const t = document.createElement("div");
    t.dataset.page = p;
    t.innerHTML = `<div class="ph"></div><span class="num">${p + 1}</span>`;
    // A double click also fires two clicks; wait so that opening a song does not split and merge it.
    let timer = null;
    t.addEventListener("click", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        toggleStart(p);
        decorateThumbs(thumbs);
      }, 250);
    });
    t.addEventListener("dblclick", () => {
      clearTimeout(timer);
      showSong(state.songs.findLast((s) => s.start <= p), p);
    });
    grid.append(t);
    thumbs.push(t);
  }
  decorateThumbs(thumbs);
  const size = $("thumbSize");
  size.addEventListener("input", () => grid.style.setProperty("--thumb", `${size.value}px`));
  lazyRender(grid, thumbs, () => Math.max(Number(size.value), 150));
  renderSongs();
}

function decorateThumbs(thumbs) {
  state.songs.forEach((song, i) => {
    for (let p = song.start; p <= songEnd(song); p++) {
      const t = thumbs[p];
      t.className = "thumb" + (i % 2 ? " alt" : "") + (p === song.start ? " start" : "") + (song.skip ? " skip" : "");
      t.querySelector(".title")?.remove();
      if (p === song.start) {
        const title = document.createElement("span");
        title.className = "title";
        title.textContent = `${i + 1}. ${songName(song)}`;
        t.append(title);
      }
    }
  });
}

$("overview").addEventListener("click", () => isBook() && showOverview());

// ---------------------------------------------------------------- song editor

function showSong(song, scrollToPage = null) {
  generation++;
  view = { kind: "song", song, pages: new Map() };
  selected = null;
  const main = $("main");
  main.replaceChildren();
  main.scrollTop = 0;
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `
    <div class="row">
      <label>Song number <input type="text" id="sNumber" size="5"></label>
      <label>Split <select id="sSplit"><option value="train">train</option><option value="val">val</option><option value="none">none</option></select></label>
      <label id="sSkipLabel"><input type="checkbox" id="sSkip"> Skip (not a song)</label>
      <button id="sDetect">Detect</button>
      <button id="sExport"></button>
      <span id="sInfo" class="help"></span>
    </div>
    <div class="row sel" id="selRow"></div>
    <details class="help"><summary>Editing help</summary>Drag inside a box to move it, drag an edge to resize (shared boundaries and row edges move together; hold <kbd>Alt</kbd> to move one box only).
      Drag on empty paper (or <kbd>Shift</kbd>+drag anywhere) to add a box. <kbd>Ctrl</kbd>+click splits a box.
      <kbd>Tab</kbd>/<kbd>Shift+Tab</kbd> next/previous, <kbd>Enter</kbd> edit value (Enter again confirms and moves on), <kbd>Del</kbd> delete, <kbd>M</kbd> merge with right neighbor, <kbd>N</kbd> renumber following, <kbd>Ctrl+Z</kbd>/<kbd>Ctrl+Y</kbd> undo/redo, <kbd>Esc</kbd> deselect.</details>`;
  main.append(bar);
  $("sNumber").value = song.number;
  // Test documents stay in the frozen test split.
  if (song.split === "test") {
    $("sSplit").replaceChildren(new Option("test (frozen)", "test"));
    $("sSplit").disabled = true;
  }
  $("sSplit").value = song.split ?? "none";
  $("sSkip").checked = song.skip;
  $("sSkipLabel").hidden = !isBook();
  $("sExport").textContent = isBook() ? "Export" : "Save";
  const meta = () => {
    song.number = $("sNumber").value.trim();
    song.split = $("sSplit").value;
    song.skip = $("sSkip").checked;
    touch(song);
    renderSongs();
  };
  for (const id of ["sNumber", "sSplit", "sSkip"]) $(id).addEventListener("change", meta);
  $("sDetect").addEventListener("click", () => enqueueDetection(song, true));
  $("sExport").addEventListener("click", () => exportSong(song));

  const holders = [];
  const end = songEnd(song);
  for (let p = song.start; p <= end; p++) {
    const box = document.createElement("div");
    box.className = "pagebox";
    const head = document.createElement("div");
    head.className = "pagehead";
    head.innerHTML = `<span>Page ${p + 1}</span>`;
    const btn = document.createElement("button");
    if (p === song.start) {
      btn.textContent = "Merge into previous song";
      btn.disabled = p === 0;
    } else btn.textContent = "Start new song here";
    btn.addEventListener("click", () => {
      toggleStart(p);
      showSong(state.songs.findLast((s) => s.start <= p), p);
    });
    if (isBook()) head.append(btn);
    const holder = document.createElement("div");
    holder.className = "page";
    holder.dataset.page = p;
    holder.innerHTML = `<div class="ph"></div>`;
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", `0 0 1000 1414`);
    svg.setAttribute("preserveAspectRatio", "none");
    holder.append(svg);
    const page = { svg, holder, aspect: 1.414 };
    view.pages.set(p, page);
    holder.addEventListener("rendered", (ev) => {
      page.aspect = ev.detail;
      svg.setAttribute("viewBox", `0 0 1000 ${1000 * page.aspect}`);
      drawPage(p);
    });
    attachEditing(svg, p);
    box.append(head, holder);
    main.append(box);
    holders.push(holder);
    drawPage(p);
  }
  lazyRender(main, holders, () => Math.min(1000, main.clientWidth - 32));
  updateInfo();
  updateSelection();
  renderSongs();
  if (scrollToPage !== null) requestAnimationFrame(() => view.pages.get(scrollToPage)?.holder.parentElement.scrollIntoView());
}

function updateInfo() {
  const song = current();
  if (!song || !$("sInfo")) return;
  const ms = song.measures;
  $("sInfo").textContent = ms === null ? "Not detected yet." : `${ms.length} measures, ${ms.filter((m) => m.edited).length} edited or confirmed.` + (song.docId && isBook() ? ` Document ${song.docId}${song.exported ? "" : " (changed since export)"}.` : "");
}

function drawPage(p) {
  const page = view.pages?.get(p);
  if (!page) return;
  const song = current();
  const { svg, aspect } = page;
  const h = 1000 * aspect;
  svg.replaceChildren();
  const pend = pending.get(p);
  const list = pend ? pend.map((m) => ({ ...m, pending: true })) : (song.measures ?? []).filter((m) => m.page === p);
  for (const m of list) {
    const cls = m.pending ? "pending" : m.id === selected ? "selected" : m.edited ? "edited" : m.conf !== null && m.conf < LOW_CONFIDENCE ? "low" : "";
    const r = document.createElementNS(SVG, "rect");
    r.setAttribute("x", m.x * 1000);
    r.setAttribute("y", m.y * h);
    r.setAttribute("width", m.w * 1000);
    r.setAttribute("height", m.h * h);
    r.setAttribute("class", `m ${cls}`);
    svg.append(r);
    if (m.pending) continue;
    const t = document.createElementNS(SVG, "text");
    t.setAttribute("x", m.x * 1000 + 3);
    t.setAttribute("y", m.y * h - 3);
    t.setAttribute("class", cls);
    t.textContent = m.value || "?";
    svg.append(t);
  }
  if (drag?.page === p && drag.draft) {
    const d = drag.draft;
    const r = document.createElementNS(SVG, "rect");
    r.setAttribute("x", d.x * 1000);
    r.setAttribute("y", d.y * h);
    r.setAttribute("width", d.w * 1000);
    r.setAttribute("height", d.h * h);
    r.setAttribute("class", "draft");
    svg.append(r);
  }
}

function redrawSong() {
  for (const p of view.pages?.keys() ?? []) drawPage(p);
  updateInfo();
}

// ---------------------------------------------------------------- geometry helpers

function vOverlap(a, b) {
  const inter = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return inter / Math.max(1e-9, Math.min(a.h, b.h));
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const ih = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = iw * ih;
  return inter / Math.max(1e-12, a.w * a.h + b.w * b.h - inter);
}

// Reading order as in the loader (src/mdet/data.py): page, row (vertical overlap > half the smaller height), x.
function readingOrder(measures) {
  const rows = [];
  for (const m of [...measures].sort((a, b) => a.page - b.page || a.y + a.h / 2 - (b.y + b.h / 2))) {
    const row = rows.find((r) => r.page === m.page && r.ms.some((o) => vOverlap(o, m) > 0.5));
    if (row) row.ms.push(m);
    else rows.push({ page: m.page, ms: [m] });
  }
  const top = (r) => Math.min(...r.ms.map((m) => m.y));
  rows.sort((a, b) => a.page - b.page || top(a) - top(b));
  return rows.flatMap((r) => r.ms.sort((a, b) => a.x - b.x));
}

const rowOf = (song, m) => song.measures.filter((o) => o.page === m.page && vOverlap(o, m) > 0.5);

function nextValue(value) {
  const n = /^\d+$/.test(value ?? "") ? Number(value) : null;
  return n === null ? "" : String(n + 1);
}

// ---------------------------------------------------------------- undo

function pushUndo(song) {
  undoStack.push({ song, measures: structuredClone(song.measures) });
  if (undoStack.length > 200) undoStack.shift();
  redoStack = [];
}

function undo(from, to) {
  const e = from.pop();
  if (!e || !state.songs.includes(e.song)) return;
  to.push({ song: e.song, measures: structuredClone(e.song.measures) });
  e.song.measures = e.measures;
  touch(e.song);
  if (current() !== e.song) showSong(e.song);
  else {
    if (!findMeasure(selected)) selected = null;
    redrawSong();
    updateSelection();
  }
  renderSongs();
}

// ---------------------------------------------------------------- editing

let drag = null;

function pointerFrac(ev, rect) {
  return { x: clamp((ev.clientX - rect.left) / rect.width, 0, 1), y: clamp((ev.clientY - rect.top) / rect.height, 0, 1) };
}

// Box and mode under the pointer; edges of the selected box win, then the smallest box.
function hitTest(song, p, pt, tol) {
  const cands = (song.measures ?? []).filter((m) => m.page === p && pt.x >= m.x - tol.x && pt.x <= m.x + m.w + tol.x && pt.y >= m.y - tol.y && pt.y <= m.y + m.h + tol.y);
  cands.sort((a, b) => (b.id === selected) - (a.id === selected) || a.w * a.h - b.w * b.h);
  for (const m of cands) {
    if (Math.abs(pt.x - m.x) <= tol.x) return { m, mode: "left" };
    if (Math.abs(pt.x - (m.x + m.w)) <= tol.x) return { m, mode: "right" };
    if (Math.abs(pt.y - m.y) <= tol.y) return { m, mode: "top" };
    if (Math.abs(pt.y - (m.y + m.h)) <= tol.y) return { m, mode: "bottom" };
  }
  const inside = cands.find((m) => pt.x > m.x && pt.x < m.x + m.w && pt.y > m.y && pt.y < m.y + m.h);
  return inside ? { m: inside, mode: "move" } : null;
}

const CURSORS = { left: "ew-resize", right: "ew-resize", top: "ns-resize", bottom: "ns-resize", move: "move" };

// Boxes whose edge coincides with the dragged edge: the neighbor across a shared boundary, or the row for top and bottom.
function linkedBoxes(song, m, mode) {
  const others = song.measures.filter((o) => o !== m && o.page === m.page && vOverlap(o, m) > 0.5);
  if (mode === "left") return others.filter((o) => Math.abs(o.x + o.w - m.x) < LINK_TOL);
  if (mode === "right") return others.filter((o) => Math.abs(o.x - (m.x + m.w)) < LINK_TOL);
  if (mode === "top") return others.filter((o) => Math.abs(o.y - m.y) < LINK_TOL);
  if (mode === "bottom") return others.filter((o) => Math.abs(o.y + o.h - (m.y + m.h)) < LINK_TOL);
  return [];
}

function attachEditing(svg, p) {
  svg.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0 || pending.has(p)) return;
    const song = current();
    const rect = svg.getBoundingClientRect();
    const pt = pointerFrac(ev, rect);
    const tol = { x: 6 / rect.width, y: 6 / rect.height };
    const hit = ev.shiftKey ? null : hitTest(song, p, pt, tol);
    if (hit && (ev.ctrlKey || ev.metaKey)) {
      splitAt(song, hit.m, pt.x);
      return;
    }
    drag = { page: p, rect, start: pt, mode: hit?.mode ?? "draw", m: hit?.m ?? null, alt: ev.altKey, moved: false, draft: null };
    selected = hit?.m.id ?? null;
    svg.setPointerCapture(ev.pointerId);
    redrawSong();
    updateSelection();
    ev.preventDefault();
  });

  svg.addEventListener("pointermove", (ev) => {
    const song = current();
    if (!drag) {
      const rect = svg.getBoundingClientRect();
      const hit = ev.shiftKey ? null : hitTest(song, p, pointerFrac(ev, rect), { x: 6 / rect.width, y: 6 / rect.height });
      svg.style.cursor = hit ? CURSORS[hit.mode] : "crosshair";
      return;
    }
    const pt = pointerFrac(ev, drag.rect);
    const dx = pt.x - drag.start.x, dy = pt.y - drag.start.y;
    if (!drag.moved) {
      if (Math.hypot(dx * drag.rect.width, dy * drag.rect.height) < DRAG_PX) return;
      drag.moved = true;
      if (drag.m) {
        pushUndo(song);
        drag.links = drag.alt ? [] : linkedBoxes(song, drag.m, drag.mode);
        drag.orig = new Map([drag.m, ...drag.links].map((o) => [o, { x: o.x, y: o.y, w: o.w, h: o.h }]));
      }
    }
    if (drag.mode === "draw") {
      drag.draft = { x: Math.min(drag.start.x, pt.x), y: Math.min(drag.start.y, pt.y), w: Math.abs(dx), h: Math.abs(dy) };
    } else applyDrag(drag, dx, dy);
    drawPage(p);
  });

  const finish = () => {
    if (!drag) return;
    const d = drag;
    drag = null;
    const song = current();
    if (d.mode === "draw" && d.moved && d.draft && d.draft.w >= MIN_SIZE && d.draft.h >= MIN_SIZE) addBox(song, p, d.draft);
    else if (d.m && d.moved) {
      for (const o of [d.m, ...d.links]) o.edited = true;
      touch(song);
      renderSongs();
    }
    drawPage(p);
    updateInfo();
    updateSelection();
  };
  svg.addEventListener("pointerup", finish);
  svg.addEventListener("pointercancel", finish);
}

function applyDrag(d, dx, dy) {
  const { m, links, orig } = d;
  const o = orig.get(m);
  const right = (b) => orig.get(b).x + orig.get(b).w;
  const bottom = (b) => orig.get(b).y + orig.get(b).h;
  if (d.mode === "move") {
    m.x = clamp(o.x + dx, 0, 1 - o.w);
    m.y = clamp(o.y + dy, 0, 1 - o.h);
  } else if (d.mode === "left") {
    const lo = Math.max(0, ...links.map((b) => orig.get(b).x + MIN_SIZE));
    const x = clamp(o.x + dx, lo, right(m) - MIN_SIZE);
    m.x = x;
    m.w = right(m) - x;
    for (const b of links) b.w = x - b.x;
  } else if (d.mode === "right") {
    const hi = Math.min(1, ...links.map((b) => right(b) - MIN_SIZE));
    const x = clamp(right(m) + dx, o.x + MIN_SIZE, hi);
    m.w = x - o.x;
    for (const b of links) {
      b.x = x;
      b.w = right(b) - x;
    }
  } else if (d.mode === "top") {
    const all = [m, ...links];
    const y = clamp(o.y + dy, 0, Math.min(...all.map((b) => bottom(b) - MIN_SIZE)));
    for (const b of all) {
      b.y = y;
      b.h = bottom(b) - y;
    }
  } else if (d.mode === "bottom") {
    const all = [m, ...links];
    const y = clamp(o.y + o.h + dy, Math.max(...all.map((b) => orig.get(b).y + MIN_SIZE)), 1);
    for (const b of all) b.h = y - b.y;
  }
}

// New box: takes the vertical extent of the row it overlaps and snaps its sides to neighboring boundaries.
function addBox(song, p, box) {
  if (!song.measures) song.measures = [];
  pushUndo(song);
  const m = { id: nextId++, page: p, ...box, value: "", conf: null, edited: true };
  const row = rowOf(song, m);
  if (row.length) {
    const med = (v) => v.sort((a, b) => a - b)[v.length >> 1];
    m.y = med(row.map((o) => o.y));
    m.h = med(row.map((o) => o.h));
    for (const o of row) {
      const r = m.x + m.w;
      if (Math.abs(o.x + o.w - m.x) < SNAP_TOL) m.x = o.x + o.w;
      m.w = (Math.abs(o.x - r) < SNAP_TOL ? o.x : r) - m.x;
    }
  }
  song.measures.push(m);
  const order = readingOrder(song.measures);
  const prev = order[order.indexOf(m) - 1];
  m.value = prev ? nextValue(prev.value) : "";
  selected = m.id;
  touch(song);
  redrawSong();
  renderSongs();
  updateSelection();
  $("value")?.focus();
  $("value")?.select();
}

function splitAt(song, m, x) {
  if (x - m.x < MIN_SIZE || m.x + m.w - x < MIN_SIZE) return;
  pushUndo(song);
  const n = { ...m, id: nextId++, x, w: m.x + m.w - x, value: nextValue(m.value), conf: null, edited: true };
  m.w = x - m.x;
  m.edited = true;
  song.measures.push(n);
  selected = n.id;
  touch(song);
  drawPage(m.page);
  updateInfo();
  renderSongs();
  updateSelection();
}

function mergeRight(song, m) {
  const right = rowOf(song, m).filter((o) => o !== m && o.x > m.x).sort((a, b) => a.x - b.x)[0];
  if (!right) return;
  pushUndo(song);
  const x1 = Math.max(m.x + m.w, right.x + right.w), y0 = Math.min(m.y, right.y), y1 = Math.max(m.y + m.h, right.y + right.h);
  Object.assign(m, { w: x1 - m.x, y: y0, h: y1 - y0, edited: true });
  song.measures.splice(song.measures.indexOf(right), 1);
  touch(song);
  drawPage(m.page);
  updateInfo();
  renderSongs();
  updateSelection();
}

function deleteMeasure(song, m) {
  pushUndo(song);
  const order = readingOrder(song.measures);
  const next = order[order.indexOf(m) + 1] ?? order[order.indexOf(m) - 1];
  song.measures.splice(song.measures.indexOf(m), 1);
  selected = next?.id ?? null;
  touch(song);
  redrawSong();
  renderSongs();
  updateSelection();
}

function renumberFollowing(song, m) {
  if (!/^\d+$/.test(m.value)) return alert("Renumbering starts from a plain number.");
  const order = readingOrder(song.measures);
  const rest = order.slice(order.indexOf(m) + 1);
  if (!rest.length || !confirm(`Renumber the ${rest.length} following measures as ${Number(m.value) + 1}, ${Number(m.value) + 2}, ...?`)) return;
  pushUndo(song);
  let n = Number(m.value);
  for (const o of rest) Object.assign(o, { value: String(++n), edited: true });
  touch(song);
  redrawSong();
  renderSongs();
}

function selectStep(step) {
  const song = current();
  if (!song?.measures?.length) return;
  const order = readingOrder(song.measures);
  const i = order.findIndex((m) => m.id === selected);
  const next = order[i < 0 ? 0 : clamp(i + step, 0, order.length - 1)];
  const prevPage = findMeasure(selected)?.page;
  selected = next.id;
  if (prevPage !== undefined && prevPage !== next.page) drawPage(prevPage);
  drawPage(next.page);
  updateSelection();
  scrollToMeasure(next);
}

function scrollToMeasure(m) {
  const page = view.pages.get(m.page);
  if (!page) return;
  const main = $("main");
  const r = page.holder.getBoundingClientRect(), mr = main.getBoundingClientRect();
  const top = r.top + m.y * r.height, bottom = r.top + (m.y + m.h) * r.height;
  const barH = main.querySelector(".bar").offsetHeight;
  if (top < mr.top + barH + 10 || bottom > mr.bottom - 10) main.scrollBy({ top: top - mr.top - barH - mr.height / 4 });
}

// Toolbar for the selected measure.
function updateSelection() {
  const row = $("selRow");
  if (!row) return;
  const song = current();
  const m = findMeasure(selected);
  row.replaceChildren();
  if (!m) {
    row.innerHTML = `<span class="help">No measure selected.</span>`;
    return;
  }
  row.innerHTML = `<label>Value <input type="text" id="value"></label>
    <span class="help" id="conf"></span>
    <button id="bSplit" title="Split into two equal measures">Split</button>
    <button id="bMerge">Merge right (M)</button>
    <button id="bRenumber">Renumber following (N)</button>
    <button id="bDelete">Delete (Del)</button>`;
  $("value").value = m.value;
  $("conf").textContent = m.edited ? "edited or confirmed" : m.conf === null ? "" : `confidence ${m.conf.toFixed(2)}`;
  $("value").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      commitValue(song, m);
      selectStep(1);
      $("value")?.focus();
      $("value")?.select();
    } else if (ev.key === "Escape") $("value").blur();
    else if (ev.key === "Tab") {
      ev.preventDefault();
      commitValue(song, m);
      selectStep(ev.shiftKey ? -1 : 1);
      $("value")?.focus();
      $("value")?.select();
    }
    ev.stopPropagation();
  });
  $("value").addEventListener("change", () => commitValue(song, m));
  $("bSplit").addEventListener("click", () => splitAt(song, m, m.x + m.w / 2));
  $("bMerge").addEventListener("click", () => mergeRight(song, m));
  $("bRenumber").addEventListener("click", () => renumberFollowing(song, m));
  $("bDelete").addEventListener("click", () => deleteMeasure(song, m));
}

// Enter on an unchanged value confirms the measure (marks it as reviewed).
function commitValue(song, m) {
  const input = $("value");
  if (!input || !song.measures.includes(m)) return;
  const v = input.value.trim();
  if (v === m.value && m.edited) return;
  pushUndo(song);
  m.value = v;
  m.edited = true;
  touch(song);
  drawPage(m.page);
  updateInfo();
  renderSongs();
}

document.addEventListener("keydown", (ev) => {
  const song = current();
  if (!song || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const m = findMeasure(selected);
  const key = ev.key.toLowerCase();
  if ((ev.ctrlKey || ev.metaKey) && key === "z") {
    ev.preventDefault();
    ev.shiftKey ? undo(redoStack, undoStack) : undo(undoStack, redoStack);
  } else if ((ev.ctrlKey || ev.metaKey) && key === "y") {
    ev.preventDefault();
    undo(redoStack, undoStack);
  } else if (ev.key === "Tab") {
    ev.preventDefault();
    selectStep(ev.shiftKey ? -1 : 1);
  } else if (ev.key === "Escape") {
    selected = null;
    redrawSong();
    updateSelection();
  } else if (!m) return;
  else if (ev.key === "Delete" || ev.key === "Backspace") {
    ev.preventDefault();
    deleteMeasure(song, m);
  } else if (ev.key === "Enter") {
    ev.preventDefault();
    $("value").focus();
    $("value").select();
  } else if (key === "m" && !ev.ctrlKey && !ev.metaKey) mergeRight(song, m);
  else if (key === "n" && !ev.ctrlKey && !ev.metaKey) renumberFollowing(song, m);
});

// ---------------------------------------------------------------- detection

let detector = null;
async function getDetector() {
  detector ??= await MeasureDetector.create({
    models: { measurenet: "../models/measurenet.onnx", labelocr: "../models/labelocr.onnx" },
    decoder: await (await fetch("../models/decoder.json")).json(),
  });
  return detector;
}

// Detection runs one song at a time; the detector sees only the song's pages, like a training document.
let detectChain = Promise.resolve();

function enqueueDetection(song, ask) {
  if (running.has(song)) return;
  if (ask && song.measures?.length && !confirm(`Replace the ${song.measures.length} measures of this song with a new detection?`)) return;
  running.set(song, "queued");
  renderSongs();
  detectChain = detectChain.then(() => detectSong(song)).catch((err) => {
    status(`Error: ${err.message ?? err}`);
    console.error(err);
  });
}

async function detectSong(song) {
  const start = song.start, end = songEnd(song);
  const label = `"${songName(song)}"`;
  const progress = (text) => {
    running.set(song, text);
    renderSongs();
  };
  try {
    if (!state.songs.includes(song)) return;
    progress("loading models");
    status("Loading models...");
    const det = await getDetector();
    const n = end - start + 1;
    const pages = { numPages: n, getPage: (k) => pdf.getPage(start + k) };
    const names = { scan: "scanning", detect: "detecting", label: "reading numbers" };
    const weights = { scan: [0, 0.2], detect: [0.2, 0.75], label: [0.95, 0.05] };
    const t0 = performance.now();
    const results = await det.detect(pages, {
      startConvention: START_CONVENTION,
      onPageMeasures: (k, ms) => {
        pending.set(start + k, ms.map((r) => ({ x: r.layout.x, y: r.layout.y, w: r.layout.width, h: r.layout.height })));
        if (current() === song) drawPage(start + k);
      },
      onProgress: (done, total, stage) => {
        const [s0, share] = weights[stage];
        $("bar").value = s0 + share * (done / total);
        const text = stage === "label" ? names.label : `${names[stage]} ${done}/${total}`;
        progress(text);
        status(`${label}: ${text}`);
      },
    });
    // The song may have been split or merged while it was detected; its measures would land on the wrong pages.
    if (!state.songs.includes(song) || song.start !== start || songEnd(song) !== end) {
      status(`${label}: page range changed during detection; result discarded.`);
      return;
    }
    const replaced = song.measures;
    song.measures = results.map((r) => ({ id: nextId++, page: start + r.layout.page, x: r.layout.x, y: r.layout.y, w: r.layout.width, h: r.layout.height, value: r.value, conf: r.confidence, edited: false }));
    if (replaced) {
      undoStack.push({ song, measures: replaced });
      redoStack = [];
    }
    touch(song);
    $("bar").value = 1;
    status(`${label}: ${results.length} measures on ${n} pages in ${((performance.now() - t0) / 1000).toFixed(1)} s (${det.provider}).`);
  } finally {
    for (let p = start; p <= end; p++) pending.delete(p);
    running.delete(song);
    renderSongs();
    if (current() === song) {
      if (!findMeasure(selected)) selected = null;
      redrawSong();
      updateSelection();
    }
  }
}

$("detectAll").addEventListener("click", () => {
  const todo = state.songs.filter((s) => !s.skip && s.measures === null && !running.has(s));
  if (!todo.length) return status("Every song that is not skipped has measures.");
  todo.forEach((s) => enqueueDetection(s, false));
});

// ---------------------------------------------------------------- export

function exportProblems(song) {
  const errors = [], warnings = [];
  const ms = song.measures ?? [];
  if (!NUMBER_RE.test(song.number)) errors.push("The song number may contain only digits, letters, '.' and '-' (at most 16 characters).");
  if (!ms.length) errors.push("The song has no measures.");
  const bad = ms.filter((m) => !m.value || !LABEL_RE.test(m.value));
  if (bad.length) errors.push(`${bad.length} measures have an empty value or one outside the label grammar (digits followed by letters, e.g. 12, 23A, A): ${bad.slice(0, 8).map((m) => JSON.stringify(m.value)).join(", ")}`);
  const review = ms.filter((m) => !m.edited && m.conf !== null && m.conf < LOW_CONFIDENCE).length;
  if (review) warnings.push(`${review} low-confidence measures were not reviewed.`);
  const seen = new Map();
  for (const m of ms) seen.set(m.value, (seen.get(m.value) ?? 0) + 1);
  const dup = [...seen].filter(([v, c]) => v && c > 1).map(([v]) => v);
  if (dup.length) warnings.push(`Values used more than once: ${dup.slice(0, 10).join(", ")}${dup.length > 10 ? ", ..." : ""}.`);
  // The loader keeps only the later of two boxes with IoU >= 0.8 (docs/CONVENTIONS.md).
  let overlaps = 0;
  for (let i = 0; i < ms.length; i++) for (let j = i + 1; j < ms.length; j++) if (ms[i].page === ms[j].page && iou(ms[i], ms[j]) >= 0.8) overlaps++;
  if (overlaps) warnings.push(`${overlaps} pairs of nearly identical boxes; the loader keeps only one of each.`);
  const pagesWithout = [];
  for (let p = song.start; p <= songEnd(song); p++) if (!ms.some((m) => m.page === p)) pagesWithout.push(p + 1);
  if (pagesWithout.length) warnings.push(`Pages without measures (fine for pages without music): ${pagesWithout.join(", ")}.`);
  if (!song.number) warnings.push("The song has no number.");
  if (song.split === "none") warnings.push("Split is none: the document is written but not used by training until it is added to a split.");
  return { errors, warnings };
}

// Book: writes the song's pages and measures as a document. Document: writes its measures back in place.
async function exportSong(song) {
  if (song.skip) return alert("This song is marked as skipped.");
  if (running.has(song)) return alert("The song is being detected.");
  const input = $("value");
  if (input && document.activeElement === input) input.blur();
  const { errors, warnings } = exportProblems(song);
  const verb = isBook() ? "export" : "save";
  if (errors.length) return alert(`Cannot ${verb}:\n\n- ` + errors.join("\n- "));
  const start = song.start, end = songEnd(song);
  const target = !isBook() ? `dataset/documents/${song.docId}/` : song.docId ? `dataset/documents/${song.docId}/ (overwrite)` : "a new folder in dataset/documents/";
  const original = !isBook() && state.original ? "\n\nThis is an original document: its measure list (a playback sequence) is replaced by the curated list in reading order. The first version is kept as measures.orig.json." : "";
  const question = isBook() ? `Export "${songName(song)}" (pages ${start + 1}-${end + 1}, ${song.measures.length} measures) to ${target}, split ${song.split}?` : `Save ${song.measures.length} measures to ${target}, split ${song.split}?`;
  if (!confirm(question + original + (warnings.length ? "\n\nWarnings:\n- " + warnings.join("\n- ") : ""))) return;
  status(isBook() ? `Exporting "${songName(song)}"...` : "Saving...");
  try {
    const measures = readingOrder(song.measures).map((m) => {
      const x = clamp(m.x, 0, 1), y = clamp(m.y, 0, 1);
      return { layout: { page: m.page - start, x, y, width: Math.min(m.w, 1 - x), height: Math.min(m.h, 1 - y) }, value: m.value };
    });
    const payload = { docId: song.docId, measures, source: { show_id: state.showId, number: song.number }, split: song.split === "none" ? null : song.split, numPages: end - start + 1 };
    let r;
    if (isBook()) {
      const bytes = await pdf.extractPages([{ document: null, includePages: [[start, end]] }]);
      const form = new FormData();
      form.append("meta", JSON.stringify(payload));
      form.append("score", new Blob([bytes], { type: "application/pdf" }), "score.pdf");
      r = await fetch("/api/export", { method: "POST", body: form });
    } else r = await fetch(`/api/documents/${song.docId}`, { method: "PUT", body: JSON.stringify(payload) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error);
    song.docId = body.docId;
    song.modified = body.modified;
    song.exported = true;
    await loadDocs();
    renderShows();
    if (isBook()) await save();
    else {
      dirty = false;
      $("saved").textContent = `Saved ${new Date().toLocaleTimeString()}`;
    }
    status(`${isBook() ? "Exported" : "Saved"} "${songName(song)}" to ${body.dir}${body.split ? ` (${body.split})` : ", in no split"}.`);
  } catch (err) {
    status(`${isBook() ? "Export" : "Save"} failed: ${err.message ?? err}`);
    console.error(err);
  }
  renderSongs();
  updateInfo();
}

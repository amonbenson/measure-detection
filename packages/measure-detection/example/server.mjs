// Server for the curation page (no dependencies).
// Serves the repository root so the page can load the built package, the models and node_modules,
// plus a small API:
//   GET  /api/session/<fingerprint>  curation state of one book PDF (dataset/curation/<fingerprint>.json)
//   PUT  /api/session/<fingerprint>  save that state
//   GET  /api/documents              every document: id, show, number, split, curated, modified
//   GET  /api/documents/<id>         one document: source.json, measures.json, split
//   PUT  /api/documents/<id>         overwrite measures.json and source.json of an existing document
//   POST /api/export                 write a song of a book as a document (with its score.pdf) and add it to a split
// Usage (repository root): npm run example -w measure-detection   then open http://localhost:8080

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const dataset = path.join(root, "dataset");
const documents = path.join(dataset, "documents");
const curation = path.join(dataset, "curation");
const port = Number(process.env.PORT ?? 8080);
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".css": "text/css", ".pdf": "application/pdf" };

const FINGERPRINT_RE = /^[0-9a-f]{32}$/;
// Document and show ids have the shape of the Pocketbase record ids of the original documents.
const ID_RE = /^[a-z0-9]{15}$/;
const LABEL_RE = /^\d*[A-Za-z]*$/;
const NUMBER_RE = /^[0-9A-Za-z.-]{0,16}$/;
// test.txt is frozen (AGENTS.md): new documents go to train or val, test documents keep their split.
const WRITABLE_SPLITS = ["train", "val"];

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(JSON.stringify(body));
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}

const readJson = (file) => JSON.parse(fs.readFileSync(file, "utf8"));

function readSplit(name) {
  const file = path.join(dataset, "splits", `${name}.txt`);
  return fs.existsSync(file) ? fs.readFileSync(file, "utf8").split(/\r?\n/).map((l) => l.trim()).filter(Boolean) : [];
}

function splitOf(docId) {
  return ["train", "val", "test"].find((name) => readSplit(name).includes(docId)) ?? null;
}

function newId() {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  for (;;) {
    const id = Array.from(crypto.randomBytes(15), (b) => chars[b % chars.length]).join("");
    if (!fs.existsSync(path.join(documents, id))) return id;
  }
}

// Checks the measure list against the loader rules (src/mdet/data.py, docs/CONVENTIONS.md).
function validateMeasures(measures, numPages) {
  if (!Array.isArray(measures) || measures.length === 0) return "no measures";
  for (const m of measures) {
    const l = m?.layout;
    if (typeof m?.value !== "string" || !m.value || !LABEL_RE.test(m.value)) return `invalid value ${JSON.stringify(m?.value)}`;
    if (!l || !Number.isInteger(l.page) || l.page < 0 || l.page >= numPages) return `measure ${m.value}: page outside the document`;
    for (const k of ["x", "y", "width", "height"]) if (typeof l[k] !== "number" || !Number.isFinite(l[k])) return `measure ${m.value}: bad ${k}`;
    if (l.width <= 0 || l.height <= 0 || l.x < 0 || l.y < 0 || l.x + l.width > 1 + 1e-6 || l.y + l.height > 1 + 1e-6) return `measure ${m.value}: box outside the page`;
  }
  return null;
}

// source.json holds only what training may use: the show a document belongs to and its song number.
// Titles, show names and file names are never stored.
function validateSource(source) {
  if (!ID_RE.test(source?.show_id ?? "")) return "bad show id";
  if (typeof source.number !== "string" || !NUMBER_RE.test(source.number)) return "bad song number";
  return null;
}

// Writes measures.json and source.json (and score.pdf if given) and puts the document into `split`.
function writeDocument(docId, { measures, source, split, score }) {
  const dir = path.join(documents, docId);
  fs.mkdirSync(dir, { recursive: true });
  const srcFile = path.join(dir, "source.json");
  const old = fs.existsSync(srcFile) ? readJson(srcFile) : null;
  // Original documents keep their first measure list (a playback sequence from the server) as a backup.
  if (old && !old.curated && !fs.existsSync(path.join(dir, "measures.orig.json"))) fs.copyFileSync(path.join(dir, "measures.json"), path.join(dir, "measures.orig.json"));
  if (score) fs.writeFileSync(path.join(dir, "score.pdf"), score);
  const modified = new Date().toISOString();
  fs.writeFileSync(path.join(dir, "measures.json"), JSON.stringify(measures, null, 1) + "\n");
  // curated: created by this page from a book (its score.pdf may be replaced); original documents keep it unset.
  const curated = old ? old.curated === true : true;
  fs.writeFileSync(srcFile, JSON.stringify({ show_id: source.show_id, number: source.number, ...(curated ? { curated } : {}), modified }, null, 1) + "\n");

  // A document belongs to at most one split; moving it between train and val is allowed, test is never touched.
  if (!readSplit("test").includes(docId)) {
    for (const name of WRITABLE_SPLITS) {
      const ids = readSplit(name);
      const want = name === split;
      if (ids.includes(docId) === want) continue;
      const next = want ? [...ids, docId].sort() : ids.filter((d) => d !== docId);
      fs.writeFileSync(path.join(dataset, "splits", `${name}.txt`), next.join("\n") + "\n");
    }
  }
  return { docId, dir: path.relative(root, dir).replaceAll("\\", "/"), split: splitOf(docId), modified };
}

function checkPayload(docId, { measures, source, split, numPages }) {
  const inTest = docId != null && readSplit("test").includes(docId);
  if (split === "test" && !inTest) return "documents cannot be added to the frozen test split";
  if (inTest && split !== "test") return "documents cannot be removed from the frozen test split";
  if (split !== null && split !== "test" && !WRITABLE_SPLITS.includes(split)) return `split must be one of ${WRITABLE_SPLITS.join(", ")} or none`;
  return validateSource(source) ?? validateMeasures(measures, numPages);
}

// A song of a book: new document, or re-export of a document this page created.
async function exportDocument(req, res) {
  const form = await new Response(await readBody(req), { headers: { "Content-Type": req.headers["content-type"] ?? "" } }).formData();
  const meta = JSON.parse(form.get("meta"));
  const pdf = form.get("score");
  if (!(pdf instanceof Blob) || pdf.size === 0) return send(res, 400, { error: "missing score.pdf" });
  const score = Buffer.from(await pdf.arrayBuffer());
  if (score.subarray(0, 5).toString("latin1") !== "%PDF-") return send(res, 400, { error: "score is not a PDF" });
  const invalid = checkPayload(meta.docId, meta);
  if (invalid) return send(res, 400, { error: invalid });

  let docId = meta.docId;
  if (docId != null) {
    if (!ID_RE.test(docId)) return send(res, 400, { error: "bad document id" });
    const src = path.join(documents, docId, "source.json");
    // Replacing score.pdf is allowed only for documents that this page created, never for the original data.
    if (fs.existsSync(src) && readJson(src).curated !== true) return send(res, 409, { error: `${docId} is an original document` });
    if (readSplit("test").includes(docId)) return send(res, 409, { error: `${docId} is in the frozen test split` });
  } else docId = newId();
  send(res, 200, writeDocument(docId, { ...meta, score }));
}

// Edit of an existing document (its pages stay as they are).
async function saveDocument(req, res, docId) {
  if (!fs.existsSync(path.join(documents, docId, "score.pdf"))) return send(res, 404, { error: "no such document" });
  const meta = JSON.parse((await readBody(req)).toString("utf8"));
  const invalid = checkPayload(docId, meta);
  if (invalid) return send(res, 400, { error: invalid });
  send(res, 200, writeDocument(docId, meta));
}

function listDocuments() {
  const split = new Map(["train", "val", "test"].flatMap((name) => readSplit(name).map((d) => [d, name])));
  return fs
    .readdirSync(documents)
    .filter((d) => ID_RE.test(d) && fs.existsSync(path.join(documents, d, "source.json")))
    .map((d) => {
      const s = readJson(path.join(documents, d, "source.json"));
      return { id: d, show_id: s.show_id ?? null, number: s.number ?? "", curated: s.curated === true, modified: s.modified ?? null, split: split.get(d) ?? null };
    });
}

async function documentApi(req, res, docId) {
  if (!docId) return req.method === "GET" ? send(res, 200, listDocuments()) : send(res, 405, { error: "method not allowed" });
  if (!ID_RE.test(docId)) return send(res, 400, { error: "bad document id" });
  const dir = path.join(documents, docId);
  if (req.method === "PUT") return saveDocument(req, res, docId);
  if (req.method !== "GET") return send(res, 405, { error: "method not allowed" });
  if (!fs.existsSync(path.join(dir, "source.json"))) return send(res, 404, { error: "no such document" });
  const measures = fs.existsSync(path.join(dir, "measures.json")) ? readJson(path.join(dir, "measures.json")) : [];
  send(res, 200, { id: docId, source: readJson(path.join(dir, "source.json")), measures, split: splitOf(docId) });
}

async function session(req, res, fingerprint) {
  if (!FINGERPRINT_RE.test(fingerprint)) return send(res, 400, { error: "bad fingerprint" });
  const file = path.join(curation, `${fingerprint}.json`);
  if (req.method === "GET") {
    if (!fs.existsSync(file)) return send(res, 404, { error: "no session" });
    res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    return res.end(fs.readFileSync(file));
  }
  if (req.method === "PUT") {
    const body = await readBody(req);
    JSON.parse(body.toString("utf8")); // reject malformed state before overwriting the saved one
    fs.mkdirSync(curation, { recursive: true });
    // Write then rename, so a crash never leaves a truncated state file.
    fs.writeFileSync(file + ".tmp", body);
    fs.renameSync(file + ".tmp", file);
    return send(res, 200, { saved: true });
  }
  send(res, 405, { error: "method not allowed" });
}

async function api(req, res, url) {
  if (url.startsWith("/api/session/")) return session(req, res, url.slice("/api/session/".length));
  if (url === "/api/documents" || url.startsWith("/api/documents/")) return documentApi(req, res, url.slice("/api/documents/".length));
  if (url === "/api/export" && req.method === "POST") return exportDocument(req, res);
  send(res, 404, { error: "not found" });
}

http
  .createServer(async (req, res) => {
    // Cross-origin isolation: onnxruntime-web runs WASM with several threads only on isolated pages.
    // Everything here is same-origin, so require-corp blocks nothing.
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (url.startsWith("/api/")) {
      try {
        return await api(req, res, url);
      } catch (err) {
        console.error(err);
        return send(res, 500, { error: String(err) });
      }
    }
    if (url === "/") {
      res.writeHead(302, { Location: "/packages/measure-detection/example/curate.html" });
      return res.end();
    }
    const file = path.join(root, url);
    if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.writeHead(404);
      return res.end("not found");
    }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store" });
    fs.createReadStream(file).pipe(res);
  })
  .listen(port, "127.0.0.1", () => console.log(`Open http://localhost:${port}`));

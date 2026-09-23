// Minimal static server for the upload example (no dependencies).
// Serves the repository root so the page can load the built package, the models and node_modules.
// Usage (repository root): npm run example -w measure-detection   then open http://localhost:8080

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const port = Number(process.env.PORT ?? 8080);
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".css": "text/css" };

http
  .createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (url === "/") {
      res.writeHead(302, { Location: "/packages/measure-detection/example/index.html" });
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

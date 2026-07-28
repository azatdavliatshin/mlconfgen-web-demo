/**
 * Copy the ONNX Runtime and RDKit WASM artifacts out of node_modules into
 * public/, so the site serves them from its own origin.
 *
 * These are build outputs of pinned dependencies, not source — they are copied
 * on install/build rather than committed, which keeps ~32 MB of binaries out of
 * git while still giving the deployed page a CDN-free runtime.
 *
 * Self-hosting also fixes RDKit's `locateFile`: left to itself, Emscripten
 * resolves RDKit_minimal.wasm relative to the script URL, which under a
 * GitHub Pages project subdirectory hits the SPA fallback and returns HTML.
 */
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

const ASSETS = [
  // Only the jsep variant is needed — it is what ORT selects for the
  // single-threaded WASM backend this demo configures.
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm", "ort"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs", "ort"],
  ["@rdkit/rdkit/dist/RDKit_minimal.wasm", "rdkit"],
  ["@rdkit/rdkit/dist/RDKit_minimal.js", "rdkit"],
];

let copied = 0;
for (const [rel, destDir] of ASSETS) {
  const src = join(root, "node_modules", rel);
  const out = join(root, "public", destDir);
  const dest = join(out, rel.split("/").pop());

  try {
    statSync(src);
  } catch {
    console.error(
      `\n  Missing ${rel}\n  Run \`npm install\` first — runtime assets come from node_modules.\n`,
    );
    process.exit(1);
  }

  mkdirSync(out, { recursive: true });
  copyFileSync(src, dest);
  copied += 1;
}

console.log(`Synced ${copied} runtime assets into public/`);

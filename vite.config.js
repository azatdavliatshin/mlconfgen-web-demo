import { createReadStream, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const here = dirname(fileURLToPath(import.meta.url));

// GitHub Pages serves project sites from /<repo>/, so assets need that prefix.
// Override with BASE_PATH when deploying elsewhere (e.g. "/" for a user site).
const base = process.env.BASE_PATH ?? "/mlconfgen-web-demo/";

/**
 * Vite's dev middleware appends `?import` to dynamically-imported files and
 * tries to transform them as source. The ONNX Runtime `.mjs` glue in public/
 * is a prebuilt Emscripten artifact that must be served verbatim, so strip the
 * query and hand back the raw file before Vite's transform pipeline sees it.
 */
function serveOrtRaw() {
  return {
    name: "serve-ort-raw",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const m = req.url?.match(/\/ort\/([\w.-]+\.mjs)(\?.*)?$/);
        if (!m) return next();
        const file = join(here, "public", "ort", m[1]);
        if (!existsSync(file)) return next();
        res.setHeader("Content-Type", "text/javascript");
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  base,
  plugins: [serveOrtRaw()],
  // Select the onnxruntime-web build that fetches its WASM at runtime instead
  // of referencing it as a bundled asset — we serve those binaries ourselves
  // from public/ort/ (see ort.env.wasm.wasmPaths). Without this condition
  // Rollup also emits an unused ~26 MB copy into assets/.
  resolve: {
    conditions: ["onnxruntime-web-use-extern-wasm", "import", "module", "browser", "default"],
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 2000,
  },
  // The inference worker is an ES module (it imports mlconfgen/onnxruntime-web);
  // the default IIFE format can't code-split those dependencies.
  worker: {
    format: "es",
  },
  optimizeDeps: {
    // Emscripten/UMD builds Vite should not try to pre-bundle as ESM.
    exclude: ["onnxruntime-web"],
  },
  server: {
    headers: {
      // Enables the multi-threaded WASM backend where the host allows it.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
});

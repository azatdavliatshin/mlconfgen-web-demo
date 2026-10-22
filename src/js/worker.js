/**
 * Inference worker.
 *
 * All ONNX/RDKit work happens here so the main thread stays responsive —
 * running it inline freezes the tab for the whole job and Chrome eventually
 * shows "Page Unresponsive".
 *
 * Protocol (main → worker):
 *   { type: "load-hub", set }                       weight set id, see weights.js
 *   { type: "weights", kind: "egnn"|"adj", bytes }  local file, structured-cloned
 *   { type: "generate", config }
 *   { type: "animate", config }
 *   { type: "abort" }
 *
 * Worker → main:
 *   { type: "progress", file, loaded, total }       hub download, throttled
 *   { type: "loaded", set, files: [{ kind, file, bytes, cached }] }
 *   { type: "load-error", message }
 *   { type: "status", message }
 *   { type: "frame", step, total, molecule }     animate only
 *   { type: "done", molecules?, elapsedMs, ... }
 *   { type: "error", message, name }
 */
import { createGenerator, seed, WeightsManager } from "mlconfgen";
import * as ort from "onnxruntime-web";

import { KINDS, WEIGHT_SETS } from "./weights.js";

// Resolve against the site origin + base path rather than the worker's own
// URL: the worker lives at src/js/ in dev but assets/ in a build, so a relative
// path would point somewhere different in each.
const BASE = new URL(import.meta.env.BASE_URL, self.location.origin).href;

ort.env.wasm.wasmPaths = `${BASE}ort/`;
// Single-threaded on purpose. Multi-threaded WASM needs SharedArrayBuffer and
// therefore cross-origin isolation, which GitHub Pages does not provide — and
// measured locally *with* isolation, 4 threads made session creation slower
// (4220ms vs 3423ms), so there is nothing to gain by detecting it.
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const RDKIT_BASE = `${BASE}rdkit/`;

let rdkitPromise = null;

/**
 * Load RDKit inside the worker with an explicit `locateFile`.
 *
 * There is no DOM here, so the package's script-tag fallback cannot run. This
 * is a *module* worker, where `importScripts` throws — so the UMD build is
 * pulled in with a dynamic `import()` instead. RDKit_minimal.js sets
 * `globalThis.initRDKitModule` as a side effect and exports nothing usable, so
 * we read the global rather than the module namespace.
 *
 * Pinning `locateFile` stops Emscripten resolving the .wasm against the worker
 * URL, which under a Pages base path lands on the SPA fallback and yields HTML.
 */
function loadRdkit() {
  if (rdkitPromise) return rdkitPromise;

  rdkitPromise = (async () => {
    let init = globalThis.initRDKitModule;

    if (typeof init !== "function") {
      // RDKit_minimal.js is a UMD bundle. In a module worker `importScripts` is
      // unavailable, and a dynamic `import()` runs it as ESM — where its
      // wrapper finds no CommonJS, exports nothing, and sets no global. So
      // fetch the source and evaluate it, returning the initialiser the file
      // declares.
      const res = await fetch(`${RDKIT_BASE}RDKit_minimal.js`);
      if (!res.ok) {
        throw new Error(
          `Could not fetch RDKit_minimal.js (HTTP ${res.status}).`,
        );
      }
      const source = await res.text();
      // eslint-disable-next-line no-new-func
      init = new Function(`${source}\nreturn initRDKitModule;`)();
    }

    if (typeof init !== "function") {
      throw new Error("RDKit_minimal.js loaded but exposed no initialiser.");
    }
    return init({ locateFile: () => `${RDKIT_BASE}RDKit_minimal.wasm` });
  })().catch((err) => {
    rdkitPromise = null;
    throw err;
  });

  return rdkitPromise;
}

/**
 * The weight bytes live here, in the worker, whichever way they arrived. A
 * change of diffusion steps rebuilds the generator, and rebuilding from these
 * bytes needs neither the network nor the Cache API — which may be missing,
 * or may have refused to store a large file.
 */
const weights = { egnn: null, adj: null };
let generator = null;
let builtWithSteps = null;
let aborted = false;
// Bumped by every weights change, so a hub load that is overtaken by another
// load (or by a local file) drops its result instead of overwriting.
let weightsVersion = 0;

const post = (msg) => self.postMessage(msg);
const status = (message) => post({ type: "status", message });

// fetch hands over the body in small chunks — thousands of callbacks for the
// full set — so only a few progress messages per second cross to the page.
const PROGRESS_INTERVAL_MS = 100;
let lastProgressAt = 0;
const downloaded = new Set();

const weightsManager = new WeightsManager({
  onProgress: ({ file, loaded, total }) => {
    downloaded.add(file);
    const now = performance.now();
    if (loaded !== total && now - lastProgressAt < PROGRESS_INTERVAL_MS) return;
    lastProgressAt = now;
    post({ type: "progress", file, loaded, total });
  },
});

function resetWeights() {
  weightsVersion += 1;
  weights.egnn = null;
  weights.adj = null;
  generator = null;
  builtWithSteps = null;
}

/**
 * Fetch a weight set by name: from the Cache API when it is there, otherwise
 * from Hugging Face (and into the cache). One file after the other, as
 * createGenerator does. A file counts as cached when no download progress was
 * reported for it.
 */
async function loadHub(setId) {
  const set = WEIGHT_SETS[setId];
  if (!set) throw new Error(`Unknown weight set "${setId}".`);

  resetWeights();
  const version = weightsVersion;
  downloaded.clear();

  const files = [];
  for (const kind of KINDS) {
    const { file } = set[kind];
    const bytes = await weightsManager.resolve(file);
    if (version !== weightsVersion) return;
    weights[kind] = bytes;
    files.push({ kind, file, bytes: bytes.byteLength, cached: !downloaded.has(file) });
  }
  post({ type: "loaded", set: setId, files });
}

async function ensureGenerator(diffusionSteps) {
  if (!weights.egnn || !weights.adj) {
    throw new Error("Both ONNX files must be loaded first.");
  }
  if (generator && builtWithSteps === diffusionSteps) return generator;

  status("Compiling ONNX graphs — first run takes a moment…");
  generator = await createGenerator({
    ort,
    rdkitLoader: loadRdkit,
    egnnOnnx: weights.egnn,
    adjMatSeerOnnx: weights.adj,
    diffusionSteps,
  });
  builtWithSteps = diffusionSteps;
  return generator;
}

/** Molecules can't cross the worker boundary — send plain data. */
function serialize(mol) {
  return {
    molBlock: mol.toMolBlock(),
    nAtoms: mol.nAtoms,
    nBonds: mol.bonds?.length ?? 0,
  };
}

async function runGenerate(config) {
  const gen = await ensureGenerator(config.diffusionSteps);
  seed(config.seed);

  status("Running reverse diffusion…");
  const started = performance.now();

  const molecules = await gen.generateConformers({
    referenceContext: config.moi,
    nAtoms: config.nAtoms,
    nSamples: config.nSamples,
    variance: config.variance,
  });

  post({
    type: "done",
    molecules: molecules.map(serialize),
    requested: config.nSamples,
    elapsedMs: performance.now() - started,
  });
}

async function runAnimate(config) {
  const gen = await ensureGenerator(config.diffusionSteps);
  seed(config.seed);

  status("Streaming denoising trajectory…");
  const started = performance.now();
  let total = config.diffusionSteps;
  let last = [];

  // The whole batch is denoised together, so every sample has a trajectory —
  // stream them all and let the UI show one viewer per molecule.
  for await (const frame of gen.animateGeneration({
    referenceContext: config.moi,
    nAtoms: config.nAtoms,
    nSamples: config.nSamples,
    variance: config.variance,
    predictBonds: config.bondsEveryFrame ? "always" : "last",
    filterInvalid: false,
  })) {
    if (aborted) break;
    if (!frame.molecules.length) continue;
    total = frame.total;
    last = frame.molecules;
    post({
      type: "frame",
      step: frame.step,
      total: frame.total,
      molecules: frame.molecules.map(serialize),
    });
  }

  post({
    type: "done",
    animated: true,
    aborted,
    total,
    molecules: last.map(serialize),
    elapsedMs: performance.now() - started,
  });
}

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === "load-hub") {
    try {
      await loadHub(msg.set);
    } catch (err) {
      post({ type: "load-error", message: err?.message ?? String(err) });
    }
    return;
  }

  if (msg.type === "weights") {
    weightsVersion += 1;
    weights[msg.kind] = msg.bytes;
    // New weights invalidate any compiled generator.
    generator = null;
    builtWithSteps = null;
    post({ type: "weights-ok", kind: msg.kind, bytes: msg.bytes.byteLength });
    return;
  }

  if (msg.type === "abort") {
    aborted = true;
    return;
  }

  if (msg.type === "generate" || msg.type === "animate") {
    aborted = false;
    try {
      if (msg.type === "generate") await runGenerate(msg.config);
      else await runAnimate(msg.config);
    } catch (err) {
      post({
        type: "error",
        name: err?.name ?? "Error",
        message: err?.message ?? String(err),
      });
    }
  }
};

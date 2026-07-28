import { createGenerator, seed } from "mlconfgen";
import * as ort from "onnxruntime-web";

// Serve the ORT WASM binaries from our own origin (public/ort/, copied into the
// build as-is) so the demo has no CDN dependency at runtime. BASE_URL keeps the
// path correct under the GitHub Pages project subdirectory.
ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`;
ort.env.wasm.numThreads = 1; // cross-origin isolation isn't guaranteed on Pages
ort.env.logLevel = "error";

const RDKIT_BASE = `${import.meta.env.BASE_URL}rdkit/`;

/** @type {Promise<any> | null} */
let rdkitPromise = null;

/**
 * Load RDKit from our own origin with an explicit `locateFile`.
 *
 * The package can resolve RDKit on its own, but its browser fallback lets
 * Emscripten locate `RDKit_minimal.wasm` relative to the script URL — which
 * under a base-path deploy (GitHub Pages project site) resolves to the SPA
 * fallback and yields HTML instead of WASM. Pinning both paths removes that
 * ambiguity.
 */
function loadRdkit() {
  if (rdkitPromise) return rdkitPromise;

  rdkitPromise = new Promise((resolve, reject) => {
    if (typeof globalThis.initRDKitModule === "function") {
      resolve(globalThis.initRDKitModule);
      return;
    }
    const script = document.createElement("script");
    script.src = `${RDKIT_BASE}RDKit_minimal.js`;
    script.async = true;
    script.onload = () =>
      typeof globalThis.initRDKitModule === "function"
        ? resolve(globalThis.initRDKitModule)
        : reject(new Error("RDKit_minimal.js loaded but exposed no initialiser."));
    script.onerror = () =>
      reject(new Error(`Could not load ${script.src}`));
    document.head.appendChild(script);
  })
    .then((init) => init({ locateFile: () => `${RDKIT_BASE}RDKit_minimal.wasm` }))
    .catch((err) => {
      rdkitPromise = null;
      throw err;
    });

  return rdkitPromise;
}

/**
 * Wraps the mlconfgen package for the demo: holds the loaded weights, builds a
 * generator on demand, and exposes both the batch and streaming paths.
 *
 * Weights arrive as `Uint8Array` read from the user's own disk — the package
 * hands them straight to `ort.InferenceSession.create`, so nothing is uploaded
 * or re-hosted.
 */
export class GeneratorSession {
  constructor() {
    /** @type {Uint8Array|null} */ this.egnnBytes = null;
    /** @type {Uint8Array|null} */ this.adjBytes = null;
    this.generator = null;
    this.builtWithSteps = null;
  }

  get ready() {
    return this.egnnBytes != null && this.adjBytes != null;
  }

  setWeights(kind, bytes) {
    if (kind === "egnn") this.egnnBytes = bytes;
    else this.adjBytes = bytes;
    // Weights changed — force a rebuild on next use.
    this.generator = null;
    this.builtWithSteps = null;
  }

  /**
   * Build (or reuse) the underlying generator. Rebuilt only when the diffusion
   * step count changes, since that is baked into the noise schedule at
   * construction time.
   */
  async ensure(diffusionSteps, onProgress) {
    if (!this.ready) throw new Error("Both ONNX files must be loaded first.");

    if (this.generator && this.builtWithSteps === diffusionSteps) {
      return this.generator;
    }

    onProgress?.("Compiling ONNX graphs — first run takes a moment…");

    this.generator = await createGenerator({
      ort,
      rdkitLoader: loadRdkit,
      egnnOnnx: this.egnnBytes,
      adjMatSeerOnnx: this.adjBytes,
      diffusionSteps,
    });
    this.builtWithSteps = diffusionSteps;
    return this.generator;
  }

  /** Batch generation — returns standardized, validity-filtered molecules. */
  async generate(config, onProgress) {
    const gen = await this.ensure(config.diffusionSteps, onProgress);
    seed(config.seed);

    onProgress?.("Running reverse diffusion…");
    const started = performance.now();

    const molecules = await gen.generateConformers({
      referenceContext: config.moi,
      nAtoms: config.nAtoms,
      nSamples: config.nSamples,
      variance: config.variance,
    });

    return { molecules, elapsedMs: performance.now() - started };
  }

  /**
   * Streaming generation — yields one frame per denoising step so the UI can
   * show the molecule condensing out of noise.
   *
   * `predictBonds: "always"` makes every frame carry real predicted
   * connectivity rather than a distance guess, which is what makes the
   * animation legible; it costs one AdjMatSeer call per step.
   */
  async *animate(config, onProgress) {
    const gen = await this.ensure(config.diffusionSteps, onProgress);
    seed(config.seed);

    yield* gen.animateGeneration({
      referenceContext: config.moi,
      nAtoms: config.nAtoms,
      nSamples: config.nSamples,
      variance: config.variance,
      predictBonds: config.bondsEveryFrame ? "always" : "last",
      filterInvalid: false,
    });
  }
}

/** Read a File into the Uint8Array the package expects. */
export async function readWeights(file) {
  const buf = await file.arrayBuffer();
  return new Uint8Array(buf);
}

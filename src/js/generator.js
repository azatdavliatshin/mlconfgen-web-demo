/**
 * Main-thread client for the inference worker.
 *
 * Inference is CPU-bound WASM that runs for tens of seconds; on the main thread
 * it blocks rendering and input for the whole job, which Chrome reports as
 * "Page Unresponsive". Everything heavy therefore lives in worker.js and this
 * module just marshals messages.
 */

/** Read a File into the Uint8Array the package expects. */
export async function readWeights(file) {
  return new Uint8Array(await file.arrayBuffer());
}

export class GeneratorSession {
  /**
   * @param {object} [options]
   * @param {(event: object) => void} [options.onWeights] hub loading events:
   *   `{ type: "progress", file, loaded, total }`,
   *   `{ type: "loaded", set, files, restored }`,
   *   `{ type: "load-error", message, restored }`
   */
  constructor({ onWeights } = {}) {
    this.onWeights = onWeights;
    // "hub" (a named set, fetched by the worker) or "local" (files from disk).
    this.source = null;
    this.hubSet = null;
    this.loaded = { egnn: false, adj: false };
    this.loading = false;
    this._load = null;
    this._pending = null;
    // Local files only: kept so a terminated worker (see abort) can be handed
    // the weights again without the user re-picking the files. Hub weights are
    // not copied here; a fresh worker resolves them again, normally from the
    // browser cache.
    this._bytes = { egnn: null, adj: null };
    this._spawn();
  }

  _spawn() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (e) => this._dispatch(e.data);
    this.worker.onerror = (e) => {
      const err = new Error(e.message || "Inference worker crashed.");
      this._reject(err);
      this._settleLoad(err);
    };

    // Restore weights into the fresh worker.
    if (this.source === "hub" && this.hubSet) {
      this._loadHub(this.hubSet, { restored: true }).catch(() => {
        // Reported through onWeights; nothing is waiting on this promise.
      });
      return;
    }
    for (const kind of ["egnn", "adj"]) {
      const bytes = this._bytes[kind];
      if (bytes) this.worker.postMessage({ type: "weights", kind, bytes });
    }
  }

  get ready() {
    return !this.loading && this.loaded.egnn && this.loaded.adj;
  }

  /**
   * Load a named weight set in the worker: from the browser cache when it is
   * there, otherwise downloaded from Hugging Face. Resolves with
   * `{ set, files: [{ kind, file, bytes, cached }] }`.
   */
  loadFromHub(setId) {
    if (this._pending) {
      return Promise.reject(new Error("Wait for the current run to finish."));
    }
    if (this.loading) {
      return Promise.reject(new Error("Weights are already loading."));
    }
    this._bytes = { egnn: null, adj: null };
    return this._loadHub(setId, { restored: false });
  }

  _loadHub(setId, { restored }) {
    this.source = "hub";
    this.hubSet = setId;
    this.loaded = { egnn: false, adj: false };
    this.loading = true;
    return new Promise((resolve, reject) => {
      this._load = { resolve, reject, restored };
      this.worker.postMessage({ type: "load-hub", set: setId });
    });
  }

  /**
   * Finish the in-flight hub load: with the worker's result, or an Error.
   * `quiet` skips the onWeights report, for a load the user replaced.
   */
  _settleLoad(result, { quiet = false } = {}) {
    const load = this._load;
    if (!load) return;
    this._load = null;
    this.loading = false;
    if (result instanceof Error) {
      this.loaded = { egnn: false, adj: false };
      if (!quiet) {
        this.onWeights?.({ type: "load-error", message: result.message, restored: load.restored });
      }
      load.reject(result);
      return;
    }
    this.loaded = { egnn: true, adj: true };
    this.onWeights?.({ ...result, restored: load.restored });
    load.resolve(result);
  }

  /**
   * Hand a local file to the worker.
   *
   * The buffer is copied rather than transferred: a copy costs one structured
   * clone of the file once, but transferring would detach the only reference we
   * have, leaving nothing to re-send if the worker is later terminated.
   *
   * Switching from a hub set starts a fresh local pair — sets are never mixed.
   */
  setWeights(kind, bytes) {
    if (this.source !== "local") {
      this._settleLoad(new Error("Replaced by local files."), { quiet: true });
      this.source = "local";
      this.hubSet = null;
      this.loaded = { egnn: false, adj: false };
    }
    this.loaded[kind] = true;
    this._bytes[kind] = bytes;
    this.worker.postMessage({ type: "weights", kind, bytes });
  }

  _dispatch(msg) {
    switch (msg.type) {
      case "progress":
        this.onWeights?.(msg);
        return;
      case "loaded":
        this._settleLoad(msg);
        return;
      case "load-error":
        this._settleLoad(new Error(msg.message));
        return;
    }

    const p = this._pending;
    if (!p) return;

    switch (msg.type) {
      case "status":
        p.onStatus?.(msg.message);
        break;
      case "frame":
        p.onFrame?.(msg);
        break;
      case "done":
        this._pending = null;
        p.resolve(msg);
        break;
      case "error": {
        this._pending = null;
        const err = new Error(msg.message);
        err.name = msg.name;
        p.reject(err);
        break;
      }
    }
  }

  _reject(err) {
    const p = this._pending;
    this._pending = null;
    p?.reject(err);
  }

  _run(type, config, { onStatus, onFrame } = {}) {
    if (!this.ready) {
      return Promise.reject(
        new Error(this.loading ? "Weights are still loading." : "Both ONNX files must be loaded first."),
      );
    }
    if (this._pending) {
      return Promise.reject(new Error("A run is already in progress."));
    }
    return new Promise((resolve, reject) => {
      this._pending = {
        resolve,
        reject,
        onStatus,
        onFrame,
        // animate yields between steps; generate does not.
        cooperative: type === "animate",
      };
      this.worker.postMessage({ type, config });
    });
  }

  /** Batch generation → { molecules, requested, elapsedMs }. */
  generate(config, onStatus) {
    return this._run("generate", config, { onStatus });
  }

  /** Streaming generation; `onFrame` fires per denoising step. */
  animate(config, { onStatus, onFrame } = {}) {
    return this._run("animate", config, { onStatus, onFrame });
  }

  /**
   * Cancel the in-flight run.
   *
   * `animateGeneration` yields between steps, so it can stop cooperatively and
   * keep the compiled generator warm. `generateConformers` is a single opaque
   * call with no cancellation point, so the only way to actually stop it is to
   * terminate the worker — which discards the compiled ONNX graphs and makes
   * the next run pay compilation again — and, for a hub set, resolve the
   * weights again (from the browser cache, or by downloading). That is the
   * honest trade: a button that says Abort should abort.
   */
  abort() {
    if (!this._pending) return;

    if (this._pending.cooperative) {
      this.worker.postMessage({ type: "abort" });
      return;
    }

    this.worker.terminate();
    const p = this._pending;
    this._pending = null;
    this._spawn();
    p.resolve({ aborted: true, terminated: true, elapsedMs: 0 });
  }
}

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
  constructor() {
    this.loaded = { egnn: false, adj: false };
    this._pending = null;
    // Kept so a terminated worker (see abort) can be handed the weights again
    // without the user re-picking the files.
    this._bytes = { egnn: null, adj: null };
    this._spawn();
  }

  _spawn() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (e) => this._dispatch(e.data);
    this.worker.onerror = (e) =>
      this._reject(new Error(e.message || "Inference worker crashed."));

    // Restore weights into the fresh worker.
    for (const kind of ["egnn", "adj"]) {
      const bytes = this._bytes[kind];
      if (bytes) this.worker.postMessage({ type: "weights", kind, bytes });
    }
  }

  get ready() {
    return this.loaded.egnn && this.loaded.adj;
  }

  /**
   * Hand weights to the worker.
   *
   * The buffer is copied rather than transferred: a copy costs one structured
   * clone of ~90 MB once, but transferring would detach the only reference we
   * have, leaving nothing to re-send if the worker is later terminated.
   */
  setWeights(kind, bytes) {
    this.loaded[kind] = true;
    this._bytes[kind] = bytes;
    this.worker.postMessage({ type: "weights", kind, bytes });
  }

  _dispatch(msg) {
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
      return Promise.reject(new Error("Both ONNX files must be loaded first."));
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
   * the next run pay compilation again. That is the honest trade: a button that
   * says Abort should abort.
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

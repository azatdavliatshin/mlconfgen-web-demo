import { GeneratorSession, readWeights } from "./generator.js";
import { CUSTOM, PRESETS } from "./presets.js";
import { ViewerPool } from "./viewer.js";
import { CACHE_NAME, DEFAULT_SET, KINDS, mb, setBytes, WEIGHT_SETS } from "./weights.js";

const $ = (id) => document.getElementById(id);

const el = {
  presets: $("presets"),
  presetLabel: $("preset-label"),
  status: $("status"),
  run: $("run"),
  stop: $("stop"),
  progress: $("progress"),
  progressBar: $("progress-bar"),
  progressTxt: $("progress-txt"),
  grid: $("grid"),
  stats: $("stats"),
  empty: $("empty"),
  code: $("code"),
  copyCode: $("copy-code"),
  moi: ["c-moi0", "c-moi1", "c-moi2"].map($),
  vMoi: $("v-moi"),
  estimate: $("estimate"),
  srcHubBtn: $("src-hub-btn"),
  srcLocalBtn: $("src-local-btn"),
  srcHub: $("src-hub"),
  srcLocal: $("src-local"),
  hubFiles: $("hub-files"),
  hubLoad: $("hub-load"),
  setSeg: $("set-seg"),
  setNote: $("set-note"),
  cacheInfo: $("cache-info"),
  cacheClear: $("cache-clear"),
  random: $("random"),
};

const session = new GeneratorSession({ onWeights: onHubEvent });
let hubSet = WEIGHT_SETS[DEFAULT_SET];
const pool = new ViewerPool();
let activePreset = PRESETS[0];
let running = false;
// "generate" | "animate" | "random" while a run is in flight.
let runKind = null;
// File names in the browser's weights cache; null when the Cache API is unavailable.
let cachedFiles = new Set();

/* ── Config ──────────────────────────────────────────────────────────── */

function config() {
  const raw = $("c-mode").value;
  return {
    moi: el.moi.map((i) => Number(i.value) || 0),
    nAtoms: Number($("c-atoms").value),
    nSamples: Number($("c-samples").value),
    variance: Number($("c-variance").value),
    diffusionSteps: Number($("c-steps").value),
    seed: Number($("c-seed").value) || 0,
    // "animate-bonds" is the animate path with AdjMatSeer on every frame. That
    // costs one extra inference per denoising step and dominates runtime under
    // WASM, so plain "animate" predicts bonds on the final frame only.
    mode: raw === "generate" ? "generate" : "animate",
    bondsEveryFrame: raw === "animate-bonds",
  };
}

function setStatus(text, kind = "") {
  el.status.textContent = text;
  el.status.className = kind ? `status ${kind}` : "status";
}

/* ── Presets ─────────────────────────────────────────────────────────── */

function buildPresets() {
  for (const p of PRESETS) {
    const b = document.createElement("button");
    b.className = "preset";
    b.textContent = p.name;
    b.dataset.id = p.id;
    b.title = p.note;
    b.addEventListener("click", () => applyPreset(p));
    el.presets.appendChild(b);
  }
  applyPreset(PRESETS[0]);
}

function applyPreset(p) {
  activePreset = p;
  el.moi.forEach((input, i) => (input.value = p.moi[i]));
  $("c-atoms").value = p.nAtoms;
  $("c-variance").value = p.variance;
  syncRangeLabels();
  refreshPresetState();
  updateCode();
}

/**
 * A preset owns the MOI triplet, the atom count and the variance. Touching any
 * of them by hand puts us off the named preset, so nothing stays highlighted
 * and the labels read "Custom" — there is no Custom button to click, the state
 * is entirely derived from the controls.
 */
function isDrifted() {
  const p = activePreset;
  return (
    p.moi.some((v, i) => Number(el.moi[i].value) !== v) ||
    Number($("c-atoms").value) !== p.nAtoms ||
    Number($("c-variance").value) !== p.variance
  );
}

function refreshPresetState() {
  const custom = isDrifted();
  el.presetLabel.textContent = custom ? CUSTOM.name : activePreset.name;
  el.vMoi.textContent = custom ? "custom" : activePreset.name.toLowerCase();
  for (const b of el.presets.children) {
    b.classList.toggle("on", !custom && b.dataset.id === activePreset.id);
  }
}

/* ── Live labels + code sample ───────────────────────────────────────── */

const RANGES = [
  ["c-samples", "v-samples"],
  ["c-steps", "v-steps"],
  ["c-atoms", "v-atoms"],
  ["c-variance", "v-variance"],
];

function syncRangeLabels() {
  for (const [input, out] of RANGES) $(out).textContent = $(input).value;
  updateEstimate();
}

// Fitted to local measurements under single-threaded WASM: 1x10 = 19 s warm,
// 2x10 ≈ 74 s, 3x20 = 129 s. Cost is close to linear in (steps x samples) at
// roughly 2.15 s each; a cold run adds ~4 s of graph compilation. Heavily
// machine-dependent and deliberately rough — the point is to stop someone
// unknowingly queuing a forty-minute run.
const SECONDS_PER_STEP_SAMPLE = 2.15;

function updateEstimate() {
  const c = config();
  const secs = c.diffusionSteps * c.nSamples * SECONDS_PER_STEP_SAMPLE;
  const text =
    secs < 90
      ? `~${Math.round(secs)}s`
      : `~${Math.round(secs / 60)} min`;
  el.estimate.textContent = `Estimated runtime ${text} (plus ~5s model compile on the first run).`;
  el.estimate.className = secs > 300 ? "ctrl-note warn" : "ctrl-note";
}

function updateCode() {
  const c = config();
  const call =
    c.mode === "animate"
      ? `for await (const frame of gen.animateGeneration({
  referenceContext: [${c.moi.join(", ")}],
  nAtoms: ${c.nAtoms},
  nSamples: ${c.nSamples},
  variance: ${c.variance},
  predictBonds: ${c.bondsEveryFrame ? '"always"' : '"last"'},
})) {
  render(frame.molecules, frame.step, frame.total);
}`
      : `const mols = await gen.generateConformers({
  referenceContext: [${c.moi.join(", ")}],
  nAtoms: ${c.nAtoms},
  nSamples: ${c.nSamples},
  variance: ${c.variance},
});`;

  const src = `import { createGenerator, seed } from "mlconfgen";
import * as ort from "onnxruntime-web";

seed(${c.seed});

// Weights come from the user's own disk — never uploaded.
const gen = await createGenerator({
  ort,
  egnnOnnx: egnnBytes,        // Uint8Array
  adjMatSeerOnnx: adjBytes,   // Uint8Array
  diffusionSteps: ${c.diffusionSteps},
});

${call}`;

  el.code.innerHTML = highlight(src);
}

const KEYWORDS = /^(?:import|from|const|await|for|of|new|return)$/;
const FUNCS =
  /^(?:createGenerator|generateConformers|animateGeneration|seed|render)$/;

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;" };
const escape = (s) => s.replace(/[&<>]/g, (c) => ESC[c]);

/**
 * Minimal JS highlighter.
 *
 * Single pass over the source: comments and strings are consumed as whole
 * tokens before anything else is examined, so a keyword inside a comment (or a
 * `//` inside a string) can't be rewritten. Chained `String.replace` calls get
 * this wrong — a later pass happily matches inside the markup an earlier one
 * emitted.
 */
function highlight(src) {
  const tokens = src.match(/\/\/[^\n]*|"[^"\n]*"|[A-Za-z_$][\w$]*|\d+\.?\d*|\s+|[^]/g) ?? [];

  return tokens
    .map((t) => {
      if (t.startsWith("//")) return `<span class="c">${escape(t)}</span>`;
      if (t.startsWith('"')) return `<span class="s">${escape(t)}</span>`;
      if (KEYWORDS.test(t)) return `<span class="k">${t}</span>`;
      if (FUNCS.test(t)) return `<span class="f">${t}</span>`;
      if (/^\d/.test(t)) return `<span class="n">${t}</span>`;
      return escape(t);
    })
    .join("");
}

/* ── Weight loading ──────────────────────────────────────────────────── */

function showSource(source) {
  const hub = source === "hub";
  el.srcHub.hidden = !hub;
  el.srcLocal.hidden = hub;
  el.srcHubBtn.classList.toggle("on", hub);
  el.srcLocalBtn.classList.toggle("on", !hub);
  el.srcHubBtn.setAttribute("aria-pressed", String(hub));
  el.srcLocalBtn.setAttribute("aria-pressed", String(!hub));
}

function renderSets() {
  for (const set of Object.values(WEIGHT_SETS)) {
    const b = document.createElement("button");
    b.className = "seg-btn";
    b.dataset.set = set.id;
    b.textContent = [set.name, `${mb(setBytes(set))} MB`, set.tag].filter(Boolean).join(" · ");
    b.addEventListener("click", () => pickSet(set.id));
    el.setSeg.appendChild(b);
  }
  pickSet(hubSet.id);
}

/** Select a set to load. What is already loaded stays loaded until Load is clicked. */
function pickSet(id) {
  hubSet = WEIGHT_SETS[id];
  for (const b of el.setSeg.children) {
    const on = b.dataset.set === id;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", String(on));
  }
  el.setNote.textContent = hubSet.note;
  renderHubFiles();
}

/**
 * Read which weight files are in the browser cache and show it. Reads the
 * cache directly (see CACHE_NAME) so no request leaves the page.
 */
async function refreshCacheInfo() {
  try {
    if (typeof caches === "undefined") throw new Error("no Cache API");
    // `has` first: `open` would create an empty cache as a side effect.
    const keys = (await caches.has(CACHE_NAME))
      ? await (await caches.open(CACHE_NAME)).keys()
      : [];
    cachedFiles = new Set(
      keys.map((req) => decodeURIComponent(new URL(req.url).pathname.split("/").pop())),
    );
  } catch {
    cachedFiles = null;
  }
  renderCacheInfo();
  if (!session.loading) renderHubFiles();
}

function renderCacheInfo() {
  if (cachedFiles === null) {
    el.cacheInfo.innerHTML =
      '<span class="warn">This browser offers no cache here, so the weights are downloaded on every load.</span>';
    el.cacheClear.hidden = true;
    return;
  }
  const known = Object.values(WEIGHT_SETS).flatMap((set) => KINDS.map((k) => set[k]));
  const cached = known.filter((f) => cachedFiles.has(f.file));
  const bytes = cached.reduce((sum, f) => sum + f.bytes, 0);
  el.cacheInfo.textContent = cached.length
    ? `Cached in this browser: ${cached.length} file${cached.length > 1 ? "s" : ""}, ${mb(bytes)} MB.`
    : "Nothing cached in this browser yet.";
  el.cacheClear.hidden = !cachedFiles.size;
}

async function clearCache() {
  el.cacheClear.disabled = true;
  try {
    await session.clearCache();
    setStatus(
      session.ready && session.source === "hub"
        ? "Cached weights deleted — the loaded weights stay usable until you reload the page."
        : "Cached weights deleted.",
      "ok",
    );
  } catch (err) {
    setStatus(`Could not clear the cache: ${err.message}`, "err");
  } finally {
    el.cacheClear.disabled = false;
    await refreshCacheInfo();
  }
}

/** One row per file of the selected set: name, size, progress bar. */
function renderHubFiles() {
  el.hubFiles.innerHTML = "";
  for (const kind of KINDS) {
    const { file, bytes } = hubSet[kind];
    const row = document.createElement("div");
    row.className = "dl";
    row.dataset.file = file;
    row.innerHTML = `
      <div class="dl-hd">
        <span class="dl-name">${file}</span>
        <span class="dl-state">${mb(bytes)} MB${cachedFiles?.has(file) ? " · cached" : ""}</span>
      </div>
      <div class="dl-bar"><div></div></div>`;
    el.hubFiles.appendChild(row);
  }
  el.hubLoad.textContent = `Load ${hubSet.name} weights · ${mb(setBytes(hubSet))} MB`;
}

function hubRow(file) {
  return [...el.hubFiles.children].find((r) => r.dataset.file === file);
}

function setRow(file, fraction, text, done = false) {
  const row = hubRow(file);
  if (!row) return;
  row.querySelector(".dl-bar > div").style.width = `${Math.round(fraction * 100)}%`;
  row.querySelector(".dl-state").textContent = text;
  row.classList.toggle("done", done);
}

function setLoading(on) {
  el.hubLoad.disabled = on;
  el.srcLocalBtn.disabled = on;
  el.srcHubBtn.disabled = on;
  el.cacheClear.disabled = on;
  for (const b of el.setSeg.children) b.disabled = on;
}

async function loadHub() {
  renderHubFiles();
  setLoading(true);
  refreshReady();
  setStatus(`Loading ${hubSet.name} weights — from the browser cache if they are there…`);
  // Outcome and errors are reported through onHubEvent.
  await session.loadFromHub(hubSet.id).catch(() => {});
}

/**
 * Hub events from the session, for a load the user started or for the
 * automatic reload after an abort restarted the worker.
 */
function onHubEvent(evt) {
  if (evt.type === "progress") {
    const { file, loaded, total } = evt;
    const of = total ? ` / ${mb(total)} MB` : " MB";
    setRow(file, total ? loaded / total : 0, `${mb(loaded)}${of}`);
    if (!running) setStatus(`Downloading ${file} from Hugging Face…`);
    return;
  }

  setLoading(false);

  if (evt.type === "loaded") {
    const what = evt.restored ? "Weights reloaded into a fresh worker" : `${WEIGHT_SETS[evt.set].name} weights loaded`;
    const fromCache = evt.files.every((f) => f.cached);
    const how = fromCache ? "from the browser cache" : "downloaded from Hugging Face";
    refreshReady();
    setStatus(`${what} (${how}) — ready to generate.`, "ok");
    showLoaded(evt);
    return;
  }

  // load-error
  renderHubFiles();
  refreshReady();
  setStatus(`Could not load the weights: ${evt.message}`, "err");
}

/**
 * Mark the loaded files, then check they really made it into the cache: the
 * package only warns in the console when the browser refuses to store a file
 * (quota, private mode, low disk), and the next visit would download it again.
 */
async function showLoaded(evt) {
  await refreshCacheInfo();
  // The user may have picked the other set meanwhile; rows show that one.
  if (evt.set !== hubSet.id) return;
  for (const f of evt.files) {
    setRow(f.file, 1, `${mb(f.bytes)} MB · ${f.cached ? "from cache" : "downloaded"}`, true);
  }
  if (cachedFiles === null) return;
  const missing = evt.files.filter((f) => !cachedFiles.has(f.file));
  if (missing.length) {
    el.cacheInfo.innerHTML =
      `<span class="warn">The browser did not keep ${missing.map((f) => f.file).join(" and ")} ` +
      "in its cache, so it will be downloaded again next time.</span>";
  }
}

function wireDrop(kind, dropId, inputId, fileId, expectHint) {
  const drop = $(dropId);
  const input = $(inputId);
  const label = $(fileId);

  const accept = async (file) => {
    if (!file) return;
    if (!file.name.endsWith(".onnx")) {
      setStatus(`"${file.name}" is not an .onnx file.`, "err");
      return;
    }
    if (session.loading) {
      setStatus("Wait for the Hugging Face download to finish.", "err");
      return;
    }
    label.textContent = `${file.name} · loading…`;
    try {
      const bytes = await readWeights(file);
      if (session.source !== "local") renderHubFiles();
      session.setWeights(kind, bytes);
      label.textContent = `${file.name} · ${mb(bytes.byteLength)} MB`;
      drop.classList.add("loaded");
      refreshReady();
      if (session.ready) setStatus("Local weights loaded — ready to generate.", "ok");
    } catch (err) {
      label.textContent = expectHint;
      setStatus(`Could not read ${file.name}: ${err.message}`, "err");
    }
  };

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => accept(input.files[0]));

  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("hot");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("hot");
    accept(e.dataTransfer.files[0]);
  });
}

/**
 * Enable Generate when a complete pair is loaded. Only the waiting message is
 * set here; loads report their own outcome.
 */
function refreshReady() {
  el.run.disabled = running || !session.ready;
  el.random.disabled = el.run.disabled;
  if (session.ready || session.loading) return;
  if (session.source === "local") {
    const need = !session.loaded.egnn ? "EGNN" : "AdjMatSeer";
    setStatus(`Waiting for the ${need} file…`);
  } else {
    setStatus("Load the model weights above to start.");
  }
}

/** Clear the local drop zones — they no longer describe what is loaded. */
function resetDrops() {
  for (const kind of KINDS) {
    $(`drop-${kind}`).classList.remove("loaded");
    $(`file-${kind}`).textContent = `e.g. ${WEIGHT_SETS[DEFAULT_SET][kind].file}`;
  }
}

/* ── Rendering ───────────────────────────────────────────────────────── */

/** `mol` is the worker's plain payload: { molBlock, nAtoms, nBonds }. */
function makeCard(index, mol, { label } = {}) {
  const card = document.createElement("article");
  card.className = "card";
  const atoms = mol.nAtoms;
  const bonds = mol.nBonds;

  card.innerHTML = `
    <div class="card-hd">
      <span class="card-id">${label ?? `SAMPLE ${String(index + 1).padStart(2, "0")}`}</span>
      <span class="card-meta">${atoms} atoms · ${bonds} bonds</span>
    </div>
    <div class="viewer"></div>
    <div class="smiles"></div>
    <div class="card-ft">
      <button class="mini" data-act="mol">Copy molfile</button>
      <button class="mini" data-act="dl">Download</button>
    </div>`;

  const block = mol.molBlock;
  const viewer = pool.create(card.querySelector(".viewer"), { spin: true });
  viewer.setMolecule(block);

  card.querySelector('[data-act="mol"]').addEventListener("click", (e) => {
    navigator.clipboard.writeText(block);
    e.target.textContent = "Copied";
    setTimeout(() => (e.target.textContent = "Copy molfile"), 1200);
  });
  card.querySelector('[data-act="dl"]').addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([block], { type: "chemical/x-mdl-molfile" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `mlconfgen_${index + 1}.mol`;
    a.click();
    URL.revokeObjectURL(url);
  });

  return { card, viewer };
}

function showStats(entries) {
  el.stats.innerHTML = entries
    .map(
      ([v, k]) =>
        `<div class="stat"><div class="stat-v">${v}</div><div class="stat-k">${k}</div></div>`,
    )
    .join("");
  el.stats.hidden = false;
}

function setProgress(done, total, text) {
  el.progress.hidden = false;
  const pct = total ? Math.round((done / total) * 100) : 0;
  el.progressBar.style.width = `${pct}%`;
  el.progressTxt.textContent = text ?? `${pct}%`;
}

function resetOutput() {
  pool.clear();
  el.grid.innerHTML = "";
  el.stats.hidden = true;
  el.empty.hidden = true;
}

/* ── Run ─────────────────────────────────────────────────────────────── */

/** @param {"config"|"random"} what — the Generate button runs the selected mode */
async function run(what = "config") {
  if (running) return;
  const c = config();
  runKind = what === "random" ? "random" : c.mode;
  running = true;
  el.run.disabled = true;
  el.random.disabled = true;
  (runKind === "random" ? el.random : el.run).textContent = "Working…";
  el.stop.hidden = false;
  resetOutput();

  try {
    if (runKind === "animate") await runAnimate(c);
    else if (runKind === "random") await runGenerate(c, { random: true });
    else await runGenerate(c);
  } catch (err) {
    console.error(err);
    const hint =
      err?.name === "RdkitLoadError"
        ? " (RDKit failed to initialise — reload the page)"
        : "";
    setStatus(`Failed: ${err.message}${hint}`, "err");
    el.empty.hidden = false;
    el.empty.innerHTML = `Generation failed. Check the browser console for details.`;
  } finally {
    running = false;
    runKind = null;
    // After an abort the worker may still be reloading weights.
    el.run.disabled = !session.ready;
    el.random.disabled = el.run.disabled;
    el.run.textContent = "Generate";
    el.random.textContent = "Random shape";
    el.stop.hidden = true;
    el.progress.hidden = true;
  }
}

async function runGenerate(c, { random = false } = {}) {
  setProgress(0, 1, "Initialising…");
  let result;
  if (random) {
    // A fresh shape on every click; logged with the run so it can be repeated.
    const contextSeed = crypto.getRandomValues(new Uint32Array(1))[0];
    result = await session.random({ ...c, contextSeed }, (m) => setStatus(m));
    result.contextSeed = contextSeed;
  } else {
    result = await session.generate(c, (m) => setStatus(m));
  }

  if (result.aborted) {
    setStatus(
      session.loading
        ? "Aborted — the worker was restarted; reloading the weights…"
        : "Aborted — the worker was restarted.",
      "err",
    );
    el.empty.hidden = false;
    el.empty.textContent = "Run aborted.";
    return;
  }

  const { molecules, requested, elapsedMs } = result;

  const shape = random ? ` for random shape #${result.contextSeed}` : "";

  if (!molecules.length) {
    el.empty.hidden = false;
    el.empty.innerHTML =
      "No valid molecules survived standardization. Try more diffusion steps, or a different shape.";
    const hint = random ? "try another random shape" : "try adjusting the settings";
    setStatus(`0 valid molecules${shape} — ${hint}.`, "err");
    return;
  }

  molecules.forEach((mol, i) => {
    const { card } = makeCard(i, mol);
    el.grid.appendChild(card);
  });

  requestAnimationFrame(() => pool.resizeAll());

  const avgAtoms = (
    molecules.reduce((s, m) => s + m.nAtoms, 0) / molecules.length
  ).toFixed(1);

  showStats([
    [molecules.length, "valid"],
    [`${requested}`, "requested"],
    [avgAtoms, "avg atoms"],
    [`${(elapsedMs / 1000).toFixed(1)}s`, "elapsed"],
    [c.diffusionSteps, "steps"],
  ]);

  setStatus(
    `${molecules.length}/${requested} valid${shape} in ${(elapsedMs / 1000).toFixed(1)}s.`,
    "ok",
  );
}

async function runAnimate(c) {
  setProgress(0, c.diffusionSteps, "Initialising…");

  // One card per sample in the batch, mounted on the first frame and then fed
  // in place for the rest of the trajectory.
  const cards = [];
  const started = performance.now();

  // Frames arrive from the worker faster than they need to paint; keep only the
  // newest and render it once per animation frame, so a fast trajectory can't
  // queue up hundreds of redundant WebGL updates.
  //
  // requestAnimationFrame alone is not enough: it is suspended in a background
  // tab, so the cards would never mount and the run would look like it produced
  // nothing. Race it against a timer, which keeps firing when hidden.
  let queued = null;
  let scheduled = false;

  const paint = () => {
    scheduled = false;
    const mols = queued;
    if (!mols) return;
    queued = null;

    mols.forEach((mol, i) => {
      if (!cards[i]) {
        const made = makeCard(i, mol, {
          label: mols.length > 1 ? `TRAJECTORY ${i + 1}` : "DIFFUSION",
        });
        cards[i] = made;
        el.grid.appendChild(made.card);
        made.viewer.resize();
      } else {
        cards[i].viewer.updateMolecule(mol.molBlock);
        cards[i].card.querySelector(".card-meta").textContent =
          `${mol.nAtoms} atoms · ${mol.nBonds} bonds`;
      }
    });
  };

  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    let done = false;
    const once = () => {
      if (done) return;
      done = true;
      paint();
    };
    requestAnimationFrame(once);
    setTimeout(once, 250);
  };

  const result = await session.animate(c, {
    onStatus: (m) => setStatus(m),
    onFrame: ({ step, total, molecules }) => {
      queued = molecules;
      schedule();

      const per = (performance.now() - started) / step;
      const left = Math.round(((total - step) * per) / 1000);
      setProgress(
        step,
        total,
        `step ${step} / ${total}${left > 0 ? ` · ~${left}s left` : ""}`,
      );
    },
  });

  paint(); // ensure the final frame is shown even if a rAF was still pending

  const secs = (result.elapsedMs / 1000).toFixed(1);
  if (result.aborted) {
    setStatus(`Aborted after ${secs}s.`, "err");
    return;
  }

  const finals = result.molecules ?? [];
  const avgAtoms = finals.length
    ? (finals.reduce((s, m) => s + m.nAtoms, 0) / finals.length).toFixed(1)
    : "—";

  showStats([
    [finals.length || "—", "trajectories"],
    [result.total, "steps"],
    [avgAtoms, "avg atoms"],
    [`${secs}s`, "elapsed"],
  ]);
  setStatus(
    `${finals.length} ${finals.length === 1 ? "trajectory" : "trajectories"} complete — ${result.total} steps in ${secs}s.`,
    "ok",
  );
}

/* ── Wire-up ─────────────────────────────────────────────────────────── */

function init() {
  buildPresets();
  syncRangeLabels();
  updateCode();

  for (const [input] of RANGES) {
    $(input).addEventListener("input", () => {
      syncRangeLabels();
      refreshPresetState();
      updateCode();
    });
  }
  for (const input of el.moi) {
    input.addEventListener("input", () => {
      refreshPresetState();
      updateCode();
    });
  }
  $("c-seed").addEventListener("input", updateCode);
  $("c-mode").addEventListener("change", () => {
    updateCode();
    updateEstimate();
    el.run.textContent = "Generate";
  });

  renderSets();
  refreshCacheInfo();
  el.cacheClear.addEventListener("click", clearCache);
  el.srcHubBtn.addEventListener("click", () => showSource("hub"));
  el.srcLocalBtn.addEventListener("click", () => showSource("local"));
  el.hubLoad.addEventListener("click", () => {
    // The reload after an abort may already be under way.
    if (session.loading) return;
    resetDrops();
    loadHub();
  });
  for (const kind of KINDS) {
    const hint = `e.g. ${WEIGHT_SETS[DEFAULT_SET][kind].file}`;
    wireDrop(kind, `drop-${kind}`, `input-${kind}`, `file-${kind}`, hint);
  }

  el.run.addEventListener("click", () => run());
  el.random.addEventListener("click", () => run("random"));
  el.stop.addEventListener("click", () => {
    // Animate stops at the next step boundary; generate and random can only be
    // stopped by restarting the worker, which throws away the compiled graphs.
    const cooperative = runKind === "animate";
    setStatus(
      cooperative ? "Stopping after this step…" : "Restarting worker to stop…",
    );
    session.abort();
  });

  el.copyCode.addEventListener("click", () => {
    navigator.clipboard.writeText(el.code.textContent);
    el.copyCode.textContent = "Copied";
    setTimeout(() => (el.copyCode.textContent = "Copy snippet"), 1200);
  });

  refreshReady();
}

init();

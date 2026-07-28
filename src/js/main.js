import { GeneratorSession, readWeights } from "./generator.js";
import { CUSTOM, PRESETS } from "./presets.js";
import { ViewerPool } from "./viewer.js";

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
};

const session = new GeneratorSession();
const pool = new ViewerPool();
let activePreset = PRESETS[0];
let running = false;
let abort = false;

/* ── Config ──────────────────────────────────────────────────────────── */

function config() {
  const raw = $("c-mode").value;
  return {
    moi: el.moi.map((i) => Number(i.value) || 0),
    nAtoms: Number($("c-atoms").value),
    // The animation only ever displays one trajectory, so denoise a single
    // molecule there — batching would multiply the per-step cost for nothing.
    nSamples: raw === "generate" ? Number($("c-samples").value) : 1,
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
  for (const p of [...PRESETS, CUSTOM]) {
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
  if (p.moi) {
    el.moi.forEach((input, i) => (input.value = p.moi[i]));
    $("c-atoms").value = p.nAtoms;
    syncRangeLabels();
  }
  el.presetLabel.textContent = p.name;
  el.vMoi.textContent = p.id === "custom" ? "custom" : p.name.toLowerCase();
  for (const b of el.presets.children) {
    b.classList.toggle("on", b.dataset.id === p.id);
  }
  updateCode();
}

/** Editing the MOI fields by hand means we're no longer on a named preset. */
function markCustom() {
  if (activePreset.id === "custom") return;
  const p = activePreset;
  const drifted = p.moi?.some((v, i) => Number(el.moi[i].value) !== v);
  if (drifted) {
    activePreset = CUSTOM;
    el.presetLabel.textContent = CUSTOM.name;
    el.vMoi.textContent = "custom";
    for (const b of el.presets.children) {
      b.classList.toggle("on", b.dataset.id === "custom");
    }
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
  render(frame.molecules[0], frame.step, frame.total);
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
    label.textContent = `${file.name} · loading…`;
    try {
      const bytes = await readWeights(file);
      session.setWeights(kind, bytes);
      const mb = (bytes.byteLength / 1024 / 1024).toFixed(0);
      label.textContent = `${file.name} · ${mb} MB`;
      drop.classList.add("loaded");
      refreshReady();
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

function refreshReady() {
  if (session.ready) {
    el.run.disabled = false;
    setStatus("Weights loaded — ready to generate.", "ok");
  } else {
    el.run.disabled = true;
    const need = !session.egnnBytes ? "EGNN" : "AdjMatSeer";
    setStatus(`Waiting for ${need} weights…`);
  }
}

/* ── Rendering ───────────────────────────────────────────────────────── */

function molStats(mol) {
  return { atoms: mol.nAtoms, bonds: mol.bonds?.length ?? 0 };
}

function makeCard(index, mol, { label } = {}) {
  const card = document.createElement("article");
  card.className = "card";
  const { atoms, bonds } = molStats(mol);

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

  const block = mol.toMolBlock();
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

async function run() {
  if (running) return;
  running = true;
  abort = false;
  el.run.disabled = true;
  el.run.textContent = "Working…";
  el.stop.hidden = false;
  resetOutput();

  const c = config();

  try {
    if (c.mode === "animate") await runAnimate(c);
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
    el.run.disabled = false;
    el.run.textContent = "Generate";
    el.stop.hidden = true;
    el.progress.hidden = true;
  }
}

async function runGenerate(c) {
  setProgress(0, 1, "Initialising…");
  const { molecules, elapsedMs } = await session.generate(c, (m) =>
    setStatus(m),
  );

  if (!molecules.length) {
    el.empty.hidden = false;
    el.empty.innerHTML =
      "No valid molecules survived standardization. Try more diffusion steps, or a different shape.";
    setStatus("0 valid molecules — try adjusting the settings.", "err");
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
    [`${c.nSamples}`, "requested"],
    [avgAtoms, "avg atoms"],
    [`${(elapsedMs / 1000).toFixed(1)}s`, "elapsed"],
    [c.diffusionSteps, "steps"],
  ]);

  setStatus(
    `${molecules.length}/${c.nSamples} valid in ${(elapsedMs / 1000).toFixed(1)}s.`,
    "ok",
  );
}

async function runAnimate(c) {
  setProgress(0, c.diffusionSteps, "Initialising…");

  let card = null;
  let viewer = null;
  const started = performance.now();
  let lastFrame = null;
  let announced = false;

  for await (const frame of session.animate(c, (m) => setStatus(m))) {
    if (!announced) {
      setStatus("Streaming denoising trajectory…");
      announced = true;
    }
    if (abort) {
      setStatus("Aborted.", "err");
      return;
    }
    const mol = frame.molecules[0];
    if (!mol) continue;
    lastFrame = frame;

    if (!card) {
      const made = makeCard(0, mol, { label: "DIFFUSION" });
      card = made.card;
      viewer = made.viewer;
      el.grid.appendChild(card);
      requestAnimationFrame(() => viewer.resize());
    } else {
      viewer.updateMolecule(mol.toMolBlock());
      card.querySelector(".card-meta").textContent =
        `${mol.nAtoms} atoms · ${mol.bonds?.length ?? 0} bonds`;
    }

    const per = (performance.now() - started) / frame.step;
    const left = Math.round(((frame.total - frame.step) * per) / 1000);
    setProgress(
      frame.step,
      frame.total,
      `step ${frame.step} / ${frame.total}${left > 0 ? ` · ~${left}s left` : ""}`,
    );
    // Yield so the frame paints. setTimeout rather than requestAnimationFrame:
    // rAF is throttled (or never fires) in a background tab, which would stall
    // the whole trajectory instead of just slowing the animation down.
    await new Promise((r) => setTimeout(r, 0));
  }

  const secs = ((performance.now() - started) / 1000).toFixed(1);
  showStats([
    [lastFrame?.total ?? c.diffusionSteps, "steps"],
    [lastFrame?.molecules[0]?.nAtoms ?? "—", "atoms"],
    [`${secs}s`, "elapsed"],
  ]);
  setStatus(`Trajectory complete — ${lastFrame?.total ?? 0} steps in ${secs}s.`, "ok");
}

/* ── Wire-up ─────────────────────────────────────────────────────────── */

function init() {
  buildPresets();
  syncRangeLabels();
  updateCode();

  for (const [input] of RANGES) {
    $(input).addEventListener("input", () => {
      syncRangeLabels();
      updateCode();
    });
  }
  for (const input of el.moi) {
    input.addEventListener("input", () => {
      markCustom();
      updateCode();
    });
  }
  $("c-seed").addEventListener("input", updateCode);
  $("c-mode").addEventListener("change", () => {
    updateCode();
    el.run.textContent = "Generate";
  });

  wireDrop("egnn", "drop-egnn", "input-egnn", "file-egnn", "egnn_chembl_15_39.onnx");
  wireDrop("adj", "drop-adj", "input-adj", "file-adj", "adj_mat_seer_chembl_15_39.onnx");

  el.run.addEventListener("click", run);
  el.stop.addEventListener("click", () => {
    abort = true;
    setStatus("Aborting after current step…");
  });

  el.copyCode.addEventListener("click", () => {
    navigator.clipboard.writeText(el.code.textContent);
    el.copyCode.textContent = "Copied";
    setTimeout(() => (el.copyCode.textContent = "Copy snippet"), 1200);
  });

  refreshReady();
}

init();

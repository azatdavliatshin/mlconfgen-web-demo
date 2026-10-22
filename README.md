# mlconfgen web demo

A browser playground for [`mlconfgen`](https://www.npmjs.com/package/mlconfgen) —
spatially-aware molecule generation via equivariant diffusion, running **entirely
client-side**. No server, no upload, no inference API.

The page loads the model weights from Hugging Face, runs the real npm package against
them through ONNX Runtime Web, and renders the results in an interactive 3D viewer.

## Using the demo

1. Pick a weight set and click **Load**. The first time, the two ONNX files are
   downloaded from [Membrizard/ml_conformer_generator](https://huggingface.co/Membrizard/ml_conformer_generator)
   on Hugging Face, with a progress bar per file. They are kept in the browser's cache,
   so later visits load them without downloading.
2. Pick a shape preset, adjust the controls, and hit **Generate** — or **Random shape**
   to generate for a molecule shape drawn from a ChEMBL sample bundled with the package.

### Weight sets

| Set | Files | Download | Notes |
|---|---|---|---|
| **Distilled** (default) | `small_egnn_chembl_15_39.onnx`, `small_adj_mat_seer_obabel_15_39.onnx` | 45 MB | About 2.6× faster per step. Its bond predictor is fine-tuned to match OpenBabel, so more samples come out valid (about 70% against 50% in the upstream benchmark). |
| **Full** | `egnn_chembl_15_39.onnx`, `adj_mat_seer_chembl_15_39.onnx` | 185 MB | The core model the paper reports on: slightly closer shape matches, four times the download, slower. |

The two files of a set belong together; the page never mixes sets.

### Local files

**Local files** loads a matching pair of ONNX files from your disk instead — to work
offline, or to try your own fine-tuned weights. The files are read with the `File` API
into a `Uint8Array` and handed straight to the package. Nothing leaves the tab.

### The browser cache

Downloads are stored with the Cache API (`mlconfgen-weights`), which needs HTTPS or
`localhost`. The page shows what is cached and has a **Clear cached weights** button.
Browsers may evict the cache under storage pressure, and may refuse to store a file at
all (private windows, a full disk); the page says so, and the weights are simply
downloaded again next time. Weights already loaded stay usable after clearing until
the page is reloaded.

## What it demonstrates

- **Shape-conditioned generation** — the model is conditioned on a triplet of
  moment-of-inertia eigenvalues describing a target shape envelope. Presets cover
  several ligand geometries; you can also enter the values by hand.
- **The package API** — `createGenerator()` resolving weights by name (download, cache,
  progress), `generateConformers()` for a batch of final molecules, `random()` for a
  random shape, and `animateGeneration()` streaming one frame per denoising step so you
  can watch molecules condense out of noise.

  Animation is a single batched reverse-diffusion pass: every sample rides in
  one tensor whose first dimension is `nSamples`, so all molecules advance the
  same step together. Their trajectories are still chemically independent —
  each starts from its own noise, and `edge_mask` blocks message passing
  between molecules — they are simply marched in lockstep because one batched
  forward pass is far cheaper than N separate ones. The synchronized display is
  therefore accurate, not a simplification.
- **Live 3D** — every result gets an interactive [3Dmol.js](https://3dmol.csb.pitt.edu/)
  viewer; molfiles can be copied or downloaded.
- **A matching code snippet** — panel 04 mirrors your current weights, settings and
  last action as real package code.

## Running locally

```bash
npm install
npm run dev
```

Then open the printed URL. Production build:

```bash
npm run build && npm run preview
```

`npm run preview` serves the build without the dev server's cross-origin isolation
headers, the same as GitHub Pages.

## Deploying

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds and
publishes to GitHub Pages. Enable Pages once under **Settings → Pages → Source:
GitHub Actions**.

The workflow sets `BASE_PATH` from the repository name so assets resolve under
`https://<user>.github.io/<repo>/`. For a user/org site served from the domain
root, build with `BASE_PATH=/`.

## Notes

- **Runtime assets are self-hosted.** The ONNX Runtime and RDKit WASM binaries
  are served from `public/`, so the deployed page has no CDN dependency. This
  also pins RDKit's `locateFile`, which otherwise resolves incorrectly under a
  GitHub Pages project subdirectory. The model weights are the one thing fetched
  from elsewhere — from Hugging Face, never re-hosted here.

  The binaries are copied out of `node_modules` by `scripts/sync-runtime-assets.js`,
  which runs automatically on `postinstall`, `dev`, and `build` — so ~32 MB of
  binaries stay out of git. `public/ort/` and `public/rdkit/` are gitignored;
  run `npm install` (or `npm run sync-assets`) to populate them.
- **Speed.** Generation is CPU-bound single-threaded WASM inference. Measured on an
  Apple M1 in Chrome, Generate costs about **0.15 s per denoising step per sample
  with Distilled** and **0.39 s with Full** for a 20-atom shape, growing roughly
  with the square of the atom count (a 38-atom shape costs about 3.3× as much).
  The page defaults (1 sample × 20 steps) take about 3 s with Distilled and 8 s
  with Full. Animate takes about twice as long as Generate, because every frame is
  decoded and sent to the page; "live bonds" adds a bond prediction per frame, which
  costs only a few percent more. The page shows a live runtime estimate for the
  loaded weight set as you move the sliders.

  Multi-threading would need `SharedArrayBuffer` and thus cross-origin isolation,
  which GitHub Pages does not provide. Measured locally *with* isolation, four
  threads made session creation slower rather than faster, so the demo pins
  `numThreads = 1`.
- **Inference runs in a Web Worker**, so the page stays interactive while the
  model runs. Doing this work on the main thread freezes the tab outright and
  Chrome reports "Page Unresponsive". The worker downloads the weights itself and
  keeps them, so changing the number of diffusion steps rebuilds the model without
  touching the network.
- **Requires `mlconfgen` ≥ 0.2.0**, the first release that resolves weights by name
  from Hugging Face, caches them in the browser and provides `random()`.

## Credits

Model and package by [Membrizard](https://github.com/Membrizard/ml_conformer_generator).
If you use MLConfGen in research, please cite:

> Sapegin, Bakharev, Krupenya, Gafurov, Pildish & Bear.
> *Moment of inertia as a simple shape descriptor for diffusion-based
> shape-constrained molecular generation.* Digital Discovery, 2025.
> [10.1039/D5DD00318K](https://doi.org/10.1039/D5DD00318K)

Demo code Apache-2.0. Model weights Apache-2.0, downloaded from
[Hugging Face](https://huggingface.co/Membrizard/ml_conformer_generator) at runtime
and not included in this repository.

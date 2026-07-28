# mlconfgen web demo

A browser playground for [`mlconfgen`](https://www.npmjs.com/package/mlconfgen) —
spatially-aware molecule generation via equivariant diffusion, running **entirely
client-side**. No server, no upload, no inference API.

You supply the model weights from your own machine; the demo runs the real npm
package against them through ONNX Runtime Web and renders the results in an
interactive 3D viewer.

## Why you bring your own weights

The trained weights are licensed **CC BY-NC-ND 4.0** and are deliberately not
redistributed here — they are not on npm, not committed to this repo, and not
re-hosted by the deployed site. That is not a limitation to work around; it is
how the model is meant to be used.

In practice it makes for a better demo. You download the weights once from
Hugging Face, drop them into the page, and everything from then on happens on
your own hardware.

## Using the demo

1. Open the [model repo on Hugging Face](https://huggingface.co/Membrizard/ml_conformer_generator)
   and accept the license terms to enable downloads.
2. Download both ONNX files from the *Files* tab:
   - `egnn_chembl_15_39.onnx` (~94 MB) — the diffusion model
   - `adj_mat_seer_chembl_15_39.onnx` (~83 MB) — the bond predictor
3. Drag each file onto its slot in the page.
4. Pick a shape preset, adjust the controls, and hit **Generate**.

The files are read with the `File` API into a `Uint8Array` and handed straight to
the package. Nothing leaves the tab.

## What it demonstrates

- **Shape-conditioned generation** — the model is conditioned on a triplet of
  moment-of-inertia eigenvalues describing a target shape envelope. Presets cover
  several ligand geometries; you can also enter the values by hand.
- **Both API paths** — `generateConformers()` for a batch of final molecules, and
  `animateGeneration()` streaming one frame per denoising step so you can watch a
  molecule condense out of noise.
- **Live 3D** — every result gets an interactive [3Dmol.js](https://3dmol.csb.pitt.edu/)
  viewer; molfiles can be copied or downloaded.
- **A matching code snippet** — panel 04 mirrors your current settings as real
  package code.

## Running locally

```bash
npm install
npm run dev
```

Then open the printed URL. Production build:

```bash
npm run build && npm run preview
```

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
  GitHub Pages project subdirectory.

  They are copied out of `node_modules` by `scripts/sync-runtime-assets.js`,
  which runs automatically on `postinstall`, `dev`, and `build` — so ~32 MB of
  binaries stay out of git. `public/ort/` and `public/rdkit/` are gitignored;
  run `npm install` (or `npm run sync-assets`) to populate them.
- **Speed.** Generation is CPU-bound single-threaded WASM inference, and it is
  not fast: measured locally at roughly **2.15 s per denoising step per sample**
  (1 sample x 10 steps ran in 19 s warm, 23.5 s cold). The defaults
  (2 samples × 15 steps) take about a minute — a measured run came in at 63.6 s
  against a 65 s estimate. The first run adds a one-off ~4 s ONNX graph
  compilation. The page shows a live runtime estimate as you move the sliders,
  and the slider ranges are capped so the worst case stays around ten minutes.

  Multi-threading would need `SharedArrayBuffer` and thus cross-origin isolation,
  which GitHub Pages does not provide. Measured locally *with* isolation, four
  threads made session creation slower rather than faster, so the demo pins
  `numThreads = 1`.

- **Inference runs in a Web Worker**, so the page stays interactive while the
  model runs. Doing this work on the main thread freezes the tab outright and
  Chrome reports "Page Unresponsive".
- **Requires `mlconfgen` ≥ 0.1.1**, which is the first release whose RDKit loader
  resolves correctly in a browser.

## Credits

Model and package by [Membrizard](https://github.com/Membrizard/ml_conformer_generator).
If you use MLConfGen in research, please cite:

> Sapegin, Bakharev, Krupenya, Gafurov, Pildish & Bear.
> *Moment of inertia as a simple shape descriptor for diffusion-based
> shape-constrained molecular generation.* Digital Discovery, 2025.
> [10.1039/D5DD00318K](https://doi.org/10.1039/D5DD00318K)

Demo code Apache-2.0. Model weights CC BY-NC-ND 4.0 and not included.

import { createViewer } from "3dmol";

/** Neon element palette, tuned to read against the dark retrowave backdrop. */
const ELEMENT_COLORS = {
  C: 0x8fe9ff,
  N: 0x4d7cff,
  O: 0xff2d95,
  F: 0x7cf76f,
  P: 0xffb547,
  S: 0xffe14d,
  Cl: 0x5cff8f,
  Br: 0xc77b4a,
  H: 0xd8ccff,
};

const STYLE = {
  stick: { radius: 0.13, colorscheme: { prop: "elem", map: ELEMENT_COLORS } },
  sphere: { scale: 0.24, colorscheme: { prop: "elem", map: ELEMENT_COLORS } },
};

/** 3Dmol needs a terminated SDF record. */
function asSdf(molBlock) {
  const body = String(molBlock).replace(/\$\$\$\$\s*$/, "").trimEnd();
  return `${body}\n$$$$\n`;
}

/**
 * An interactive 3D molecule view bound to one container element.
 *
 * `setMolecule` replaces the model outright, which is what the animation path
 * wants — one instance per card, re-fed each frame, rather than a new viewer
 * (and a new WebGL context) per update.
 */
export class MoleculeViewer {
  constructor(element, { spin = false } = {}) {
    this.element = element;
    this.viewer = createViewer(element, {
      backgroundColor: "#05010f",
      antialias: true,
    });
    this.spin = spin;
    this.hasModel = false;
  }

  /** @param {string} molBlock V2000 molblock */
  setMolecule(molBlock, { resetView = true } = {}) {
    const viewer = this.viewer;
    viewer.removeAllModels();
    viewer.addModel(asSdf(molBlock), "sdf");
    viewer.setStyle({}, STYLE);
    if (resetView) viewer.zoomTo();
    viewer.render();
    if (this.spin && !this.hasModel) viewer.spin("y", 0.6);
    this.hasModel = true;
  }

  /**
   * Swap coordinates without resetting the camera — used per animation frame so
   * the molecule appears to condense in place instead of jumping around.
   */
  updateMolecule(molBlock) {
    this.setMolecule(molBlock, { resetView: false });
  }

  resize() {
    try {
      this.viewer.resize();
      this.viewer.render();
    } catch {
      // Viewer may be detached mid-teardown.
    }
  }

  destroy() {
    try {
      this.viewer.spin(false);
      this.viewer.removeAllModels();
      this.viewer.clear();
    } catch {
      // ignore
    }
  }
}

/** Tracks viewers so a new run can tear down the previous batch. */
export class ViewerPool {
  constructor() {
    this.viewers = [];
    this._onResize = () => this.resizeAll();
    window.addEventListener("resize", this._onResize);
  }

  create(element, options) {
    const v = new MoleculeViewer(element, options);
    this.viewers.push(v);
    return v;
  }

  resizeAll() {
    for (const v of this.viewers) v.resize();
  }

  clear() {
    for (const v of this.viewers) v.destroy();
    this.viewers.length = 0;
  }
}

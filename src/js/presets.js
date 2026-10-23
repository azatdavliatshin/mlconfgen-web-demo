/**
 * Shape presets — moment-of-inertia eigenvalue triplets.
 *
 * The EDM is conditioned on the three principal moments of inertia of a
 * reference conformer, which together describe how mass is distributed along
 * each principal axis. Small, near-equal values read as compact and globular;
 * one large value against two small ones reads as elongated.
 *
 * Each preset also carries the atom count and variance for that shape (steps
 * 20, samples 1, seed 42). With the Distilled weights these give valid
 * molecules; with Full the same settings can give none — retuning per weight
 * set is deferred.
 */
export const PRESETS = [
  {
    id: "flat_disc",
    name: "Flat Disc",
    moi: [101.40107727050781, 102.84884643554688, 201.9473876953125],
    nAtoms: 20,
    variance: 1,
    note: "Planar envelope — two equal light axes, one much heavier.",
  },
  {
    id: "elongated_rod",
    name: "Elongated Rod",
    moi: [101.6028060913086, 1135.322509765625, 1167.136962890625],
    nAtoms: 38,
    variance: 0,
    note: "Linear, extended — one light axis, two heavy ones.",
  },
  {
    id: "v_shaped",
    name: "V-Shaped",
    moi: [89.87, 210.78, 217.78],
    nAtoms: 23,
    variance: 0,
    note: "Bent scaffold — the package's reference triplet.",
  },
];

export const CUSTOM = {
  id: "custom",
  name: "Custom",
  note: "Your own moment-of-inertia triplet.",
};

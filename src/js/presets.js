/**
 * Shape presets — moment-of-inertia eigenvalue triplets.
 *
 * The EDM is conditioned on the three principal moments of inertia of a
 * reference conformer, which together describe how mass is distributed along
 * each principal axis. Small, near-equal values read as compact and globular;
 * one large value against two small ones reads as elongated.
 *
 * The "kinase-like" triplet is the reference used in the package's own example;
 * the others are scaled to sketch out distinct regions of the same space.
 */
export const PRESETS = [
  {
    id: "kinase",
    name: "Kinase-like",
    moi: [89.87, 210.78, 217.78],
    nAtoms: 20,
    note: "Flat, fused-ring scaffold — the package's reference shape.",
  },
  {
    id: "globular",
    name: "Globular",
    moi: [140.0, 152.0, 165.0],
    nAtoms: 22,
    note: "Near-spherical: three similar moments, densely packed.",
  },
  {
    id: "rod",
    name: "Elongated",
    moi: [42.0, 330.0, 345.0],
    nAtoms: 24,
    note: "Linear, extended — one light axis, two heavy ones.",
  },
  {
    id: "disc",
    name: "Flat Disc",
    moi: [175.0, 180.0, 340.0],
    nAtoms: 21,
    note: "Planar aromatic sheet — two equal axes, one much heavier.",
  },
  {
    id: "compact",
    name: "Fragment",
    moi: [55.0, 95.0, 110.0],
    nAtoms: 16,
    note: "Small fragment-sized envelope for lead-like starting points.",
  },
  {
    id: "large",
    name: "Macrocycle",
    moi: [260.0, 420.0, 470.0],
    nAtoms: 34,
    note: "Large, sprawling scaffold approaching the 39-atom ceiling.",
  },
];

export const CUSTOM = {
  id: "custom",
  name: "Custom",
  note: "Your own moment-of-inertia triplet.",
};

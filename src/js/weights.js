/**
 * Weight sets published in the Hugging Face repo.
 *
 * Files are named the way mlconfgen expects bare names: the package downloads
 * them from Membrizard/ml_conformer_generator on first use and keeps them in
 * the browser's Cache API afterwards. Sizes are the published byte counts,
 * shown before anything is downloaded.
 *
 * The two files of a set belong together — the distilled AdjMatSeer is trained
 * against the distilled EGNN — so sets are never mixed.
 */
export const WEIGHT_SETS = {
  distilled: {
    id: "distilled",
    name: "Distilled",
    egnn: { file: "small_egnn_chembl_15_39.onnx", bytes: 38_100_052 },
    adj: { file: "small_adj_mat_seer_obabel_15_39.onnx", bytes: 6_494_943 },
  },
  full: {
    id: "full",
    name: "Full",
    egnn: { file: "egnn_chembl_15_39.onnx", bytes: 98_173_589 },
    adj: { file: "adj_mat_seer_chembl_15_39.onnx", bytes: 87_288_156 },
  },
};

export const DEFAULT_SET = "distilled";

export const KINDS = ["egnn", "adj"];

/** Total download size of a set in bytes. */
export function setBytes(set) {
  return set.egnn.bytes + set.adj.bytes;
}

/** Megabytes for display, without pretending to more precision than a size label needs. */
export function mb(bytes) {
  const v = bytes / 1e6;
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

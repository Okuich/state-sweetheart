/**
 * Material Metric Intelligence — public surface.
 */
export { BUILTIN_MATERIALS } from "./library";
export {
  buildIndex, recommend, similar, substitute,
  type CorpusIndex,
} from "./engine";
export {
  rawFeatures, embed, fitScaler, l2, cosine, FEATURE_DIM,
  type FeatureScaler,
} from "./embeddings";
export {
  DEFAULT_WEIGHTS,
  type MaterialRecord, type MaterialFamily, type Fabrication,
  type Environment, type DesignConstraints, type ObjectiveWeights,
  type MaterialScore,
} from "./types";

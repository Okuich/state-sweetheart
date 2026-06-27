/** Physics OS — Neural Operator Co-Simulation Engine (PIKAN-powered). */
export * from "./types";
export {
  defaultConfig, initShockScenario, initState,
  staggeredStep, decoupledStep, coupledStep,
  refineCouplingKAN, buildCouplingKAN,
  benchmark, coupledL2, fieldL2, couplingResidual,
} from "./engine";
export { makeKAN, kanForward, kanRefine, type KAN, type KANLayer } from "./pikan";

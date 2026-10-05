import type { Catalog, ExperimentConfig, GameSpec } from "./types";

export function initialConfig(game: GameSpec): ExperimentConfig {
  return {
    game_name: game.name,
    algorithm_names: [game.algorithms.includes("Hedge") ? "Hedge" : game.algorithms[0]],
    iterations: 1000, runs: 3, seed: 42,
    learning_rate: 0.2, decay: -0.5, temperature: 0.1, exploration: 0.1, initialization: "random",
  };
}

export function validateConfig(config: ExperimentConfig, catalog: Catalog): string | null {
  const game = catalog.games.find((entry) => entry.name === config.game_name);
  if (!game) return "Choose a supported game.";
  if (!config.algorithm_names.length) return "Choose at least one algorithm.";
  if (config.algorithm_names.some((name) => !game.algorithms.includes(name))) return "Choose algorithms supported by this game.";
  if (!Number.isInteger(config.iterations) || config.iterations < 100 || config.iterations > catalog.limits.max_iterations) return `Iterations must be a whole number from 100 to ${catalog.limits.max_iterations.toLocaleString()}.`;
  if (!Number.isInteger(config.runs) || config.runs < 1 || config.runs > catalog.limits.max_runs) return `Repeats must be a whole number from 1 to ${catalog.limits.max_runs}.`;
  if (config.iterations * config.runs * config.algorithm_names.length > catalog.limits.max_steps) return `Use at most ${catalog.limits.max_steps.toLocaleString()} steps across algorithms and repeats.`;
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295) return "Seed must be a whole number from 0 to 4,294,967,295.";
  if (!Number.isFinite(config.learning_rate) || config.learning_rate <= 0) return "Learning rate must be positive.";
  if (!Number.isFinite(config.temperature) || config.temperature <= 0) return "Temperature must be positive.";
  if (!Number.isFinite(config.decay)) return "Decay must be a finite number.";
  if (!Number.isFinite(config.exploration) || config.exploration < 0 || config.exploration > 1) return "Exploration must be between 0 and 1.";
  return null;
}

export function configChanged(config: ExperimentConfig, completed: ExperimentConfig): boolean {
  return JSON.stringify(config) !== JSON.stringify(completed);
}

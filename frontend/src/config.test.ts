import { describe, expect, it } from "vitest";
import { configChanged, initialConfig, validateConfig } from "./config";
import type { Catalog, ExperimentConfig, GameSpec } from "./types";

const game: GameSpec = { name: "Example", category: "Box Games", num_players: 3, action_labels: ["A", "B"], description: "Example", algorithms: ["Hedge", "EXP3"], equilibria: [], payoffs: [] };
const catalog: Catalog = { games: [game], algorithms: [], limits: { max_steps: 200000, max_runs: 5, max_iterations: 100000 } };
const config = initialConfig(game);

describe("experiment configuration", () => {
  it("uses deterministic defaults and a compatible algorithm", () => {
    expect(config).toMatchObject({ algorithm_names: ["Hedge"], seed: 42, runs: 3 });
    expect(initialConfig({ ...game, algorithms: ["EXP3"] }).algorithm_names).toEqual(["EXP3"]);
    expect(validateConfig(config, catalog)).toBeNull();
  });
  it.each<[Partial<ExperimentConfig>, string]>([
    [{ game_name: "missing" }, "supported game"],
    [{ algorithm_names: [] }, "at least one"],
    [{ algorithm_names: ["missing"] }, "supported by"],
    [{ iterations: 0 }, "Iterations"],
    [{ iterations: 100001 }, "Iterations"],
    [{ iterations: 100.5 }, "Iterations"],
    [{ runs: 0 }, "Repeats"],
    [{ runs: 6 }, "Repeats"],
    [{ runs: 1.1 }, "Repeats"],
    [{ iterations: 100000 }, "200,000"],
    [{ seed: -1 }, "Seed"],
    [{ seed: 1.1 }, "Seed"],
    [{ seed: 4294967296 }, "Seed"],
    [{ learning_rate: 0 }, "Learning rate"],
    [{ learning_rate: Number.NaN }, "Learning rate"],
    [{ temperature: 0 }, "Temperature"],
    [{ temperature: Infinity }, "Temperature"],
    [{ decay: Infinity }, "Decay"],
    [{ exploration: -0.1 }, "Exploration"],
    [{ exploration: 1.1 }, "Exploration"],
    [{ exploration: Number.NaN }, "Exploration"],
  ])("rejects invalid settings %#", (patch, expected) => {
    expect(validateConfig({ ...config, ...patch }, catalog)).toContain(expected);
  });
  it("detects edits without mutating the completed setup", () => {
    const edited = { ...config, seed: 43 };
    expect(configChanged(edited, config)).toBe(true);
    expect(configChanged({ ...config }, config)).toBe(false);
    expect(config.seed).toBe(42);
  });
});

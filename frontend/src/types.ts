export interface ExperimentConfig {
  game_name: string;
  algorithm_names: string[];
  iterations: number;
  runs: number;
  seed: number;
  learning_rate: number;
  decay: number;
  temperature: number;
  exploration: number;
  initialization: "random" | "uniform" | "biased";
}

export interface GameSpec {
  name: string;
  category: string;
  num_players: number;
  action_labels: string[];
  description: string;
  algorithms: string[];
  equilibria: number[][][];
  payoffs: { actions: number[]; values: number[] }[];
}

export interface AlgorithmSpec {
  name: string;
  description: string;
  feedback: string;
}

export interface Catalog {
  games: GameSpec[];
  algorithms: AlgorithmSpec[];
  limits: { max_steps: number; max_runs: number; max_iterations: number };
}

export interface ExperimentResult {
  id: string;
  config: ExperimentConfig;
  summary: {
    run_count: number;
    runtime_seconds: number;
    final_nash_gap: number;
    expected_regret: number;
    mean_payoffs: number[];
  };
  empirical_available: boolean;
  comparison: ComparisonRow[];
}

export interface ComparisonRow {
  algorithm: string;
  run: number;
  seed: string;
  nash_gap_final: number;
  nash_gap_mean: number;
  [metric: `payoff_mean_p${number}` | `payoff_final_p${number}` | `average_expected_regret_final_p${number}`]: number;
}

export type FigureKind = "trajectory" | "diagnostics" | "strategies";
export type PolicyView = "Current policy" | "Time-average policy" | "Empirical frequencies";
export type DiagnosticMetric = "nash_gap" | "average_regret" | "payoffs";

export interface PlotlyFigure {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
  frames?: { name: string; data: Record<string, unknown>[]; traces?: number[] }[];
}

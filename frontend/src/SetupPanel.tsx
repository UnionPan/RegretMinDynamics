import type { Catalog, ExperimentConfig, GameSpec } from "./types";

interface Props {
  catalog: Catalog;
  config: ExperimentConfig;
  running: boolean;
  changed: boolean;
  onChange: (config: ExperimentConfig) => void;
  onRun: () => void;
}

export default function SetupPanel({ catalog, config, running, changed, onChange, onRun }: Props) {
  const game = catalog.games.find((entry) => entry.name === config.game_name)!;
  const games = catalog.games.filter((entry) => entry.category === game.category);
  const update = <Key extends keyof ExperimentConfig>(key: Key, value: ExperimentConfig[Key]) => onChange({ ...config, [key]: value });
  const selectGame = (selected: GameSpec) => onChange({
    ...config, game_name: selected.name,
    algorithm_names: config.algorithm_names.filter((name) => selected.algorithms.includes(name)).length
      ? config.algorithm_names.filter((name) => selected.algorithms.includes(name))
      : [selected.algorithms.includes("Hedge") ? "Hedge" : selected.algorithms[0]],
  });
  const toggleAlgorithm = (name: string) => update("algorithm_names", config.algorithm_names.includes(name)
    ? config.algorithm_names.filter((entry) => entry !== name)
    : game.algorithms.filter((entry) => config.algorithm_names.includes(entry) || entry === name));
  return (
    <aside className="setup-rail" aria-label="Experiment setup">
      <form className="setup-form" onSubmit={(event) => { event.preventDefault(); onRun(); }} noValidate>
        <div className="rail-heading"><span className="eyebrow">SETUP</span>{changed && <span className="setup-changed">Setup changed</span>}</div>
        <fieldset className="field family-field"><legend className="field-label">Game family</legend>
          <div className="segmented">
            {[{ category: "Box Games", label: "3 players" }, { category: "RPS Games", label: "2 players" }].map(({ category, label }) => (
              <button type="button" className="segment" aria-pressed={game.category === category} key={category} onClick={() => { const selected = catalog.games.find((entry) => entry.category === category); if (selected) selectGame(selected); }}>{label}</button>
            ))}
          </div>
        </fieldset>
        <label className="field" htmlFor="game"><span className="field-label">Game</span>
          <select id="game" value={config.game_name} onChange={(event) => selectGame(catalog.games.find((entry) => entry.name === event.target.value)!)}>{games.map((entry) => <option key={entry.name}>{entry.name}</option>)}</select>
        </label>
        <div className="field"><span className="field-label">Algorithms</span>
          <details className="algorithm-picker"><summary>{config.algorithm_names.length === 1 ? config.algorithm_names[0] : `${config.algorithm_names.length} selected`}</summary>
            <div className="algorithm-chips" role="group" aria-label="Algorithms">
              {game.algorithms.map((name) => <button type="button" className="algorithm-chip" key={name} aria-pressed={config.algorithm_names.includes(name)} onClick={() => toggleAlgorithm(name)}>{name}</button>)}
            </div>
          </details>
        </div>
        <label className="field" htmlFor="iterations"><span className="field-label">Iterations</span>
          <input id="iterations" type="number" min={100} max={catalog.limits.max_iterations} step={100} value={Number.isNaN(config.iterations) ? "" : config.iterations} onChange={(event) => update("iterations", event.target.valueAsNumber)} />
        </label>
        <details className="advanced-settings"><summary>Advanced settings</summary>
          <div className="settings-grid">
            <label className="field" htmlFor="runs"><span className="field-label">Repeats</span><input id="runs" type="number" min={1} max={catalog.limits.max_runs} value={Number.isNaN(config.runs) ? "" : config.runs} onChange={(event) => update("runs", event.target.valueAsNumber)} /></label>
            <label className="field" htmlFor="seed"><span className="field-label">Seed</span><input id="seed" type="number" min={0} max={4294967295} value={Number.isNaN(config.seed) ? "" : config.seed} onChange={(event) => update("seed", event.target.valueAsNumber)} /></label>
            <label className="field settings-wide" htmlFor="initialization"><span className="field-label">Initialization</span><select id="initialization" value={config.initialization} onChange={(event) => update("initialization", event.target.value as ExperimentConfig["initialization"])}><option value="random">Random</option><option value="uniform">Uniform</option><option value="biased">Biased</option></select></label>
            {([{ key: "learning_rate", label: "Learning rate", min: 0.001, step: 0.05 }, { key: "decay", label: "Decay", step: 0.1 }, { key: "temperature", label: "Temperature", min: 0.001, step: 0.05 }, { key: "exploration", label: "Exploration", min: 0, max: 1, step: 0.05 }] as const).map(({ key, label, ...attributes }) => <label className="field" htmlFor={key} key={key}><span className="field-label">{label}</span><input id={key} type="number" {...attributes} value={Number.isNaN(config[key]) ? "" : config[key]} onChange={(event) => update(key, event.target.valueAsNumber)} /></label>)}
          </div>
        </details>
        <button type="submit" className="run-button" disabled={running}>{running ? "Running..." : "Run experiment"}<span aria-hidden="true">↗</span></button>
      </form>
      <div className="rail-footnote"><span className="rail-dot" aria-hidden="true" />Seeded. Reproducible.</div>
    </aside>
  );
}

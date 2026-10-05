import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { downloadExperiment, errorMessage, getCatalog, runExperiment } from "./api";
import { configChanged, initialConfig, validateConfig } from "./config";
import InfoDrawer from "./InfoDrawer";
import PlotView from "./PlotView";
import SetupPanel from "./SetupPanel";
import type { Catalog, DiagnosticMetric, ExperimentConfig, ExperimentResult, FigureKind, PolicyView } from "./types";

const tabs = [{ value: "trajectory", label: "Trajectory" }, { value: "diagnostics", label: "Diagnostics" }, { value: "strategies", label: "Final policies" }] as const;

function EmptyState() {
  return <div className="empty-state">
    <svg className="empty-geometry" viewBox="0 0 430 260" fill="none" aria-hidden="true">
      <path d="M99 204L215 30L331 204H99Z" stroke="currentColor" strokeWidth="1" opacity=".2" />
      <path d="M99 204L157 117M215 30L273 117M331 204H215" stroke="currentColor" strokeWidth="1" opacity=".08" />
      <path d="M117 191C159 149 311 201 261 115C232 65 166 109 193 159C217 203 279 162 256 135C236 111 212 132 224 148" stroke="#173e38" strokeWidth="1.7" />
      <circle cx="117" cy="191" r="3.5" fill="#173e38" opacity=".35" /><circle cx="224" cy="148" r="4.5" fill="#173e38" />
      <circle cx="215" cy="30" r="3" fill="currentColor" opacity=".25" /><circle cx="99" cy="204" r="3" fill="currentColor" opacity=".25" /><circle cx="331" cy="204" r="3" fill="currentColor" opacity=".25" />
    </svg>
    <h2>A game. A rule. A trajectory.</h2><p>Choose your setup, then run an experiment.</p>
  </div>;
}

function Metrics({ result }: { result: ExperimentResult }) {
  const metricValue = (value: number) => value !== 0 && Math.abs(value) < 0.001 ? value.toExponential(1) : value.toFixed(3);
  const entries = [
    { label: "Nash gap", value: metricValue(result.summary.final_nash_gap), definition: "Final sum of unilateral expected utility gains, averaged across runs. Zero indicates a Nash equilibrium." },
    { label: "Expected regret", value: metricValue(result.summary.expected_regret), definition: "Final average expected external regret, averaged across players and runs. This is not realized bandit regret." },
    { label: "Runtime", value: `${result.summary.runtime_seconds.toFixed(2)}s`, definition: "Server simulation runtime. Figure rendering and downloads are excluded." },
  ];
  return <div className="metrics-strip">{entries.map((entry) => <div className="metric-item" key={entry.label}><span className="metric-label" title={entry.definition}>{entry.label}<span className="metric-help" aria-label={entry.definition}>ⓘ</span></span><strong className="metric-value">{entry.value}</strong></div>)}</div>;
}

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [config, setConfig] = useState<ExperimentConfig | null>(null);
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [catalogAttempt, setCatalogAttempt] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  const [kind, setKind] = useState<FigureKind>("trajectory");
  const [view, setView] = useState<PolicyView>("Current policy");
  const [replay, setReplay] = useState(false);
  const [metric, setMetric] = useState<DiagnosticMetric>("nash_gap");
  const runController = useRef<AbortController | null>(null);
  const exportController = useRef<AbortController | null>(null);
  const runSequence = useRef(0);
  const tabList = useRef<HTMLDivElement>(null);
  const closeInfo = useCallback(() => setInfoOpen(false), []);

  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    getCatalog(controller.signal).then((next) => {
      if (controller.signal.aborted) return;
      const game = next.games.find((entry) => entry.name === "Pure Coordination") ?? next.games[0];
      if (!game) throw new Error("The game catalog is empty.");
      setCatalog(next);
      setConfig(initialConfig(game));
    }).catch((failure: unknown) => { if (!controller.signal.aborted) setError(errorMessage(failure)); });
    return () => controller.abort();
  }, [catalogAttempt]);
  useEffect(() => () => { runController.current?.abort(); exportController.current?.abort(); }, []);

  async function run() {
    if (!config || !catalog || running) return;
    const invalid = validateConfig(config, catalog);
    if (invalid) { setError(invalid); return; }
    const submitted = { ...config, algorithm_names: [...config.algorithm_names] };
    const sequence = ++runSequence.current;
    runController.current?.abort();
    const controller = new AbortController();
    runController.current = controller;
    setError(null);
    setRunning(true);
    try {
      const next = await runExperiment(submitted, controller.signal);
      if (controller.signal.aborted || sequence !== runSequence.current) return;
      setResult(next);
      setKind("trajectory"); setView("Current policy"); setReplay(false);
    } catch (failure: unknown) {
      if (!controller.signal.aborted && sequence === runSequence.current) setError(errorMessage(failure));
    } finally {
      if (!controller.signal.aborted && sequence === runSequence.current) setRunning(false);
    }
  }

  async function exportResult() {
    if (!result || exporting) return;
    const controller = new AbortController();
    exportController.current?.abort();
    exportController.current = controller;
    setExporting(true); setError(null);
    try { await downloadExperiment(result.id, controller.signal); }
    catch (failure: unknown) { if (!controller.signal.aborted) setError(errorMessage(failure)); }
    finally { if (!controller.signal.aborted) setExporting(false); }
  }

  function selectTab(next: FigureKind) { setKind(next); setReplay(false); }
  function navigateTabs(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const keys: Record<string, number> = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 };
    const next = keys[event.key];
    if (next === undefined) return;
    event.preventDefault();
    selectTab(tabs[next].value);
    tabList.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
  }

  const shownConfig = result?.config ?? config;
  const game = catalog?.games.find((entry) => entry.name === shownConfig?.game_name);
  return <div className="app-shell">
    <header className="studio-header"><a className="brand" href="/" aria-label="regret lab home">regret <span>/</span> lab</a><div className="header-actions"><span className="status-pill" aria-live="polite"><span aria-hidden="true" />{running ? "Computing" : catalog ? "Ready" : "Connecting"}</span><button className="export-button" type="button" disabled={!result || exporting} onClick={() => { void exportResult(); }}>{exporting ? "Exporting..." : "Export"}<span aria-hidden="true">↓</span></button></div></header>
    {error && <div className="error-banner" role="alert">{error}{!catalog && <button type="button" onClick={() => setCatalogAttempt((attempt) => attempt + 1)}>Retry</button>}</div>}
    {!catalog || !config || !game || !shownConfig ? <div className="loading-note" role="status">{error ? "Unable to load the workspace." : "Opening the workspace..."}</div> : <div className="studio-layout">
      <SetupPanel catalog={catalog} config={config} running={running} changed={Boolean(result && configChanged(config, result.config))} onChange={setConfig} onRun={() => { void run(); }} />
      <main className="workspace">
        <div className="workspace-heading"><div><span className="eyebrow">GAME DYNAMICS</span><h1>{game.name}</h1><div className="workspace-meta">{game.num_players} players <span>·</span> {game.action_labels.length} actions{result && <><span>·</span> {result.config.iterations.toLocaleString()} iterations</>}</div></div><button className="info-button" type="button" onClick={() => setInfoOpen(true)}>Info<span aria-hidden="true">↗</span></button></div>
        <section className={`chart-card${!result ? " chart-card-empty" : ""}`} aria-label="Research visualization">
          <div className="chart-toolbar"><div className="chart-tabs" role="tablist" aria-label="Visualization" ref={tabList}>
            {tabs.map((tab, index) => <button className="chart-tab" type="button" role="tab" id={`tab-${tab.value}`} aria-controls="visualization-panel" aria-selected={kind === tab.value} tabIndex={kind === tab.value ? 0 : -1} disabled={!result} key={tab.value} onClick={() => selectTab(tab.value)} onKeyDown={(event) => navigateTabs(event, index)}>{tab.label}</button>)}
          </div>{result && <div className="chart-controls">
            {kind === "diagnostics" && <select className="metric-select" aria-label="Diagnostic metric" value={metric} onChange={(event) => setMetric(event.target.value as DiagnosticMetric)}><option value="nash_gap">Nash gap</option><option value="average_regret">Expected regret</option><option value="payoffs">Expected payoff</option></select>}
            {kind === "trajectory" && <div className="segmented policy-segmented" role="group" aria-label="Policy view">{([{ value: "Current policy", label: "Current" }, { value: "Time-average policy", label: "Average" }, ...(result.empirical_available ? [{ value: "Empirical frequencies", label: "Empirical" }] : [])] as { value: PolicyView; label: string }[]).map((choice) => <button className="segment" type="button" key={choice.value} aria-pressed={view === choice.value} title={choice.value === "Empirical frequencies" ? "Historical action frequencies for Fictitious Play methods only." : undefined} onClick={() => setView(choice.value)}>{choice.label}</button>)}</div>}
            {kind === "trajectory" && <button className="replay-button" type="button" aria-pressed={replay} onClick={() => setReplay((active) => !active)}>{replay ? "Stop" : "Replay"}<span aria-hidden="true">{replay ? "Ⅱ" : "▷"}</span></button>}
          </div>}</div>
          <div className="chart-panel" id="visualization-panel" role="tabpanel" aria-labelledby={`tab-${kind}`} tabIndex={0}>{result ? <PlotView resultId={result.id} kind={kind} view={kind === "trajectory" ? view : "Current policy"} replay={replay} metric={metric} /> : <EmptyState />}</div>
          <div className="chart-footer"><div className="player-key">{Array.from({ length: game.num_players }, (_, player) => <span className={`player-chip player-${player + 1}`} key={player}><span aria-hidden="true" />P{player + 1}</span>)}</div>{result && <span className="chart-run-note">{result.config.runs} repeats · seed {result.config.seed}</span>}</div>
        </section>
        {result && <Metrics result={result} />}
      </main>
      {infoOpen && <InfoDrawer key={game.name} game={game} algorithms={catalog.algorithms} config={shownConfig} onClose={closeInfo} />}
    </div>}
  </div>;
}

import { useEffect, useRef, useState } from "react";
import type { AlgorithmSpec, ExperimentConfig, GameSpec } from "./types";

interface Props { game: GameSpec; algorithms: AlgorithmSpec[]; config: ExperimentConfig; onClose: () => void }

function PayoffExplorer({ game }: { game: GameSpec }) {
  const [actions, setActions] = useState(() => Array.from({ length: game.num_players }, () => 0));
  const row = game.payoffs.find((entry) => entry.actions.every((action, player) => action === actions[player]));
  return <details className="drawer-section"><summary>Payoff explorer</summary>
    <div className="payoff-controls">{actions.map((action, player) => <label className="field" key={player} htmlFor={`payoff-action-${player}`}><span className="field-label">Player {player + 1} action</span><select id={`payoff-action-${player}`} value={action} onChange={(event) => setActions(actions.map((value, index) => index === player ? Number(event.target.value) : value))}>{game.action_labels.map((label, index) => <option value={index} key={label}>{label}</option>)}</select></label>)}</div>
    <div className="payoff-values">{Array.from({ length: game.num_players }, (_, player) => <div className={`payoff-value player-${player + 1}`} data-testid={`payoff-p${player + 1}`} key={player}><span>P{player + 1}</span><strong>{row ? Number(row.values[player].toFixed(3)) : "-"}</strong></div>)}</div>
    <p>Base utilities for this joint action. Learning feedback may include noise in noisy games.</p>
  </details>;
}

export default function InfoDrawer({ game, algorithms, config, onClose }: Props) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const drawer = useRef<HTMLElement>(null);
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButton.current?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab") return;
      const focusable = Array.from(drawer.current?.querySelectorAll<HTMLElement>("button, select, summary, [tabindex='0']") ?? []).filter((node) => node.getClientRects().length > 0);
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener("keydown", handleKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", handleKey); document.body.style.overflow = previousOverflow; previous?.focus(); };
  }, [onClose]);
  return <div className="drawer-backdrop" onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <aside className="info-drawer" ref={drawer} role="dialog" aria-modal="true" aria-labelledby="info-title">
      <div className="drawer-header"><span className="eyebrow">FIELD NOTES</span><button className="close-button" type="button" ref={closeButton} onClick={onClose} aria-label="Close Info">×</button></div>
      <h2 id="info-title">{game.name}</h2><p>{game.description}</p>
      <details className="drawer-section"><summary>How to read</summary><p>Each path follows the policy used to sample actions, including any exploration. Pale context shows history, while the moving point shows the selected iteration. Player colors stay consistent.</p><p>{game.num_players === 3 ? `Each cube axis is one player's probability of ${game.action_labels[0]}.` : game.action_labels.length === 2 ? "A two-action policy lies on a line between its actions." : game.action_labels.length === 3 ? "Triangle coordinates identify the three action probabilities." : "The polygon projection is not one-to-one. Distinct policies can occupy the same point; use Final policies for exact probabilities."}</p><p>Time-average policy averages decisions over time. Fictitious Play empirical frequencies include five initialization pseudocounts per player along with observed action counts.</p></details>
      <details className="drawer-section"><summary>Methods & metrics</summary>{algorithms.filter((algorithm) => config.algorithm_names.includes(algorithm.name)).map((algorithm) => <div className="method-note" key={algorithm.name}><h3>{algorithm.name}</h3><p>{algorithm.description}</p><span className="feedback-tag">{algorithm.feedback}</span></div>)}<p>Nash gap sums gains from unilateral best-response deviations. Zero certifies a Nash equilibrium for the displayed profile.</p><p>Expected external regret compares the best fixed action with the learner's expected utilities against the observed sequence of opponent policies. It is distinct from realized bandit regret or a convergence guarantee.</p><p>Diagnostics use base expected utilities, including for noisy games.</p></details>
      <details className="drawer-section"><summary>Equilibrium examples <span>{game.equilibria.length}</span></summary><p>Diamonds mark numerically certified examples. Pure equilibria are enumerated; mixed profiles and continuous equilibrium sets may have additional members.</p>{game.equilibria.map((profile, index) => <div className="equilibrium-profile" key={index}><span>Example {index + 1}</span>{profile.map((probabilities, player) => <p key={player}>P{player + 1}: {probabilities.map((probability, action) => `${game.action_labels[action]} ${(probability * 100).toFixed(1)}%`).join(" · ")}</p>)}</div>)}</details>
      <PayoffExplorer game={game} />
      <details className="drawer-section"><summary>Reproducibility</summary><p>Seed {config.seed} · {config.runs} repeats · {config.iterations.toLocaleString()} iterations · {config.initialization} initialization.</p><p>Each repeat is independent. Export contains the full numerical histories, seeds, settings, metric definitions, and scalar summaries.</p></details>
    </aside>
  </div>;
}

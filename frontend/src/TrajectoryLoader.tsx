import { useEffect, useState } from "react";
import type { ComponentType } from "react";
import type { PlotlyFigure } from "./types";

interface RenderProps { figure: PlotlyFigure; replay: boolean }
type RendererModule = { default: ComponentType<RenderProps> };
const loadRenderer = () => import("./TrajectoryCanvas");

/** A failed optional chunk stays inside the chart, preserving setup and results. */
export default function TrajectoryLoader({ figure, replay, load = loadRenderer }: RenderProps & { load?: () => Promise<RendererModule> }) {
  const [Renderer, setRenderer] = useState<ComponentType<RenderProps> | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let current = true;
    setFailed(false);
    load().then((module) => { if (current) setRenderer(() => module.default); })
      .catch(() => { if (current) setFailed(true); });
    return () => { current = false; };
  }, [load, attempt]);
  if (failed) return <div className="error-banner" role="alert">Could not load this view.
    <span><button type="button" onClick={() => setAttempt((value) => value + 1)}>Retry</button><button type="button" onClick={() => window.location.reload()}>Reload page</button></span>
  </div>;
  return Renderer ? <Renderer figure={figure} replay={replay} /> : <div className="loading-note" role="status">Drawing...</div>;
}

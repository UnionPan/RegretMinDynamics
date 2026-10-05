import { memo, useEffect, useRef, useState } from "react";
import { errorMessage } from "./api";
import type { PlotlyFigure } from "./types";

export interface PlotlyRuntime {
  react: (element: HTMLElement, data: Record<string, unknown>[], layout: Record<string, unknown>, config: Record<string, unknown>) => Promise<unknown>;
  addFrames: (element: HTMLElement, frames: NonNullable<PlotlyFigure["frames"]>) => Promise<unknown>;
  animate: (element: HTMLElement, frames: string[] | null, options: Record<string, unknown>) => Promise<unknown>;
  purge: (element: HTMLElement) => void;
  Plots: { resize: (element: HTMLElement) => Promise<unknown> | void };
}

interface Props { figure: PlotlyFigure; replay: boolean }

const PlotCanvas = memo(function PlotCanvas({ figure, replay }: Props) {
  const element = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const host = element.current!;
    const node = document.createElement("div");
    node.className = "plot-target";
    host.replaceChildren(node);
    let cancelled = false;
    let runtime: PlotlyRuntime | undefined;
    let observer: ResizeObserver | undefined;
    setError(null);
    async function draw() {
      const loaded = await import("plotly.js-basic-dist-min");
      if (cancelled) return;
      runtime = loaded.default;
      const margin = typeof figure.layout.margin === "object" && figure.layout.margin !== null ? figure.layout.margin : {};
      await runtime.react(node, figure.data, { ...figure.layout, title: undefined, autosize: true, margin: { ...margin, t: 35 } }, {
        responsive: false, displaylogo: false, displayModeBar: "hover", scrollZoom: false,
      });
      if (cancelled) { runtime.purge(node); return; }
      observer = new ResizeObserver(() => {
        if (runtime && !cancelled) Promise.resolve(runtime.Plots.resize(node)).catch((failure: unknown) => {
          if (!cancelled) setError(errorMessage(failure));
        });
      });
      observer.observe(node);
      if (replay && figure.frames?.length) {
        await runtime.addFrames(node, figure.frames);
        if (!cancelled) await runtime.animate(node, null, {
          frame: { duration: 80, redraw: false }, transition: { duration: 0 }, mode: "immediate",
        });
      }
    }
    draw().catch((failure: unknown) => { if (!cancelled) setError(errorMessage(failure)); });
    return () => {
      cancelled = true;
      observer?.disconnect();
      runtime?.purge(node);
      node.remove();
    };
  }, [figure, replay]);
  return <>{error && <div className="error-banner" role="alert">{error}</div>}<div ref={element} className="plot-surface" data-testid="plot-surface" aria-label="Experiment visualization" /></>;
});

export default PlotCanvas;

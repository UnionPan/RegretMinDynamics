import { memo, useEffect, useState } from "react";
import { errorMessage, getFigure } from "./api";
import PlotCanvas from "./PlotCanvas";
import TrajectoryLoader from "./TrajectoryLoader";
import type { DiagnosticMetric, FigureKind, PlotlyFigure, PolicyView } from "./types";


interface Props {
  resultId: string;
  kind: FigureKind;
  view: PolicyView;
  replay: boolean;
  metric: DiagnosticMetric;
}

const PlotView = memo(function PlotView({ resultId, kind, view, replay, metric }: Props) {
  const [figure, setFigure] = useState<{ value: PlotlyFigure; key: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestKey = JSON.stringify([resultId, kind, view, metric]);
  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    setLoading(true);
    setError(null);
    setFigure(null);
    getFigure(resultId, kind, view, false, metric, controller.signal)
      .then((next) => { if (current) setFigure({ value: next, key: requestKey }); })
      .catch((failure: unknown) => { if (current && !controller.signal.aborted) setError(errorMessage(failure)); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; controller.abort(); };
  }, [resultId, kind, view, metric, requestKey]);
  return <div className="plot-container" aria-busy={loading}>
    {figure?.key === requestKey && <>{kind === "trajectory" ? <TrajectoryLoader key={requestKey} figure={figure.value} replay={replay} /> : <PlotCanvas key={requestKey} figure={figure.value} replay={false} />}</>}
    {loading && <div className="loading-overlay" role="status">Drawing...</div>}
    {error && <div className="error-banner" role="alert">{error}</div>}
  </div>;
});

export default PlotView;

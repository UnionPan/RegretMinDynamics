import { memo, useEffect, useMemo, useRef, useState } from "react";
import { errorMessage } from "./api";
import { extractTrajectory, pathAppearance, PLAYER_COLORS, PLAYBACK_NOTE, projectPanel, sampleAt, sampleDescription, TRAIL_FRACTION, trailAt } from "./trajectoryData";
import type { CubeHandle, Panel, TrajectoryData } from "./trajectoryData";
import type { PlotlyFigure } from "./types";

interface Props { figure: PlotlyFigure; replay: boolean }
const TAIL_BANDS = 6;

const SvgPanel = memo(function SvgPanel({ panel, data, replay }: { panel: Panel; data: TrajectoryData; replay: boolean }) {
  const color = PLAYER_COLORS[panel.player % PLAYER_COLORS.length];
  const coordinate = (position: [number, number, number]) => projectPanel(panel, position).join(",");
  const paths = data.paths.filter((path) => path.player === panel.player);
  return <svg className="trajectory-svg" viewBox="0 0 480 360" role="img" aria-label={"Player " + (panel.player + 1) + " strategy trajectory"}>
    <text x="24" y="28" fill={color} fontWeight="600">Player {panel.player + 1}</text>
    {panel.boundary.length > 0 && <polyline points={panel.boundary.map(coordinate).join(" ")} fill="#f8faf9" stroke={color} strokeOpacity="0.5" strokeWidth="1.2" />}
    {panel.actions.map((action, index) => {
      const [x, y] = projectPanel(panel, action.position);
      return <text key={index} x={x} y={y - 12} fill={color} textAnchor="middle">{action.label}</text>;
    })}
    {paths.map((path) => <polyline key={path.key} data-history-key={path.key} data-group={path.algorithm} display={replay ? "none" : "block"} points={path.points.map((point) => coordinate(point.position)).join(" ")} fill="none" stroke={path.color} strokeWidth={pathAppearance(path.run).width} strokeOpacity={pathAppearance(path.run).opacity} strokeLinecap="round" strokeLinejoin="round">
      <title>{path.algorithm + " · Repeat " + (path.run + 1) + "\nRecorded iterations " + path.points[0].iteration.toLocaleString() + " to " + path.points.at(-1)!.iteration.toLocaleString()}</title>
    </polyline>)}
    {paths.flatMap((path) => Array.from({ length: TAIL_BANDS }, (_, band) => <polyline key={path.key + "-" + band} data-trail-key={path.key} data-tail-band={band} data-group={path.algorithm} display={replay ? "block" : "none"} points="" fill="none" stroke={path.color} strokeWidth={pathAppearance(path.run).width} strokeOpacity={pathAppearance(path.run).opacity * Math.pow((band + 0.5) / TAIL_BANDS, 1.6)} strokeLinecap="round" strokeLinejoin="round" />))}
    {data.references.filter((reference) => reference.player === panel.player).map((reference) => {
      const [x, y] = projectPanel(panel, reference.position);
      return <path key={reference.key} data-reference data-group="Equilibrium references" visibility="visible" d={"M" + x + " " + (y - 4) + "l4 4l-4 4l-4 -4Z"} fill="white" stroke="#64748b" strokeWidth="1.1">
        <title>{reference.label + "\n" + reference.probabilities.map((value, action) => (panel.actions[action]?.label ?? "Action " + action) + ": " + value.toFixed(3)).join(", ")}</title>
      </path>;
    })}
    {paths.map((path) => {
      const sample = path.points.at(-1)!, [x, y] = projectPanel(panel, sample.position);
      return <circle key={path.key} data-path-key={path.key} data-group={path.algorithm} cx={x} cy={y} r={path.run === 0 ? 3 : 2.3} fill="white" stroke={path.color} strokeWidth="1.2"><title>{sampleDescription(path, sample)}</title></circle>;
    })}
  </svg>;
});

const TrajectoryCanvas = memo(function TrajectoryCanvas({ figure, replay }: Props) {
  const parsed = useMemo(() => {
    try { return { data: extractTrajectory(figure), error: null }; }
    catch (failure) { return { data: null, error: errorMessage(failure) }; }
  }, [figure]);
  const host = useRef<HTMLDivElement>(null);
  const slider = useRef<HTMLInputElement>(null);
  const readout = useRef<HTMLOutputElement>(null);
  const seek = useRef<(iteration: number) => void>(() => {});
  const play = useRef<() => void>(() => {});
  const cancelPlayback = useRef<() => void>(() => {});
  const cube = useRef<CubeHandle | null>(null);
  const ready = useRef(false);
  const iteration = useRef(1);
  const replayRequested = useRef(replay);
  const fading = useRef(false);
  replayRequested.current = replay;
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const hiddenChoices = useRef<Record<string, boolean>>({});

  useEffect(() => {
    const data = parsed.data;
    if (!data?.paths.length || !host.current) return;
    const container = host.current, controller = new AbortController();
    let frame = 0, started: number | null = null, startIteration = 1;
    ready.current = false;
    const svgPaths = data.cube ? [] : data.paths.map((path) => ({
      path, panel: data.panels.find((item) => item.player === path.player)!,
      marker: container.querySelector<SVGCircleElement>('[data-path-key="' + path.key + '"]'),
      history: container.querySelector<SVGPolylineElement>('[data-history-key="' + path.key + '"]'),
      trails: Array.from(container.querySelectorAll<SVGPolylineElement>('[data-trail-key="' + path.key + '"]')),
    }));
    const update = (value: number) => {
      iteration.current = value;
      cube.current?.seek(value);
      svgPaths.forEach(({ path, panel, marker, history, trails }) => {
        const sample = sampleAt(path, value);
        const [x, y] = projectPanel(panel, sample.position);
        marker?.setAttribute("cx", String(x)); marker?.setAttribute("cy", String(y));
        const title = marker?.querySelector("title");
        if (title) title.textContent = sampleDescription(path, sample);
      const active = replayRequested.current;
        history?.setAttribute("display", active && fading.current ? "none" : "block");
        if (!fading.current || !active) {
          const points = active ? trailAt(path, sample.iteration, Infinity) : path.points;
          history?.setAttribute("points", points.map((point) => projectPanel(panel, point.position).join(",")).join(" "));
        }
        const duration = Math.max(1, data.lastIteration * TRAIL_FRACTION), start = sample.iteration - duration;
        trails.forEach((trail, band) => {
          trail.setAttribute("display", active && fading.current ? "block" : "none");
          if (!active || !fading.current) return;
          const bandEnd = start + duration * (band + 1) / TAIL_BANDS;
          const points = bandEnd <= path.points[0].iteration ? [] : trailAt(path, bandEnd, duration / TAIL_BANDS);
          trail.setAttribute("points", points.map((point) => projectPanel(panel, point.position).join(",")).join(" "));
        });
      });
      if (slider.current) slider.current.value = String(value);
      if (readout.current) readout.current.textContent = (data.paths.some((path) => sampleAt(path, value).interpolated) ? "≈ " : "") + "iteration " + Number(value.toFixed(1)).toLocaleString();
    };
    seek.current = update;
    const stop = () => { cancelAnimationFrame(frame); frame = 0; };
    cancelPlayback.current = stop;
    const tick = (now: number) => {
      if (controller.signal.aborted) return;
      if (started === null) started = now;
      const progress = Math.min((now - started) / 12000, 1);
      if (progress === 1) {
        fading.current = false;
        cube.current?.setReplay(replayRequested.current, false);
      }
      update(startIteration + progress * (data.lastIteration - startIteration));
      if (progress < 1) frame = requestAnimationFrame(tick);
      else { frame = 0; setPlaying(false); }
    };
    play.current = () => {
      if (!ready.current) return;
      stop(); started = null;
      startIteration = iteration.current >= data.lastIteration ? 1 : iteration.current;
      fading.current = true;
      cube.current?.setReplay(true, true);
      update(startIteration);
      setPlaying(true); frame = requestAnimationFrame(tick);
    };
    setError(null);
    const finish = () => {
      if (controller.signal.aborted) return;
      ready.current = true;
      fading.current = replayRequested.current;
      cube.current?.setReplay(replayRequested.current, fading.current);
      update(replayRequested.current ? 1 : data.lastIteration);
      if (replayRequested.current) play.current();
    };
    if (data.cube) import("./trajectoryCube").then(({ mountCube }) => mountCube(container, data, controller.signal)).then((mounted) => {
      if (controller.signal.aborted) mounted?.dispose();
      else { cube.current = mounted; Object.entries(hiddenChoices.current).forEach(([name, isHidden]) => mounted?.setVisible(name, !isHidden)); finish(); }
    }).catch((failure) => { if (!controller.signal.aborted) setError("3D graphics unavailable. " + errorMessage(failure)); });
    else finish();
    return () => { controller.abort(); ready.current = false; stop(); cube.current?.dispose(); cube.current = null; };
  }, [parsed]);

  useEffect(() => {
    if (!ready.current || !parsed.data) return;
    cancelPlayback.current();
    fading.current = replay;
    cube.current?.setReplay(replay, replay);
    seek.current(replay ? 1 : parsed.data.lastIteration);
    if (replay) play.current(); else setPlaying(false);
  }, [replay, parsed]);

  const data = parsed.data;
  if (parsed.error) return <div className="error-banner" role="alert">{parsed.error}</div>;
  if (!data?.paths.length) return <div className="trajectory-empty">No recorded trajectory is available.</div>;
  const algorithms = [...new Map(data.paths.map((path) => [path.algorithm, path])).values()];
  const pausePlayback = () => {
    cancelPlayback.current(); fading.current = false;
    cube.current?.setReplay(replayRequested.current, false);
    setPlaying(false);
  };
  const togglePlayback = () => {
    if (playing) { pausePlayback(); seek.current(iteration.current); } else play.current();
  };
  const toggleGroup = (name: string) => {
    const willHide = !hiddenChoices.current[name];
    hiddenChoices.current = { ...hiddenChoices.current, [name]: willHide };
    setHidden(hiddenChoices.current);
    cube.current?.setVisible(name, !willHide);
    host.current?.querySelectorAll<SVGElement>("[data-group]").forEach((node) => {
      if (node.dataset.group === name) node.setAttribute("visibility", willHide ? "hidden" : "visible");
    });
  };
  const legend = (name: string, color: string, diamond = false) => <button type="button" className="trajectory-legend-item" key={name} aria-pressed={!hidden[name]} title={diamond ? "Verified equilibrium reference" : "First repeat emphasized; additional repeats shown as lighter lines"} onClick={() => toggleGroup(name)}><i aria-hidden="true" className={diamond ? "reference-swatch" : "line-swatch"} style={{ backgroundColor: diamond ? "white" : color, border: diamond ? "1px solid #64748b" : undefined, transform: diamond ? "rotate(45deg)" : undefined }} />{name}</button>;
  return <section className="trajectory-workspace" aria-label="Strategy trajectories">
    {error && <div className="error-banner" role="alert">{error}</div>}
    <div ref={host} className={"trajectory-viewport " + (data.cube ? "trajectory-cube" : "trajectory-panels")}>
      {!data.cube && data.panels.map((panel) => <SvgPanel key={panel.player} panel={panel} data={data} replay={replay} />)}
    </div>
    {replay && <div className="trajectory-playback">
      <button type="button" onClick={togglePlayback}>{playing ? "Pause" : "Play"}</button>
      <input ref={slider} type="range" min="1" max={data.lastIteration} step="any" defaultValue="1" aria-label="Playback iteration" onInput={(event) => { pausePlayback(); seek.current(Number(event.currentTarget.value)); }} />
      <output ref={readout} className="trajectory-readout">iteration 1</output>
      <span className="trajectory-help" title={PLAYBACK_NOTE} aria-label={PLAYBACK_NOTE}>ⓘ</span>
    </div>}
    <div className="trajectory-legend">{algorithms.map((path) => legend(path.algorithm, path.color))}{data.references.length > 0 && legend("Equilibrium references", "#172235", true)}{data.cube && <button className="trajectory-reset" type="button" onClick={() => cube.current?.reset()}>Reset view</button>}</div>
  </section>;
});

export default TrajectoryCanvas;

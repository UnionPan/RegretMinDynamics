import type { PlotlyFigure } from "./types";

export const PLAYER_COLORS = ["#3866e9", "#c78322", "#218c63"];
export const PLAYBACK_NOTE = "Playback interpolates sampled points; research metrics use recorded policies";
export type Position = [number, number, number];
export interface Sample { position: Position; iteration: number; probabilities: number[]; interpolated: boolean }
export interface TrajectoryPath { key: string; algorithm: string; run: number; player: number; color: string; dash: string; opacity: number; points: Sample[]; probabilityLabels?: string[] }
export interface Reference { key: string; label: string; player: number; position: Position; probabilities: number[] }
export interface Panel { player: number; boundary: Position[]; actions: { label: string; position: Position }[]; bounds: [number, number, number, number] }
export interface TrajectoryData { cube: boolean; paths: TrajectoryPath[]; references: Reference[]; panels: Panel[]; axes: string[]; lastIteration: number }
const record = (value: unknown): Record<string, unknown> => value && typeof value === "object" ? value as Record<string, unknown> : {};

/** Decode Plotly's little-endian NumPy payloads without alignment assumptions. */
export function decodeNumeric(value: unknown): number[] | number[][] {
  if (Array.isArray(value)) return value.map((item) => Array.isArray(item) ? item.map(Number) : Number(item)) as number[] | number[][];
  const encoded = record(value);
  if (!encoded.bdata) return [];
  const dtype = String(encoded.dtype);
  const match = dtype.match(/^([<>|=]?)([fiu])(1|2|4|8)$/);
  if (!match || (match[2] === "f" && !["4", "8"].includes(match[3]))) throw new Error("Unsupported numeric dtype.");
  const width = Number(match[3]);
  const bytes = Uint8Array.from(atob(String(encoded.bdata)), (character) => character.charCodeAt(0));
  if (bytes.length % width) throw new Error("Invalid numeric byte length.");
  const view = new DataView(bytes.buffer);
  const little = match[1] !== ">";
  const methods: Record<string, (at: number) => number> = {
    f4: (at) => view.getFloat32(at, little), f8: (at) => view.getFloat64(at, little),
    i1: (at) => view.getInt8(at), u1: (at) => view.getUint8(at),
    i2: (at) => view.getInt16(at, little), u2: (at) => view.getUint16(at, little),
    i4: (at) => view.getInt32(at, little), u4: (at) => view.getUint32(at, little),
    i8: (at) => Number(view.getBigInt64(at, little)), u8: (at) => Number(view.getBigUint64(at, little)),
  };
  const result = Array.from({ length: bytes.length / width }, (_, index) => methods[match[2] + width](index * width));
  if (width === 8 && match[2] !== "f" && result.some((item) => !Number.isSafeInteger(item))) throw new Error("Integer exceeds browser precision.");
  if (!encoded.shape) return result;
  const shape = String(encoded.shape).split(",").map((size) => Number(size.trim()));
  if (shape.length > 2 || shape.some((size) => !Number.isInteger(size) || size < 0) || shape.reduce((a, b) => a * b, 1) !== result.length) throw new Error("Invalid numeric shape.");
  return shape.length === 1 ? result : Array.from({ length: shape[0] }, (_, row) => result.slice(row * shape[1], (row + 1) * shape[1]));
}

function vector(value: unknown): number[] { return decodeNumeric(value).flat(); }
function rows(value: unknown): number[][] { const decoded = decodeNumeric(value); return decoded.length && !Array.isArray(decoded[0]) ? [decoded as number[]] : decoded as number[][]; }
function playerOf(trace: Record<string, unknown>): number { return Number(record(trace.meta).player ?? (Number(String(trace.xaxis ?? "x").slice(1) || 1) - 1)); }
function positions(trace: Record<string, unknown>): Position[] {
  const x = vector(trace.x), y = vector(trace.y), z = vector(trace.z);
  if (x.length !== y.length || (z.length && z.length !== x.length)) throw new Error("Trajectory coordinate lengths differ.");
  if ([...x, ...y, ...z].some((item) => !Number.isFinite(item))) throw new Error("Trajectory contains invalid coordinates.");
  return x.map((item, index) => [item, y[index], z[index] ?? 0]);
}
function referenceLabel(trace: Record<string, unknown>): string {
  const hover = String(trace.hovertemplate ?? "");
  const extra = hover.match(/<extra>([^<]+)<\/extra>/)?.[1];
  return (String(record(trace.meta).reference ?? extra ?? hover.split("<br>")[0] ?? trace.name) || "Equilibrium reference").replace(/<[^>]*>/g, "");
}

export function extractTrajectory(figure: PlotlyFigure): TrajectoryData {
  const histories = figure.data.filter((trace) => record(trace.meta).role === "history");
  const cube = histories.some((trace) => trace.type === "scatter3d");
  const paths = histories.filter((trace) => vector(trace.x).length).map((trace, index): TrajectoryPath => {
    const meta = record(trace.meta), custom = rows(trace.customdata), line = record(trace.line);
    const coordinates = positions(trace);
    if (custom.length !== coordinates.length || custom.some((row, at) => row.length < 2 || !row.every(Number.isFinite) || row[0] < 1 || (at > 0 && row[0] <= custom[at - 1][0]))) throw new Error("Invalid recorded sample metadata.");
    return { key: `path-${index}`, algorithm: String(meta.algorithm ?? trace.name ?? "Algorithm"), run: Number(meta.run ?? 0), player: playerOf(trace), color: String(line.color ?? "#3866e9"), dash: String(line.dash ?? "solid"), opacity: Number(trace.opacity ?? 0.8), points: coordinates.map((position, at) => ({ position, iteration: custom[at]?.[0] ?? at + 1, probabilities: custom[at]?.slice(1) ?? [], interpolated: false })) };
  });
  const references = figure.data.filter((trace) => record(trace.meta).role === "equilibrium").flatMap((trace, index): Reference[] => {
    const custom = rows(trace.customdata);
    return positions(trace).map((position, at) => ({ key: `reference-${index}-${at}`, label: referenceLabel(trace), player: playerOf(trace), position, probabilities: custom[at] ?? (cube ? [...position] : []) }));
  });
  const panels = cube ? [] : [...new Set(paths.map((path) => path.player))].map((player): Panel => {
    const geometry = figure.data.filter((trace) => playerOf(trace) === player);
    const boundary = positions(geometry.find((trace) => record(trace.meta).role === "boundary") ?? { x: [], y: [] });
    const labels = geometry.find((trace) => record(trace.meta).role === "action_labels");
    const actions = labels ? positions(labels).map((position, at) => ({ position, label: String((labels.text as string[])?.[at] ?? "") })) : [];
    const points = [...boundary, ...actions.map((action) => action.position), ...paths.filter((path) => path.player === player).flatMap((path) => path.points.map((point) => point.position))];
    const xs = points.map((point) => point[0]), ys = points.map((point) => point[1]);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    return { player, boundary, actions, bounds: [Math.min(...xs), Math.max(...xs), minY === maxY ? minY - 0.1 : minY, minY === maxY ? maxY + 0.1 : maxY] };
  });
  const scene = record(figure.layout.scene);
  const axes = ["xaxis", "yaxis", "zaxis"].map((axis, player) => String(record(record(scene[axis]).title).text ?? `P${player + 1}: P(Action 0)`).replace(/<br>.*/, "").replace(":", " ·"));
  return { cube, paths: paths.map((path) => ({ ...path, probabilityLabels: cube ? axes : panels.find((panel) => panel.player === path.player)?.actions.map((action) => action.label) })), references, panels, axes, lastIteration: Math.max(1, ...paths.map((path) => path.points.at(-1)!.iteration)) };
}

/** Straight-line visual interpolation only. No simulation or metric is recomputed. */
export function sampleAt(path: TrajectoryPath, iteration: number): Sample {
  const points = path.points;
  if (iteration <= points[0].iteration) return points[0];
  if (iteration >= points.at(-1)!.iteration) return points.at(-1)!;
  let low = 0, high = points.length - 1;
  while (high - low > 1) { const mid = (low + high) >>> 1; if (points[mid].iteration <= iteration) low = mid; else high = mid; }
  if (points[low].iteration === iteration) return points[low];
  if (points[high].iteration === iteration) return points[high];
  const a = points[low], b = points[high], weight = (iteration - a.iteration) / (b.iteration - a.iteration);
  return { position: a.position.map((value, axis) => value + weight * (b.position[axis] - value)) as Position, iteration, probabilities: a.probabilities.map((value, action) => value + weight * (b.probabilities[action] - value)), interpolated: true };
}

export function sampleDescription(path: TrajectoryPath, sample: Sample): string {
  return `${path.algorithm} · Repeat ${path.run + 1} · ${sample.interpolated ? "Approximate" : "Recorded"} iteration ${Number(sample.iteration.toFixed(1)).toLocaleString()}${sample.probabilities.length ? "\nProbabilities: " + sample.probabilities.map((value, index) => (path.probabilityLabels?.[index] ?? "Action " + index) + ": " + value.toFixed(3)).join(", ") : ""}`;
}
export function projectPanel(panel: Panel, position: Position): [number, number] {
  const [left, right, bottom, top] = panel.bounds;
  const scale = Math.min(360 / Math.max(right - left, 0.001), 250 / (top - bottom));
  return [240 + (position[0] - (left + right) / 2) * scale, 185 - (position[1] - (bottom + top) / 2) * scale];
}
export function clampLabel(x: number, y: number, width: number, height: number, labelWidth: number, labelHeight: number): [number, number] {
  return [Math.max(8, Math.min(x, width - labelWidth - 8)), Math.max(8, Math.min(y, height - labelHeight - 8))];
}

/** Keep the exact sampled segment shape, including interpolated clipping boundaries. */
export function trailAt(path: TrajectoryPath, iteration: number, duration: number): Sample[] {
  const end = sampleAt(path, iteration);
  const start = sampleAt(path, Math.max(path.points[0].iteration, end.iteration - duration));
  if (start.iteration === end.iteration) return [end];
  return [start, ...path.points.filter((point) => point.iteration > start.iteration && point.iteration < end.iteration), end];
}

export const TRAIL_FRACTION = 0.12;
export const pathAppearance = (run: number) => ({ width: run === 0 ? 1.4 : 1, opacity: run === 0 ? 0.78 : 0.3 });

export interface CubeHandle { seek: (iteration: number) => void; setReplay: (active: boolean, fading?: boolean) => void; setVisible: (name: string, visible: boolean) => void; reset: () => void; dispose: () => void }


interface LabelBox { x: number; y: number; width: number; height: number }
/** Keep overlay text readable when rotation projects several axis labels together. */
export function layoutLabels(labels: LabelBox[], width: number, height: number): [number, number][] {
  const placed: { x: number; y: number; width: number; height: number }[] = [];
  return labels.map((label) => {
    let candidate = clampLabel(label.x - label.width / 2, label.y, width, height, label.width, label.height);
    const overlaps = ([x, y]: [number, number]) => placed.some((box) => x < box.x + box.width + 4 && x + label.width + 4 > box.x && y < box.y + box.height + 4 && y + label.height + 4 > box.y);
    for (let step = 0; overlaps(candidate) && step < 40; step++) {
      const distance = (Math.floor(step / 4) + 1) * (label.height + 7);
      const direction = step % 4;
      candidate = clampLabel(label.x - label.width / 2 + (direction === 2 ? distance : direction === 3 ? -distance : 0), label.y + (direction === 0 ? distance : direction === 1 ? -distance : 0), width, height, label.width, label.height);
    }
    placed.push({ ...label, x: candidate[0], y: candidate[1] });
    return candidate;
  });
}

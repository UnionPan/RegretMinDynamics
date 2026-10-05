import { describe, expect, it } from "vitest";
import { clampLabel, decodeNumeric, extractTrajectory, sampleAt, sampleDescription, projectPanel } from "./trajectoryData";
import type { PlotlyFigure } from "./types";
import * as trajectoryData from "./trajectoryData";

function encoded(dtype: string, values: number[], shape?: string) {
  const widths: Record<string, number> = { f4: 4, f8: 8, i1: 1, i2: 2, i4: 4, i8: 8, u1: 1, u2: 2, u4: 4, u8: 8 };
  const buffer = new ArrayBuffer(values.length * widths[dtype]);
  const view = new DataView(buffer);
  values.forEach((value, index) => {
    const at = index * widths[dtype];
    if (dtype === "f4") view.setFloat32(at, value, true);
    else if (dtype === "f8") view.setFloat64(at, value, true);
    else if (dtype === "i1") view.setInt8(at, value);
    else if (dtype === "u1") view.setUint8(at, value);
    else if (dtype === "i2") view.setInt16(at, value, true);
    else if (dtype === "u2") view.setUint16(at, value, true);
    else if (dtype === "i4") view.setInt32(at, value, true);
    else if (dtype === "u4") view.setUint32(at, value, true);
    else if (dtype === "i8") view.setBigInt64(at, BigInt(value), true);
    else view.setBigUint64(at, BigInt(value), true);
  });
  return { dtype, bdata: btoa(String.fromCharCode(...new Uint8Array(buffer))), shape };
}

export const fixture: PlotlyFigure = { layout: {}, data: [
  { meta: { role: "boundary" }, x: [0, 1, 0.5, 0], y: [0, 0, 1, 0], xaxis: "x" },
  { meta: { role: "action_labels" }, x: [0, 1, 0.5], y: [0, 0, 1], text: ["Rock", "Paper", "Scissors"], xaxis: "x" },
  { meta: { role: "history", algorithm: "Hedge", run: 1, player: 0 }, x: [0, 0.5, 1], y: [0, 0.5, 0], customdata: [[1, 1, 0, 0], [5, 0.25, 0.25, 0.5], [9, 0, 1, 0]], line: { color: "#3866e9", dash: "dash" }, opacity: 0.6 },
  { meta: { role: "equilibrium" }, x: [0.5], y: [1 / 3], customdata: [[1 / 3, 1 / 3, 1 / 3]], hovertemplate: "Rock<extra>Uniform reference</extra>" },
] };

it("clips fading trails to past samples with exact interpolated boundaries and no data mutation", () => {
  expect(trajectoryData.trailAt).toBeTypeOf("function");
  const path = extractTrajectory(fixture).paths[0];
  const original = JSON.stringify(path);
  const trail = trajectoryData.trailAt(path, 7, 4);
  expect(trail.map((point) => point.iteration)).toEqual([3, 5, 7]);
  expect(trail.map((point) => point.position)).toEqual([[0.25, 0.25, 0], [0.5, 0.5, 0], [0.75, 0.25, 0]]);
  expect(trajectoryData.trailAt(path, 1, 4)).toHaveLength(1);
  expect(trajectoryData.trailAt(path, 9, 2).map((point) => point.iteration)).toEqual([7, 9]);
  expect(JSON.stringify(path)).toBe(original);
});

describe("Plotly numeric decoding", () => {
  it.each(["f4", "f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"])("decodes %s without browser typed-array alignment assumptions", (dtype) => {
    expect(decodeNumeric(encoded(dtype, [1, 2, 3]))).toEqual([1, 2, 3]);
  });
  it("preserves two-dimensional customdata and copies ordinary arrays", () => {
    expect(decodeNumeric(encoded("f8", [1, 0.2, 3, 0.4], "2, 2"))).toEqual([[1, 0.2], [3, 0.4]]);
    const values = [[1, 2], [3, 4]];
    expect(decodeNumeric(values)).toEqual(values);
    expect(decodeNumeric(values)).not.toBe(values);
    expect(decodeNumeric(undefined)).toEqual([]);
  });
  it("rejects inconsistent shapes and unsupported dtypes", () => {
    expect(() => decodeNumeric(encoded("f8", [1, 2], "2,3"))).toThrow(/shape/);
    expect(() => decodeNumeric({ dtype: "c16", bdata: "AAAA" })).toThrow(/dtype/);
    expect(() => decodeNumeric({ dtype: "f8", bdata: "AA==" })).toThrow(/length/);
  });
});

it("extracts sampled geometry, metadata, reference identity and final probabilities immutably", () => {
  const before = JSON.stringify(fixture);
  const data = extractTrajectory(fixture);
  expect(data.cube).toBe(false);
  expect(data.paths[0]).toMatchObject({ algorithm: "Hedge", run: 1, player: 0, color: "#3866e9", dash: "dash", opacity: 0.6 });
  expect(data.lastIteration).toBe(9);
  expect(data.references[0].label).toBe("Uniform reference");
  expect(data.panels[0].actions.map((action) => action.label)).toEqual(["Rock", "Paper", "Scissors"]);
  expect(JSON.stringify(fixture)).toBe(before);
});

it("interpolates straight sampled segments with exact endpoints and probability coordinates", () => {
  const path = extractTrajectory(fixture).paths[0];
  expect(sampleAt(path, -1)).toMatchObject({ position: [0, 0, 0], iteration: 1, interpolated: false });
  expect(sampleAt(path, 9)).toMatchObject({ position: [1, 0, 0], probabilities: [0, 1, 0], interpolated: false });
  expect(sampleAt(path, 3)).toMatchObject({ position: [0.25, 0.25, 0], probabilities: [0.625, 0.125, 0.25], interpolated: true });
  expect(sampleAt(path, 5).interpolated).toBe(false);
  expect(sampleAt(path, 900).iteration).toBe(9);
  expect(sampleDescription(path, sampleAt(path, 3))).toContain("Approximate iteration 3");
  expect(sampleDescription(path, sampleAt(path, 9))).toContain("Hedge · Repeat 2");
});

it("decodes cube profiles and axis text without silently changing probability semantics", () => {
  const data = extractTrajectory({ layout: { scene: { xaxis: { title: { text: "P1: P(Dove)" } } } }, data: [
    { type: "scatter3d", meta: { role: "history", algorithm: "EXP3", run: 0 }, x: encoded("f8", [1, 0]), y: [0.5, 0.2], z: [0, 1], customdata: encoded("f8", [1, 1, 0.5, 0, 10, 0, 0.2, 1], "2,4") },
    { type: "scatter3d", meta: { role: "equilibrium" }, x: [0], y: [1], z: [0.5], hovertemplate: "Pure reference<br>P1:0<extra></extra>" },
  ] });
  expect(data.cube).toBe(true);
  expect(data.axes[0]).toBe("P1 · P(Dove)");
  expect(data.paths[0].points[1].probabilities).toEqual([0, 0.2, 1]);
  expect(data.references[0].label).toBe("Pure reference");
  expect(data.panels).toHaveLength(0);
});

it("fits line and polygon panels and clamps labels entirely inside small viewports", () => {
  const panel = extractTrajectory(fixture).panels[0];
  for (const point of panel.boundary) {
    const [x, y] = projectPanel(panel, point);
    expect(x).toBeGreaterThanOrEqual(55);
    expect(x).toBeLessThanOrEqual(425);
    expect(y).toBeGreaterThanOrEqual(55);
    expect(y).toBeLessThanOrEqual(310);
  }
  expect(clampLabel(-100, 900, 320, 200, 90, 18)).toEqual([8, 174]);
  expect(extractTrajectory({ data: [], layout: {} }).paths).toHaveLength(0);
});

it("names probabilities and refuses mismatched or unordered recorded samples", () => {
  const data = extractTrajectory(fixture);
  expect(sampleDescription(data.paths[0], data.paths[0].points.at(-1)!)).toContain("Paper: 1.000");
  const invalid = { ...fixture, data: [{ ...fixture.data[2], customdata: [[5, 1, 0], [1, 0, 1]] }] };
  expect(() => extractTrajectory(invalid)).toThrow(/sample/);
});

it("separates captions and coincident ticks while keeping each label inside the viewport", async () => {
  const { layoutLabels } = await import("./trajectoryData");
  const boxes = [{ x: 0, y: 0, width: 90, height: 18 }, { x: 0, y: 0, width: 90, height: 18 }, { x: 0, y: 0, width: 12, height: 18 }];
  const positions = layoutLabels(boxes, 320, 200);
  positions.forEach(([x, y], index) => {
    expect(x).toBeGreaterThanOrEqual(8);
    expect(x + boxes[index].width).toBeLessThanOrEqual(312);
    expect(y).toBeGreaterThanOrEqual(8);
    expect(y + boxes[index].height).toBeLessThanOrEqual(192);
  });
  expect(new Set(positions.map(String)).size).toBe(3);
});

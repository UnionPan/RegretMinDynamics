import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import TrajectoryCanvas from "./TrajectoryCanvas";
import type { PlotlyFigure } from "./types";

const cubeMock = vi.hoisted(() => ({ mount: vi.fn(), seek: vi.fn(), replay: vi.fn(), dispose: vi.fn(), visible: vi.fn(), reset: vi.fn() }));
vi.mock("./trajectoryCube", () => ({ mountCube: cubeMock.mount }));
const cubeFigure: PlotlyFigure = { layout: {}, data: [{
  type: "scatter3d", meta: { role: "history", algorithm: "Hedge", run: 0 }, x: [0, 1], y: [0, 1], z: [0, 1], customdata: [[1, 0, 0, 0], [10, 1, 1, 1]],
}] };
const figure: PlotlyFigure = { layout: {}, data: [
  { meta: { role: "history", algorithm: "Hedge", run: 0, player: 0 }, x: [0, 1], y: [0, 0], customdata: [[1, 1, 0], [10, 0, 1]], line: { color: "#3866e9" } },
  { meta: { role: "action_labels" }, x: [0, 1], y: [0, 0], text: ["Left", "Right"] },
  { meta: { role: "equilibrium" }, x: [0.5], y: [0], hovertemplate: "<extra>Mixed reference</extra>" },
] };

it("shows crisp SVG final policies without scheduling an idle animation loop", () => {
  const raf = vi.spyOn(window, "requestAnimationFrame");
  render(<TrajectoryCanvas figure={figure} replay={false} />);
  expect(screen.getByRole("img", { name: "Player 1 strategy trajectory" })).toBeVisible();
  expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  expect(screen.getByText(/Hedge · Repeat 1 · Recorded iteration 10/)).toBeInTheDocument();
  expect(screen.getByText("Equilibrium references")).toBeVisible();
  expect(raf).not.toHaveBeenCalled();
});

it("provides a scrubber and explicitly discloses interpolated playback", () => {
  vi.spyOn(window, "requestAnimationFrame").mockReturnValue(77);
  render(<TrajectoryCanvas figure={figure} replay />);
  const slider = screen.getByRole("slider", { name: "Playback iteration" });
  fireEvent.input(slider, { target: { value: "5.5" } });
  expect(screen.getByText("≈ iteration 5.5")).toBeVisible();
  expect(screen.getByLabelText(/Playback interpolates sampled points/)).toHaveAttribute("title", expect.stringContaining("recorded policies"));
  expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
});

it("handles missing histories without attempting to render misleading geometry", () => {
  render(<TrajectoryCanvas figure={{ data: [], layout: {} }} replay={false} />);
  expect(screen.getByText("No recorded trajectory is available.")).toBeVisible();
});

it("toggles all equilibrium references from one concise legend control", () => {
  const { container } = render(<TrajectoryCanvas figure={figure} replay={false} />);
  const button = screen.getByRole("button", { name: "Equilibrium references" });
  fireEvent.click(button);
  expect(button).toHaveAttribute("aria-pressed", "false");
  expect(container.querySelector('[data-reference]')).toHaveAttribute("visibility", "hidden");
  fireEvent.click(button);
  expect(container.querySelector('[data-reference]')).toHaveAttribute("visibility", "visible");
});

it("applies legend choices made while the GPU module is still loading", async () => {
  let finish: (value: unknown) => void = () => {};
  cubeMock.mount.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  render(<TrajectoryCanvas figure={cubeFigure} replay={false} />);
  await waitFor(() => expect(cubeMock.mount).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "Hedge" }));
  await act(async () => { finish({ seek: cubeMock.seek, setReplay: cubeMock.replay, dispose: cubeMock.dispose, setVisible: cubeMock.visible, reset: cubeMock.reset }); });
  expect(cubeMock.visible).toHaveBeenCalledWith("Hedge", false);
  fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
  expect(cubeMock.reset).toHaveBeenCalled();
});

it("disposes a cube that resolves after the view has unmounted", async () => {
  let finish: (value: unknown) => void = () => {};
  cubeMock.mount.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  const view = render(<TrajectoryCanvas figure={cubeFigure} replay={false} />);
  await waitFor(() => expect(cubeMock.mount).toHaveBeenCalled());
  view.unmount();
  const dispose = vi.fn();
  await act(async () => { finish({ dispose }); });
  expect(dispose).toHaveBeenCalledTimes(1);
});

it("uses fine continuous SVG paths and outlined cursors while retaining repeat coordinates", () => {
  const dotted = { ...figure, data: figure.data.map((trace, index) => index === 0 ? { ...trace, line: { color: "#3866e9", dash: "dot" } } : trace) };
  const { container } = render(<TrajectoryCanvas figure={dotted} replay={false} />);
  const path = container.querySelector('polyline[data-group="Hedge"]')!;
  expect(path.getAttribute("stroke-dasharray")).toBeNull();
  expect(Number(path.getAttribute("stroke-width"))).toBeLessThanOrEqual(1.5);
  expect(container.querySelector("circle")).toHaveAttribute("fill", "white");
  expect(container.querySelector("circle")).toHaveAttribute("stroke", "#3866e9");
});

it("draws fading SVG trails behind particles without showing future paths and restores history", () => {
  const frames: FrameRequestCallback[] = [];
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => { frames.push(callback); return frames.length; });
  const view = render(<TrajectoryCanvas figure={figure} replay />);
  const history = view.container.querySelector('polyline[data-history-key]')!;
  expect(history).toHaveAttribute("display", "none");
  act(() => frames.at(-1)!(0));
  act(() => frames.at(-1)!(12000 * 4 / 9));
  const head = view.container.querySelector("circle")!;
  const trails = Array.from(view.container.querySelectorAll('polyline[data-trail-key]'));
  expect(trails).toHaveLength(6);
  expect(trails.every((trail) => trail.getAttribute("display") !== "none")).toBe(true);
  const latest = trails.at(-1)!;
  expect(latest.getAttribute("points")?.split(" ").at(-1)).toBe(head.getAttribute("cx") + "," + head.getAttribute("cy"));
  expect(trails.map((trail) => Number(trail.getAttribute("stroke-opacity"))).at(-1)).toBeGreaterThan(Number(trails[0].getAttribute("stroke-opacity")));
  view.rerender(<TrajectoryCanvas figure={figure} replay={false} />);
  expect(history).toHaveAttribute("display", "block");
  expect(trails.every((trail) => trail.getAttribute("display") === "none")).toBe(true);
});

it("keeps thin complete trajectories after particles finish and past paths when paused", () => {
  const frames: FrameRequestCallback[] = [];
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => { frames.push(callback); return frames.length; });
  const { container } = render(<TrajectoryCanvas figure={figure} replay />);
  act(() => frames.at(-1)!(0));
  act(() => frames.at(-1)!(12000));
  expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
  const history = container.querySelector('polyline[data-history-key]')!;
  expect(history).toHaveAttribute("display", "block");
  expect(history.getAttribute("points")?.split(" ")).toHaveLength(2);
  expect(Number(history.getAttribute("stroke-width"))).toBeLessThanOrEqual(1.5);
  fireEvent.input(screen.getByRole("slider"), { target: { value: "5" } });
  const head = container.querySelector("circle")!;
  expect(history.getAttribute("points")?.split(" ").at(-1)).toBe(head.getAttribute("cx") + "," + head.getAttribute("cy"));
});

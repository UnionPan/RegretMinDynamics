import { act, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import PlotCanvas from "./PlotCanvas";
import type { PlotlyFigure } from "./types";

const runtime = vi.hoisted(() => ({ react: vi.fn(), addFrames: vi.fn().mockResolvedValue(undefined), animate: vi.fn().mockResolvedValue(undefined), purge: vi.fn(), Plots: { resize: vi.fn() } }));
vi.mock("plotly.js-basic-dist-min", () => ({ default: runtime }));

const oldFigure: PlotlyFigure = { data: [{ type: "scatter", name: "old" }], layout: {} };
const newFigure: PlotlyFigure = { data: [{ type: "scatter", name: "new" }], layout: {} };

beforeEach(() => {
  vi.clearAllMocks();
  runtime.react.mockResolvedValue(undefined);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
});

it("isolates a late Plotly render from the current figure", async () => {
  let finishOld: () => void = () => {};
  runtime.react.mockImplementationOnce((element: HTMLElement) => new Promise<void>((resolve) => {
    finishOld = () => { element.textContent = "old drawing"; resolve(); };
  }));
  runtime.react.mockImplementationOnce(async (element: HTMLElement) => { element.textContent = "new drawing"; });
  const workspace = render(<PlotCanvas figure={oldFigure} replay={false} />);
  await waitFor(() => expect(runtime.react).toHaveBeenCalledTimes(1));
  workspace.rerender(<PlotCanvas figure={newFigure} replay={false} />);
  await screen.findByText("new drawing");
  await act(async () => { finishOld(); });
  expect(screen.getByText("new drawing")).toBeVisible();
  expect(screen.queryByText("old drawing")).not.toBeInTheDocument();
});

it("memoizes unchanged figures and purges the graph on unmount", async () => {
  const workspace = render(<PlotCanvas figure={oldFigure} replay={false} />);
  await waitFor(() => expect(runtime.react).toHaveBeenCalledTimes(1));
  workspace.rerender(<PlotCanvas figure={oldFigure} replay={false} />);
  expect(runtime.react).toHaveBeenCalledTimes(1);
  workspace.unmount();
  expect(runtime.purge).toHaveBeenCalledTimes(1);
});

it("adds and plays bounded server frames only when replay is requested", async () => {
  const animated = { ...oldFigure, frames: [{ name: "0", data: [{ type: "scatter" }] }] };
  const workspace = render(<PlotCanvas figure={animated} replay={false} />);
  await waitFor(() => expect(runtime.react).toHaveBeenCalledTimes(1));
  expect(runtime.addFrames).not.toHaveBeenCalled();
  workspace.rerender(<PlotCanvas figure={animated} replay />);
  await waitFor(() => expect(runtime.animate).toHaveBeenCalledTimes(1));
  expect(runtime.addFrames).toHaveBeenCalledWith(expect.any(HTMLElement), animated.frames);
});

it("reports a graphics failure without losing the workspace", async () => {
  runtime.react.mockRejectedValueOnce(new Error("WebGL unavailable."));
  render(<PlotCanvas figure={oldFigure} replay={false} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("WebGL unavailable.");
});

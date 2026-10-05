import { act, fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import TrajectoryLoader from "./TrajectoryLoader";

const figure = { data: [], layout: {} };
it("keeps a failed trajectory chunk inside the chart and retries without recomputing", async () => {
  const load = vi.fn().mockRejectedValueOnce(new Error("Network unavailable")).mockResolvedValueOnce({ default: () => <div>Recovered trajectory</div> });
  render(<TrajectoryLoader figure={figure} replay={false} load={load} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load this view");
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByText("Recovered trajectory")).toBeVisible();
  expect(load).toHaveBeenCalledTimes(2);
});

it("ignores a loader that resolves after navigation away", async () => {
  let finish: (value: { default: () => null }) => void = () => {};
  const load = vi.fn(() => new Promise<{ default: () => null }>((resolve) => { finish = resolve; }));
  const renderer = vi.fn(() => null);
  const view = render(<TrajectoryLoader figure={figure} replay={false} load={load} />);
  view.unmount();
  await act(async () => { finish({ default: renderer }); });
  expect(renderer).not.toHaveBeenCalled();
});

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Mock } from "vitest";
import App from "./App";

const plotly = vi.hoisted(() => ({ react: vi.fn().mockResolvedValue(undefined), addFrames: vi.fn().mockResolvedValue(undefined), animate: vi.fn().mockResolvedValue(undefined), purge: vi.fn(), Plots: { resize: vi.fn() } }));
vi.mock("plotly.js-basic-dist-min", () => ({ default: plotly }));

const catalog = {
  games: [
    { name: "Pure Coordination", category: "Box Games", num_players: 3, action_labels: ["Action 0", "Action 1"], description: "Everyone earns one for even parity.", algorithms: ["Hedge", "EXP3"], equilibria: [[[1, 0], [1, 0], [1, 0]]], payoffs: [{ actions: [0, 0, 0], values: [1, 1, 1] }, { actions: [1, 0, 0], values: [0, 0, 0] }] },
    { name: "Rock Paper Scissors", category: "RPS Games", num_players: 2, action_labels: ["Rock", "Paper", "Scissors"], description: "A cyclic zero-sum game.", algorithms: ["Hedge", "Fictitious Play"], equilibria: [], payoffs: [{ actions: [0, 0], values: [0, 0] }] },
  ],
  algorithms: [{ name: "Hedge", description: "Exponential weights.", feedback: "Full feedback" }, { name: "EXP3", description: "Importance-weighted updates.", feedback: "Bandit feedback" }, { name: "Fictitious Play", description: "Best responses to historical frequencies.", feedback: "Full feedback" }],
  limits: { max_steps: 200000, max_runs: 5, max_iterations: 100000 },
};

let fetchMock: Mock<(input: string, options?: RequestInit) => Promise<ReturnType<typeof response>>>;
const response = (data: unknown, status = 200) => ({ ok: status < 400, status, json: async () => data, blob: async () => new Blob(["archive"], { type: "application/zip" }) });
const figure = { data: [{ type: "scatter", meta: { role: "history", algorithm: "Hedge", run: 0, player: 0 }, x: [0, 1], y: [0, 1], customdata: [[1, 1, 0], [1000, 0, 1]] }], layout: {} };

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  fetchMock = vi.fn(async (input: string, options?: RequestInit) => {
    const url = String(input);
    if (url === "/api/catalog") return response(catalog);
    if (url === "/api/experiments") return response({ id: "result-1", config: JSON.parse(String(options?.body)), summary: { run_count: 3, runtime_seconds: 0.12, final_nash_gap: 0.02, expected_regret: 0.01, mean_payoffs: [0.5, 0.5, 0.5] }, empirical_available: false, comparison: [] });
    if (url.includes("/figure")) return response(figure);
    if (url.includes("/download")) return response(null);
    throw new Error(`Unexpected request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
});

async function openWorkspace() {
  render(<App />);
  await screen.findByRole("button", { name: "Run experiment" });
}

describe("research workspace", () => {
  it("loads setup without automatically running or loading a plot", async () => {
    await openWorkspace();
    expect(screen.getByText("A game. A rule. A trajectory.")).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("Everyone earns one for even parity.")).not.toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(plotly.react).not.toHaveBeenCalled();
  });

  it("runs explicit seeded config and preserves completed results while editing setup", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    const request = fetchMock.mock.calls.find(([url]) => url === "/api/experiments");
    expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({ game_name: "Pure Coordination", algorithm_names: ["Hedge"], runs: 3, seed: 42 });
    fireEvent.change(screen.getByLabelText("Iterations"), { target: { value: "1500" } });
    expect(screen.getByText("Setup changed")).toBeVisible();
    expect(screen.getByText("0.020")).toBeVisible();
    expect(plotly.react).not.toHaveBeenCalled();
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/experiments")).toHaveLength(1);
  });

  it("changes figures without starting a new experiment and keeps replay opt-in", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    expect(fetchMock.mock.calls.find(([url]) => String(url).includes("/figure"))?.[0]).toContain("animate=false");
    await userEvent.click(screen.getByRole("tab", { name: "Diagnostics" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("kind=diagnostics"))).toBe(true));
    await userEvent.click(screen.getByRole("tab", { name: "Trajectory" }));
    await userEvent.click(screen.getByRole("button", { name: "Replay" }));
    await screen.findByRole("slider", { name: "Playback iteration" });
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("animate=true"))).toBe(false);
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/experiments")).toHaveLength(1);
  });

  it("shows explanations and interactive payoff exploration only in the Info drawer", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Info" }));
    expect(screen.getByRole("dialog")).toBeVisible();
    expect(screen.getByText("Everyone earns one for even parity.")).toBeVisible();
    await userEvent.click(screen.getByText("Payoff explorer"));
    expect(screen.getByLabelText("Player 1 action")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Player 1 action"), { target: { value: "1" } });
    expect(screen.getByTestId("payoff-p1")).toHaveTextContent("0");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("validates selections and public compute limits before sending work", async () => {
    await openWorkspace();
    fireEvent.change(screen.getByLabelText("Iterations"), { target: { value: "100000" } });
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    expect(screen.getByRole("alert")).toHaveTextContent("200,000");
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/experiments")).toHaveLength(0);
  });

  it("shows API errors without discarding a completed result", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    fetchMock.mockImplementationOnce(async () => response({ detail: "Experiment queue is full." }, 429));
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Experiment queue is full.");
    expect(screen.getByText("0.020")).toBeVisible();
  });

  it("requests ZIP bytes only after Export and releases the object URL", async () => {
    const create = vi.fn(() => "blob:test-download");
    const revoke = vi.fn();
    vi.stubGlobal("URL", class extends URL { static createObjectURL = create; static revokeObjectURL = revoke; });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/download"))).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Export" }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    expect(revoke).toHaveBeenCalledWith("blob:test-download");
  });

  it("switches families, chooses compatible methods and submits advanced settings", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "2 players" }));
    expect(screen.getByLabelText("Game")).toHaveValue("Rock Paper Scissors");
    await userEvent.click(screen.getByText("Hedge", { selector: "summary" }));
    await userEvent.click(screen.getByRole("button", { name: "Fictitious Play" }));
    await userEvent.click(screen.getByRole("button", { name: "Hedge" }));
    await userEvent.click(screen.getByText("Advanced settings"));
    fireEvent.change(screen.getByLabelText("Repeats"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "13" } });
    fireEvent.change(screen.getByLabelText("Initialization"), { target: { value: "uniform" } });
    fireEvent.change(screen.getByLabelText("Learning rate"), { target: { value: "0.3" } });
    fireEvent.change(screen.getByLabelText("Decay"), { target: { value: "-0.4" } });
    fireEvent.change(screen.getByLabelText("Temperature"), { target: { value: "0.2" } });
    fireEvent.change(screen.getByLabelText("Exploration"), { target: { value: "0.15" } });
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    const request = fetchMock.mock.calls.find(([url]) => url === "/api/experiments");
    expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({ game_name: "Rock Paper Scissors", algorithm_names: ["Fictitious Play"], runs: 2, seed: 13, initialization: "uniform", learning_rate: 0.3, decay: -0.4, temperature: 0.2, exploration: 0.15 });
    await userEvent.click(screen.getByRole("button", { name: "3 players" }));
    expect(screen.getByLabelText("Game")).toHaveValue("Pure Coordination");
    expect(screen.getByRole("heading", { name: "Rock Paper Scissors" })).toBeVisible();
    expect(screen.getByText("Setup changed")).toBeVisible();
  });

  it("keeps final-policy controls honest and fetches average policies without rerunning", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    await userEvent.click(screen.getByRole("button", { name: "Average" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("view=Time-average+policy"))).toBe(true));
    await userEvent.click(screen.getByRole("tab", { name: "Final policies" }));
    expect(screen.queryByRole("group", { name: "Policy view" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Diagnostics" }));
    fireEvent.change(screen.getByLabelText("Diagnostic metric"), { target: { value: "average_regret" } });
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("metric=average_regret"))).toBe(true));
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/experiments")).toHaveLength(1);
  });

  it("recovers from catalog and figure request failures", async () => {
    fetchMock.mockImplementationOnce(async () => response({ detail: "Catalog unavailable." }, 503));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Catalog unavailable.");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    await screen.findByRole("button", { name: "Run experiment" });
    const defaultFetch = fetchMock.getMockImplementation();
    fetchMock.mockImplementation(async (url, options) => String(url).includes("/figure") ? response({ detail: "Figure expired. Run a new experiment." }, 404) : defaultFetch!(url, options));
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Figure expired.");
    expect(screen.getByText("0.020")).toBeVisible();
  });

  it("never displays an earlier trajectory under a failed new figure selection", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    expect(screen.getByRole("region", { name: "Strategy trajectories" })).toBeInTheDocument();
    fetchMock.mockImplementationOnce(async () => response({ detail: "Figure expired." }, 410));
    await userEvent.click(screen.getByRole("tab", { name: "Diagnostics" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Figure expired.");
    expect(screen.queryByRole("region", { name: "Strategy trajectories" })).not.toBeInTheDocument();
    expect(screen.getByText("0.020")).toBeVisible();
  });

  it("displays empirical frequencies only when the experiment supplies them", async () => {
    await openWorkspace();
    const defaultFetch = fetchMock.getMockImplementation();
    fetchMock.mockImplementation(async (url, options) => {
      const received = await defaultFetch!(url, options);
      if (url !== "/api/experiments") return received;
      return response({ ...(await received.json() as Record<string, unknown>), empirical_available: true });
    });
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    expect(screen.getByRole("button", { name: "Empirical" })).toHaveAttribute("title", expect.stringContaining("Fictitious Play"));
    await userEvent.click(screen.getByRole("button", { name: "Empirical" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("view=Empirical+frequencies"))).toBe(true));
  });

  it("retains small nonzero Nash gaps instead of rounding them to a certificate", async () => {
    await openWorkspace();
    const defaultFetch = fetchMock.getMockImplementation();
    fetchMock.mockImplementation(async (url, options) => {
      const received = await defaultFetch!(url, options);
      if (url !== "/api/experiments") return received;
      const result = await received.json() as { summary: Record<string, unknown> };
      return response({ ...result, summary: { ...result.summary, final_nash_gap: 0.00000034, expected_regret: -0.00000012 } });
    });
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    expect(await screen.findByText("3.4e-7")).toBeVisible();
    expect(screen.getByText("-1.2e-7")).toBeVisible();
  });

  it("connects the chart panel to accessible tabs with keyboard navigation", async () => {
    await openWorkspace();
    await userEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByRole("region", { name: "Strategy trajectories" });
    const trajectory = screen.getByRole("tab", { name: "Trajectory" });
    const diagnostics = screen.getByRole("tab", { name: "Diagnostics" });
    const policies = screen.getByRole("tab", { name: "Final policies" });
    const panel = screen.getByRole("tabpanel");
    expect(trajectory).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", trajectory.id);
    trajectory.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(diagnostics).toHaveFocus();
    expect(diagnostics).toHaveAttribute("aria-selected", "true");
    expect(trajectory).toHaveAttribute("tabindex", "-1");
    await userEvent.keyboard("{End}");
    expect(policies).toHaveFocus();
    await userEvent.keyboard("{Home}");
    expect(trajectory).toHaveFocus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(policies).toHaveFocus();
  });
});

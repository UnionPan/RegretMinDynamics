import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { initialConfig } from "./config";
import InfoDrawer from "./InfoDrawer";
import type { GameSpec } from "./types";

function game(actions: number): GameSpec {
  return { name: "Cyclic example", category: "RPS Games", num_players: 2, action_labels: Array.from({ length: actions }, (_, index) => `A${index}`), description: "A small example game.", algorithms: ["Hedge"], equilibria: [[[1, 0], [0, 1]]], payoffs: [{ actions: [0, 0], values: [1, -1] }] };
}

it.each([
  [2, "A two-action policy lies on a line between its actions."],
  [3, "Triangle coordinates identify the three action probabilities."],
  [4, "The polygon projection is not one-to-one."],
])("explains the correct %i-action geometry behind a disclosure", async (actions, expected) => {
  const selected = game(Number(actions));
  render(<InfoDrawer game={selected} config={initialConfig(selected)} algorithms={[]} onClose={() => {}} />);
  await userEvent.click(screen.getByText("How to read"));
  expect(screen.getByText(String(expected), { exact: false })).toBeVisible();
});

it("shows selected methods and exact joint-action utilities without a table", async () => {
  const selected = game(2);
  render(<InfoDrawer game={selected} config={initialConfig(selected)} algorithms={[{ name: "Hedge", description: "Exponentiated utility scores.", feedback: "Full feedback" }, { name: "EXP3", description: "Other method.", feedback: "Bandit feedback" }]} onClose={() => {}} />);
  await userEvent.click(screen.getByText("Methods & metrics"));
  expect(screen.getByText("Exponentiated utility scores.")).toBeVisible();
  expect(screen.queryByText("Other method.")).not.toBeInTheDocument();
  await userEvent.click(screen.getByText("Payoff explorer"));
  expect(screen.getByTestId("payoff-p2")).toHaveTextContent("-1");
  fireEvent.change(screen.getByLabelText("Player 2 action"), { target: { value: "1" } });
  expect(screen.getByTestId("payoff-p2")).toHaveTextContent("-");
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});

it("traps focus, restores the opener and closes only through dismiss controls", async () => {
  const selected = game(2);
  const onClose = vi.fn();
  const opener = document.createElement("button");
  document.body.append(opener);
  opener.focus();
  const rectangles = vi.spyOn(HTMLElement.prototype, "getClientRects").mockReturnValue([new DOMRect(0, 0, 10, 10)] as unknown as DOMRectList);
  const drawer = render(<InfoDrawer game={selected} config={initialConfig(selected)} algorithms={[]} onClose={onClose} />);
  const close = screen.getByRole("button", { name: "Close Info" });
  expect(close).toHaveFocus();
  await userEvent.keyboard("{Shift>}{Tab}{/Shift}");
  expect(screen.getByText("Reproducibility")).toHaveFocus();
  await userEvent.keyboard("{Tab}");
  expect(close).toHaveFocus();
  await userEvent.click(screen.getByRole("heading", { name: "Cyclic example" }));
  expect(onClose).not.toHaveBeenCalled();
  fireEvent.click(drawer.container.querySelector(".drawer-backdrop")!);
  expect(onClose).toHaveBeenCalledTimes(1);
  await userEvent.click(close);
  expect(onClose).toHaveBeenCalledTimes(2);
  drawer.unmount();
  expect(opener).toHaveFocus();
  opener.remove();
  rectangles.mockRestore();
});

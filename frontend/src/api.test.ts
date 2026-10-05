import { beforeEach, describe, expect, it, vi } from "vitest";
import { downloadExperiment, errorMessage, getCatalog, getFigure } from "./api";

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => { fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock); });

describe("API boundaries", () => {
  it("encodes figure identifiers and explicit policy choices", async () => {
    fetchMock.mockResolvedValue({ ok: true, json: async () => ({ data: [], layout: {} }) });
    const signal = new AbortController().signal;
    await getFigure("id/with space", "trajectory", "Time-average policy", false, "nash_gap", signal);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toContain("id%2Fwith%20space");
    expect(url).toContain("view=Time-average+policy");
    expect(options.signal).toBe(signal);
  });
  it.each([false, true])("handles non-JSON and structured errors: %#", async (structured) => {
    fetchMock.mockResolvedValue({ ok: false, status: 503, json: async () => { if (!structured) throw new Error("HTML page"); return { detail: ["invalid"] }; } });
    await expect(getCatalog(new AbortController().signal)).rejects.toThrow("Request failed (503)");
  });
  it("retains readable API errors and formats unknown failures", () => {
    expect(errorMessage(new Error("Connection lost."))).toBe("Connection lost.");
    expect(errorMessage({ message: "unknown" })).toBe("The request could not be completed.");
  });
  it("does not start a browser download after cancellation", async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValue({ ok: true, blob: async () => { controller.abort(); return new Blob(["zip"]); } });
    await expect(downloadExperiment("id", controller.signal)).resolves.toBeUndefined();
    expect(document.querySelector("a[download]")).toBeNull();
  });
});

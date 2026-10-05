import type { Catalog, DiagnosticMetric, ExperimentConfig, ExperimentResult, FigureKind, PlotlyFigure, PolicyView } from "./types";

async function request(url: string, options: RequestInit): Promise<Response> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed (${response.status}). Please try again.`;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string") message = body.detail;
    } catch { /* A proxy can return a non-JSON error page. */ }
    throw new Error(message);
  }
  return response;
}

export async function getCatalog(signal: AbortSignal): Promise<Catalog> {
  return (await request("/api/catalog", { signal })).json();
}

export async function runExperiment(config: ExperimentConfig, signal: AbortSignal): Promise<ExperimentResult> {
  return (await request("/api/experiments", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config), signal,
  })).json();
}

export async function getFigure(
  id: string, kind: FigureKind, view: PolicyView, animate: boolean,
  metric: DiagnosticMetric, signal: AbortSignal,
): Promise<PlotlyFigure> {
  const query = new URLSearchParams({ kind, view, animate: String(animate), metric });
  return (await request(`/api/experiments/${encodeURIComponent(id)}/figure?${query}`, { signal })).json();
}

export async function downloadExperiment(id: string, signal: AbortSignal): Promise<void> {
  const response = await request(`/api/experiments/${encodeURIComponent(id)}/download`, { signal });
  const blob = await response.blob();
  if (signal.aborted) return;
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  try {
    anchor.href = url;
    anchor.download = `regret-lab-${id}.zip`;
    document.body.append(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(url);
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request could not be completed.";
}

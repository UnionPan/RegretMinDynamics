import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const budgets = [
  { name: "Initial JavaScript", pattern: /^index-.*\.js$/, bytes: 100_000 },
  { name: "Optional 2D Plotly", pattern: /^plotly-basic\.min-.*\.js$/, bytes: 450_000 },
  { name: "Optional trajectory controls", pattern: /^TrajectoryCanvas-.*\.js$/, bytes: 50_000 },
  { name: "Optional 3D trajectory", pattern: /^trajectoryCube-.*\.js$/, bytes: 250_000 },
];

export async function checkBundles(directory = "dist/assets") {
  const files = await readdir(directory);
  const report = [];
  for (const budget of budgets) {
    const matching = files.filter((file) => budget.pattern.test(file));
    if (matching.length !== 1) throw new Error(`Expected one ${budget.name} bundle, found ${matching.length}.`);
    const bytes = gzipSync(await readFile(resolve(directory, matching[0]))).length;
    if (bytes > budget.bytes) throw new Error(`${budget.name}: ${bytes} gzip bytes exceeds ${budget.bytes}.`);
    report.push(`${budget.name}: ${(bytes / 1000).toFixed(1)} KB gzip / ${budget.bytes / 1000} KB`);
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  for (const line of await checkBundles()) console.log(line);
}

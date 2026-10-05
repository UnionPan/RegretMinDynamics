import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8512" } },
  // Lazy chart and trajectory bundles have separate gzip budgets.
  build: { chunkSizeWarningLimit: 1800 },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/testSetup.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts", "src/**/*.tsx"],
      exclude: ["src/**/*.test.*", "src/testSetup.ts", "src/main.tsx", "src/declarations.d.ts"],
      thresholds: { statements: 80, branches: 80, functions: 80, lines: 80 },
    },
  },
});

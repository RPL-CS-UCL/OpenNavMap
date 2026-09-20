import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const backend = process.env.VITE_BACKEND_ORIGIN ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: true },
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
  build: {
    rollupOptions: {
      output: {
        advancedChunks: {
          groups: [
            { name: "three", test: /node_modules[\\/](three|@react-three)[\\/]/ },
            { name: "charts", test: /node_modules[\\/]recharts[\\/]/ },
          ],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    alias: {
      // jsdom reports zero sizes, so the real library writes "NaN%" flex-basis and jsdom's CSS parser throws.
      "react-resizable-panels": path.resolve(__dirname, "./src/test/resizable-mock.tsx"),
    },
  },
});

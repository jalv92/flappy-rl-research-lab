/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Integration plumbing provided by the backend side: the dev server proxies the API so the app can use
// same-origin relative URLs (/api/..., /ws/game) in both `vite dev` and the FastAPI-served production build.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  // Preserve the real stylesheet for the desktop minimum-size regression test.
  test: { environment: "jsdom", globals: true, css: { include: [/styles\.css\?raw$/] } },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Builds the SPA into ../static, which the Flask app serves. In dev, `npm run
// dev` runs Vite on :5173 and proxies the API to Flask on :8080.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8080",
    },
  },
});

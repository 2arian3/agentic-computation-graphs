import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the API to the FastAPI backend on :8100 (the model server is on
// :8000, so the app deliberately avoids that port). Production build is served by
// FastAPI itself from webapp/frontend/dist.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8100",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

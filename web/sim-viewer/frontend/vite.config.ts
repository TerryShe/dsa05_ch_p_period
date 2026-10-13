import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies the API to the backend (README §8); both stay on 127.0.0.1.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend talks to the FastAPI backend (default http://localhost:8000).
// Override with VITE_API_URL in .env for a deployed backend.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true },
});

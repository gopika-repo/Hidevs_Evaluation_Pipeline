import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => {
  // Load env variables based on the current mode (e.g. development, production)
  const env = loadEnv(mode, process.cwd());

  // Hard production build guard: fail fast if VITE_API_BASE_URL is missing
  if (command === 'build') {
    const apiBaseUrl = (env.VITE_API_BASE_URL || '').trim();
    if (!apiBaseUrl) {
      throw new Error(
        'BUILD ERROR: VITE_API_BASE_URL is not configured for the production build. ' +
        'Please define VITE_API_BASE_URL in your environment variables/deployment settings.'
      );
    }
  }

  return {
    plugins: [react()],
  }
})

/**
 * api.ts — Centralized API configuration module (Phase 10D)
 *
 * All backend calls MUST use `getApiBaseUrl()` from this module.
 * No component should have its own fallback to localhost.
 *
 * Rules:
 * - Development (MODE = 'development'): Falls back to http://localhost:8000.
 * - Production  (MODE != 'development'): VITE_API_BASE_URL MUST be set.
 *   If it is missing, throws a clear, actionable error.
 */

export function getApiBaseUrl(): string {
    const raw = import.meta.env.VITE_API_BASE_URL ?? '';
    const url = raw.trim();

    if (url) {
        // Strip trailing slash for consistent path concatenation
        return url.replace(/\/+$/, '');
    }

    // No VITE_API_BASE_URL provided
    if (import.meta.env.MODE !== 'development') {
        throw new Error(
            'VITE_API_BASE_URL is not configured. ' +
            'Set VITE_API_BASE_URL to your backend API URL in the deployment environment.'
        );
    }

    // Development fallback only
    return 'http://localhost:8000';
}

/**
 * Pre-resolved API base URL.
 * Throws eagerly in production if VITE_API_BASE_URL is missing.
 */
export const API_BASE_URL = getApiBaseUrl;

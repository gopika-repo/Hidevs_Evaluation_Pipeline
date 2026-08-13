import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { getApiBaseUrl } from './api';

describe('API Configuration Guard Tests (Phase 10D)', () => {

    beforeEach(() => {
        // Reset env variables
        vi.stubEnv('VITE_API_BASE_URL', '');
        vi.stubEnv('MODE', 'production');
    });

    afterEach(() => {
        vi.unstubAllEnvs();
    });

    it('should fall back to localhost in development mode if VITE_API_BASE_URL is missing', () => {
        vi.stubEnv('MODE', 'development');
        vi.stubEnv('VITE_API_BASE_URL', '');
        expect(getApiBaseUrl()).toBe('http://localhost:8000');
    });

    it('should return trimmed configured VITE_API_BASE_URL if present in development mode', () => {
        vi.stubEnv('MODE', 'development');
        vi.stubEnv('VITE_API_BASE_URL', 'https://dev.example.com/api/');
        expect(getApiBaseUrl()).toBe('https://dev.example.com/api');
    });

    it('should return trimmed configured VITE_API_BASE_URL if present in production mode', () => {
        vi.stubEnv('MODE', 'production');
        vi.stubEnv('VITE_API_BASE_URL', 'https://prod.example.com/api/ ');
        expect(getApiBaseUrl()).toBe('https://prod.example.com/api');
    });

    it('should throw Error if VITE_API_BASE_URL is missing/empty in production mode', () => {
        vi.stubEnv('MODE', 'production');
        vi.stubEnv('VITE_API_BASE_URL', '   ');
        expect(() => getApiBaseUrl()).toThrowError(
            'VITE_API_BASE_URL is not configured.'
        );
    });
});

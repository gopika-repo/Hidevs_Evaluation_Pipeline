import { describe, it, expect } from 'vitest';
import { renderScoreDisplay } from './App';

// Helper to extract text representation from React JSX element
function getElementText(element: any): string {
    if (element === null || element === undefined) return '';
    if (typeof element === 'string' || typeof element === 'number') {
        return String(element);
    }
    if (Array.isArray(element)) {
        return element.map(getElementText).join('');
    }
    if (element.props && element.props.children !== undefined) {
        return getElementText(element.props.children);
    }
    return '';
}

describe('renderScoreDisplay - Phase 10 Score Integrity', () => {
    it('{status:"success", score:20} -> 20 / 20', () => {
        const res = renderScoreDisplay(20, 20, 'success');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('20 / 20');
    });

    it('{status:"success", score:0} -> 0 / 20', () => {
        const res = renderScoreDisplay(0, 20, 'success');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('0 / 20');
    });

    it('{status:"success", score:null} -> N/A', () => {
        const res = renderScoreDisplay(null, 20, 'success');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('N/A');
    });

    it('{status:"timeout", score:null} -> TIMED OUT', () => {
        const res = renderScoreDisplay(null, 20, 'timeout');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('TIMED OUT');
    });

    it('{status:"failed", score:null} -> FAILED', () => {
        const res = renderScoreDisplay(null, 20, 'failed');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('FAILED');
    });

    it('{status:"unavailable", score:null} -> UNAVAILABLE', () => {
        const res = renderScoreDisplay(null, 20, 'unavailable');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('UNAVAILABLE');
    });

    it('{status:"invalid_output", score:null} -> INVALID OUTPUT', () => {
        const res = renderScoreDisplay(null, 20, 'invalid_output');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('INVALID OUTPUT');
    });

    it('{status:"not_applicable", score:null} -> NOT APPLICABLE', () => {
        const res = renderScoreDisplay(null, 20, 'not_applicable');
        const text = getElementText(res).replace(/\s+/g, ' ').trim();
        expect(text).toBe('NOT APPLICABLE');
    });
});

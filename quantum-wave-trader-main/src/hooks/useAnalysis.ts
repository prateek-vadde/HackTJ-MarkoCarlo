import { useState, useCallback } from 'react';
import type { AnalysisResponse, AppStatus } from '../types/analysis';

export function useAnalysis() {
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [status, setStatus] = useState<AppStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (date: string) => {
    setStatus('analyzing');
    setError(null);
    setData(null);

    try {
      const res = await fetch('http://localhost:8000/analyze-paths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || errBody.message || res.statusText);
      }

      const json: AnalysisResponse = await res.json();
      setData(json);
      setStatus('loaded');
    } catch (e: any) {
      setError(e.message || 'Unknown error');
      setStatus('error');
    }
  }, []);

  return { data, status, error, analyze };
}

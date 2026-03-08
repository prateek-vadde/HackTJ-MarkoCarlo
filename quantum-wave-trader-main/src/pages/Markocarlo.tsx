import { useState, useCallback } from 'react';
import Header from '../components/Header';
import PresetButtons from '../components/PresetButtons';
import AnalyzeBar from '../components/AnalyzeBar';
import MetricCards from '../components/MetricCards';
import PriceChart from '../components/PriceChart';
import WavefunctionChart from '../components/WavefunctionChart';
import CurvatureChart from '../components/CurvatureChart';
import PathDivergenceBar from '../components/PathDivergenceBar';
import EmptyState from '../components/EmptyState';
import { useAnalysis } from '../hooks/useAnalysis';

export default function Markocarlo() {
  const [date, setDate] = useState('');
  const { data, status, error, analyze } = useAnalysis();

  const handleAnalyze = useCallback(() => {
    if (date) analyze(date);
  }, [date, analyze]);

  const handlePreset = useCallback((d: string) => {
    setDate(d);
    analyze(d);
  }, [analyze]);

  return (
    <div className="min-h-screen bg-background relative">
      {/* Loading bar */}
      {status === 'analyzing' && (
        <div className="fixed top-0 left-0 right-0 h-1 bg-primary animate-progress-pulse z-50" />
      )}

      <div className="max-w-[1400px] mx-auto px-6">
        <Header status={status} />
        <PresetButtons onSelect={handlePreset} />
        <AnalyzeBar date={date} onDateChange={setDate} onAnalyze={handleAnalyze} disabled={status === 'analyzing' || !date} />

        {/* Loading state */}
        {status === 'analyzing' && (
          <p className="text-center text-muted-foreground py-20">COMPUTING GEOMETRIC MANIFOLD…</p>
        )}

        {/* Error state */}
        {status === 'error' && error && (
          <div className="border border-destructive rounded-lg p-6 mb-6" style={{ backgroundColor: '#1a0000' }}>
            <p className="text-destructive text-sm">ANALYSIS FAILED — {error}</p>
          </div>
        )}

        {/* Empty state */}
        {status === 'idle' && <EmptyState onSelect={handlePreset} />}

        {/* Results */}
        {status === 'loaded' && data && (
          <>
            <MetricCards data={data} />
            <PriceChart data={data} />

            <div className="grid grid-cols-1 lg:grid-cols-[55%_45%] gap-4 mb-6">
              <WavefunctionChart data={data} />
              <CurvatureChart data={data} />
            </div>

            <PathDivergenceBar value={data.path_divergence} />

            {/* MC Metadata */}
            <p className="text-[11px] text-muted-foreground mb-8">
              Monte Carlo: {data.n_accepted_paths} accepted paths · Acceptance rate: {(data.acceptance_rate * 100).toFixed(1)}% · Geometric manifold walker converged
            </p>
          </>
        )}

        {/* Footer */}
        <footer className="text-center py-8 text-[11px]" style={{ color: '#333333' }}>
          MARKOCARLO · Quantum Geometric Market Analysis · Built at HackTJ 2026
        </footer>
      </div>
    </div>
  );
}

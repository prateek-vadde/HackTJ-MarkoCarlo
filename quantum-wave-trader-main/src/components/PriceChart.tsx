import { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import type { AnalysisResponse } from '../types/analysis';

interface PriceChartProps {
  data: AnalysisResponse;
}

const DARK_LAYOUT: any = {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#0a0a0a',
  font: { family: 'Inter, sans-serif', color: '#888888', size: 11 },
  margin: { l: 60, r: 20, t: 10, b: 40 },
  legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(17,17,17,0.8)', font: { color: '#ffffff', size: 8 }, tracegroupgap: 2 },
};

function findClosestPaths(mcPaths: number[][], actual: number[], topN = 1): number[] {
  if (!actual?.length || !mcPaths?.length) return [];
  const scored = mcPaths.map((path, idx) => {
    const len = Math.min(path.length, actual.length);
    let sse = 0;
    for (let i = 0; i < len; i++) sse += (path[i] - actual[i]) ** 2;
    return { idx, sse };
  });
  scored.sort((a, b) => a.sse - b.sse);
  return scored.slice(0, topN).map(s => s.idx);
}

type AnimPhase = 'paths' | 'bands' | 'actual' | 'done';

export default function PriceChart({ data }: PriceChartProps) {
  const boundaryDate = data.dates[data.dates.length - 1];
  const lastPrice = data.prices[data.prices.length - 1];
  const nForward = data.forward_dates?.length || 0;

  const closestIdxs = new Set(
    findClosestPaths(data.mc_paths || [], data.actual_forward_prices || [])
  );
  const closestPaths = (data.mc_paths || []).filter((_, i) => closestIdxs.has(i));

  // Animation state
  const [phase, setPhase] = useState<AnimPhase>('paths');
  const [drawStep, setDrawStep] = useState(0); // how many forward points to show on green paths
  const animRef = useRef<number | null>(null);
  const dataIdRef = useRef<string>('');

  // Reset animation when data changes
  const dataId = data.dates[0] + data.dates[data.dates.length - 1];
  useEffect(() => {
    if (dataIdRef.current !== dataId) {
      dataIdRef.current = dataId;
      setPhase('paths');
      setDrawStep(0);
    }
  }, [dataId]);

  // Animation sequencer
  useEffect(() => {
    if (phase === 'paths') {
      if (drawStep < nForward) {
        animRef.current = window.setTimeout(() => setDrawStep(s => s + 1), 120);
      } else {
        animRef.current = window.setTimeout(() => setPhase('bands'), 300);
      }
    } else if (phase === 'bands') {
      animRef.current = window.setTimeout(() => setPhase('actual'), 800);
    } else if (phase === 'actual') {
      animRef.current = window.setTimeout(() => setPhase('done'), 100);
    }
    return () => { if (animRef.current) clearTimeout(animRef.current); };
  }, [phase, drawStep, nForward]);

  // --- Build traces based on animation phase ---

  const traces: any[] = [
    // Always show observed white line
    {
      x: data.dates,
      y: data.prices,
      mode: 'lines',
      name: 'Observed',
      line: { color: '#ffffff', width: 2 },
    },
  ];

  // Spaghetti MC paths — show after bands phase
  if (phase === 'bands' || phase === 'actual' || phase === 'done') {
    (data.mc_paths || [])
      .filter((_, i) => !closestIdxs.has(i))
      .forEach((path, i) => {
        traces.push({
          x: data.forward_dates,
          y: path,
          mode: 'lines',
          line: { color: 'rgba(0, 212, 255, 0.15)', width: 0.5 },
          showlegend: i === 0,
          name: i === 0 ? 'MC Paths' : undefined,
          hoverinfo: 'skip',
        });
      });
  }

  // Green closest paths — animate drawing point by point
  const greenDates = [data.dates[data.dates.length - 1], ...data.forward_dates.slice(0, drawStep)];
  closestPaths.forEach((path, i) => {
    traces.push({
      x: greenDates,
      y: [lastPrice, ...path.slice(0, drawStep)],
      mode: 'lines',
      line: { color: '#00ff88', width: 1.8 },
      showlegend: i === 0,
      name: i === 0 ? 'Closest Paths' : undefined,
      hoverinfo: 'skip',
    });
  });

  // Prediction band + mean — appear after paths finish drawing
  if (phase === 'bands' || phase === 'actual' || phase === 'done') {
    traces.push({
      x: [...data.forward_dates, ...data.forward_dates.slice().reverse()],
      y: [...data.predicted_upper, ...data.predicted_lower.slice().reverse()],
      fill: 'toself',
      fillcolor: 'rgba(0, 212, 255, 0.12)',
      line: { color: 'rgba(0, 212, 255, 0.3)', width: 1 },
      name: 'Prediction Band (25-75%)',
      showlegend: true,
    });
    traces.push({
      x: data.forward_dates,
      y: data.predicted_mean,
      mode: 'lines',
      name: 'Predicted Mean',
      line: { color: '#00d4ff', width: 2, dash: 'dash' },
    });
  }

  // Orange actual line — appears last
  if ((phase === 'actual' || phase === 'done') && data.actual_forward_dates?.length) {
    traces.push({
      x: data.actual_forward_dates,
      y: data.actual_forward_prices,
      mode: 'lines',
      name: 'Actual (Ground Truth)',
      line: { color: '#ffa500', width: 2.5 },
    });
  }

  const layout: any = {
    ...DARK_LAYOUT,
    height: 400,
    xaxis: { type: 'date', gridcolor: '#1a1a1a', range: [data.dates[0], data.forward_dates[data.forward_dates.length - 1]] },
    yaxis: { gridcolor: '#1a1a1a', tickprefix: '$' },
    shapes: [
      {
        type: 'line',
        x0: boundaryDate,
        x1: boundaryDate,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: { color: 'rgba(255,255,255,0.3)', width: 1 },
      },
    ],
    annotations: [
      {
        x: boundaryDate,
        y: 1,
        yref: 'paper',
        text: 'PREDICTION START',
        showarrow: false,
        font: { color: 'rgba(255,255,255,0.5)', size: 9 },
        xanchor: 'left',
        xshift: 5,
      },
    ],
  };

  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-2">
        <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Price Trajectory & Geometric Prediction</p>
        {data.fixed_node_active && (
          <span className="text-[9px] uppercase tracking-wider font-mono px-2 py-0.5 rounded bg-red-900/40 text-red-400 border border-red-800/50">
            Fixed-Node Active — VIX Backwardation (ratio {data.vix_ratio.toFixed(3)})
          </span>
        )}
      </div>
      <Plot data={traces} layout={layout} config={{ responsive: true }} className="w-full" />
    </div>
  );
}

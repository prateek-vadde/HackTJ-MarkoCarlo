import Plot from 'react-plotly.js';
import type { AnalysisResponse } from '../types/analysis';

interface CurvatureChartProps {
  data: AnalysisResponse;
}

function mean(arr: number[]) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[]) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

export default function CurvatureChart({ data }: CurvatureChartProps) {
  const threshold = mean(data.curvature) + std(data.curvature);

  const traces: any[] = [
    {
      x: data.dates,
      y: data.curvature,
      type: 'scatter',
      fill: 'tozeroy',
      line: { color: '#ff4444', width: 1.5 },
      fillcolor: 'rgba(255, 68, 68, 0.2)',
      name: 'Geometric Curvature',
    },
  ];

  const layout: any = {
    paper_bgcolor: '#0a0a0a',
    plot_bgcolor: '#0a0a0a',
    font: { family: 'Inter, sans-serif', color: '#888888', size: 11 },
    margin: { l: 60, r: 10, t: 10, b: 40 },
    height: 280,
    xaxis: { type: 'date', gridcolor: '#1a1a1a' },
    yaxis: { gridcolor: '#1a1a1a', exponentformat: 'e' },
    legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(17,17,17,0.8)', font: { color: '#ffffff', size: 10 } },
    shapes: [
      {
        type: 'line',
        x0: data.dates[0],
        x1: data.dates[data.dates.length - 1],
        y0: threshold,
        y1: threshold,
        line: { color: '#ff9500', width: 1, dash: 'dashdot' },
      },
    ],
    annotations: [
      {
        x: data.dates[Math.floor(data.dates.length * 0.85)],
        y: threshold,
        text: 'CRISIS THRESHOLD',
        showarrow: false,
        font: { color: '#ff9500', size: 10 },
        yshift: 12,
      },
    ],
  };

  return (
    <div>
      <p className="text-[11px] uppercase tracking-wider text-muted-foreground mb-2">Fubini-Study Curvature</p>
      <Plot data={traces} layout={layout} config={{ responsive: true }} className="w-full" />
    </div>
  );
}

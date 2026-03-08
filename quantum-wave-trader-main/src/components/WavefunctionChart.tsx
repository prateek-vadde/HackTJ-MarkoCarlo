import Plot from 'react-plotly.js';
import type { AnalysisResponse } from '../types/analysis';

interface WavefunctionChartProps {
  data: AnalysisResponse;
}

export default function WavefunctionChart({ data }: WavefunctionChartProps) {
  const traces: any[] = [
    {
      x: data.dates,
      y: data.psi_real,
      mode: 'lines',
      name: 'ψ Real',
      line: { color: '#00d4ff', width: 1.5 },
    },
    {
      x: data.dates,
      y: data.psi_imag,
      mode: 'lines',
      name: 'ψ Imaginary',
      line: { color: '#7b61ff', width: 1.5 },
    },
  ];

  const layout: any = {
    paper_bgcolor: '#0a0a0a',
    plot_bgcolor: '#0a0a0a',
    font: { family: 'Inter, sans-serif', color: '#888888', size: 11 },
    margin: { l: 50, r: 10, t: 10, b: 40 },
    height: 280,
    xaxis: { type: 'date', gridcolor: '#1a1a1a' },
    yaxis: { gridcolor: '#1a1a1a', title: { text: 'Amplitude', font: { size: 10 } } },
    legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(17,17,17,0.8)', font: { color: '#ffffff', size: 10 } },
  };

  return (
    <div>
      <p className="text-[11px] uppercase tracking-wider text-muted-foreground mb-2">Market Wavefunction ψ</p>
      <Plot data={traces} layout={layout} config={{ responsive: true }} className="w-full" />
    </div>
  );
}

import type { AnalysisResponse } from '../types/analysis';

interface MetricCardsProps {
  data: AnalysisResponse;
}

function thresholdColor(val: number) {
  if (val > 0.6) return 'text-destructive';
  if (val > 0.3) return 'text-warning';
  return 'text-success';
}

function MetricCard({ label, value, colorClass }: { label: string; value: string; colorClass: string }) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 flex-1 min-w-[140px]">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">{label}</p>
      <p className={`font-mono text-xl font-semibold ${colorClass}`}>{value}</p>
    </div>
  );
}

export default function MetricCards({ data }: MetricCardsProps) {
  return (
    <div className="flex flex-wrap gap-3 mb-6">
      <MetricCard
        label="Crisis Probability"
        value={`${(data.crisis_prob * 100).toFixed(1)}%`}
        colorClass={thresholdColor(data.crisis_prob)}
      />
      <MetricCard
        label="Risk Score"
        value={data.risk_score.toExponential(2)}
        colorClass={thresholdColor(data.crisis_prob)}
      />
      <MetricCard
        label="Berry Magnitude"
        value={data.berry_magnitude.toExponential(2)}
        colorClass="text-primary"
      />
      <MetricCard
        label="Expected Return"
        value={`${data.expected_return >= 0 ? '+' : ''}${(data.expected_return * 100).toFixed(2)}%`}
        colorClass={data.expected_return >= 0 ? 'text-success' : 'text-destructive'}
      />
      <MetricCard
        label="Expected Drawdown"
        value={`${(data.expected_drawdown * 100).toFixed(2)}%`}
        colorClass="text-destructive"
      />
      <MetricCard
        label="Worst Case (5th %ile)"
        value={`${(data.worst_case_drawdown * 100).toFixed(2)}%`}
        colorClass="text-destructive"
      />
      <MetricCard
        label="Crisis Path Fraction"
        value={`${(data.crisis_path_fraction * 100).toFixed(0)}%`}
        colorClass={data.crisis_path_fraction > 0.3 ? 'text-destructive' : data.crisis_path_fraction > 0.1 ? 'text-warning' : 'text-success'}
      />
    </div>
  );
}

import { PRESETS } from '../types/analysis';

interface EmptyStateProps {
  onSelect: (date: string) => void;
}

export default function EmptyState({ onSelect }: EmptyStateProps) {
  return (
    <div className="py-16 text-center">
      <p className="text-base mb-8" style={{ color: '#333333' }}>SELECT A PRESET OR ENTER A DATE TO BEGIN ANALYSIS</p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
        {PRESETS.map((p) => (
          <button
            key={p.date}
            onClick={() => onSelect(p.date)}
            className="bg-card border border-border rounded-lg p-5 text-left transition-colors hover:border-primary group"
          >
            <p className="text-xs uppercase tracking-widest text-foreground mb-2 group-hover:text-primary transition-colors">{p.label}</p>
            <p className="text-xs text-muted-foreground">{p.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

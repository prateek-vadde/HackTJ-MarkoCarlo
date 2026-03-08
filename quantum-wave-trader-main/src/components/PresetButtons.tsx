import { PRESETS } from '../types/analysis';

interface PresetButtonsProps {
  onSelect: (date: string) => void;
}

export default function PresetButtons({ onSelect }: PresetButtonsProps) {
  return (
    <div className="flex flex-wrap gap-3 mb-4">
      {PRESETS.map((p) => (
        <button
          key={p.date}
          onClick={() => onSelect(p.date)}
          className="border border-border bg-card text-foreground text-xs uppercase tracking-widest px-4 py-2.5 rounded-lg transition-colors hover:border-primary hover:text-primary"
        >
          {p.label}
        </button>
      ))}
    </div>
  );
}

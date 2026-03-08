interface AnalyzeBarProps {
  date: string;
  onDateChange: (d: string) => void;
  onAnalyze: () => void;
  disabled: boolean;
}

export default function AnalyzeBar({ date, onDateChange, onAnalyze, disabled }: AnalyzeBarProps) {
  return (
    <div className="flex gap-3 mb-6">
      <input
        type="date"
        value={date}
        onChange={(e) => onDateChange(e.target.value)}
        className="bg-card border border-border rounded-lg px-4 py-2 text-foreground text-sm font-mono focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-colors"
      />
      <button
        onClick={onAnalyze}
        disabled={disabled}
        className="bg-primary text-primary-foreground font-bold text-sm px-6 py-2 rounded-lg uppercase tracking-wider hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        Analyze
      </button>
    </div>
  );
}

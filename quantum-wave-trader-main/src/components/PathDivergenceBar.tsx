interface PathDivergenceBarProps {
  value: number;
}

export default function PathDivergenceBar({ value }: PathDivergenceBarProps) {
  const fillWidth = Math.min(value * 1000, 100);

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-2">
        <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Path Divergence</p>
        <p className="font-mono text-sm text-foreground">{value.toFixed(4)}</p>
      </div>
      <div className="w-full h-2 rounded-full bg-border overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${fillWidth}%`,
            background: 'linear-gradient(90deg, #00d4ff 0%, #ff4444 100%)',
          }}
        />
      </div>
      <div className="flex justify-between mt-1">
        <span className="text-[10px] text-success">STABLE</span>
        <span className="text-[10px] text-destructive">CRITICAL</span>
      </div>
    </div>
  );
}

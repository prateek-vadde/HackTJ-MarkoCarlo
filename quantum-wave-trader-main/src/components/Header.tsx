import type { AppStatus } from '../types/analysis';

interface HeaderProps {
  status: AppStatus;
}

export default function Header({ status }: HeaderProps) {
  const dotColor = status === 'analyzing' ? 'bg-destructive' : status === 'loaded' ? 'bg-success' : 'bg-primary';

  return (
    <div className="flex items-center justify-between py-6">
      <div>
        <h1 className="text-[28px] font-bold tracking-widest text-foreground">MARKOCARLO</h1>
        <p className="text-xs text-muted-foreground">Quantum Geometric Market Analysis</p>
      </div>
      <div className={`h-3 w-3 rounded-full ${dotColor} animate-pulse-dot`} />
    </div>
  );
}

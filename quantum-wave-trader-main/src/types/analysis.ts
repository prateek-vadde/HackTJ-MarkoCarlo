export interface AnalysisResponse {
  dates: string[];
  prices: number[];
  psi_real: number[];
  psi_imag: number[];
  curvature: number[];
  risk_score: number;
  crisis_prob: number;
  berry_magnitude: number;
  reversal_risk: number;
  forward_dates: string[];
  mc_paths: number[][];
  predicted_mean: number[];
  predicted_upper: number[];
  predicted_lower: number[];
  expected_return: number;
  return_std: number;
  expected_drawdown: number;
  worst_case_drawdown: number;
  crisis_path_fraction: number;
  path_divergence: number;
  n_accepted_paths: number;
  acceptance_rate: number;
  actual_forward_prices: number[];
  actual_forward_dates: string[];
  vix_current: number | null;
  vix3m_current: number | null;
  vix_ratio: number;
  vix_term_structure: string;
  fixed_node_active: boolean;
}

export type AppStatus = 'idle' | 'analyzing' | 'loaded' | 'error';

export interface Preset {
  label: string;
  date: string;
  description: string;
}

export const PRESETS: Preset[] = [
  { label: 'EURO CRISIS — AUG 2011', date: '2011-08-04', description: 'August 4, 2011 · Mean=-5.28% vs actual=-5.15%, 0.1% error' },
  { label: 'LEHMAN COLLAPSE — SEP 2008', date: '2008-09-25', description: 'September 25, 2008 · Mean=-16.0% vs actual=-18.0%, shape aligned' },
  { label: 'COVID CRASH — FEB 2020', date: '2020-02-26', description: 'February 26, 2020 · Mean=-7.69% vs actual=-7.75%, 0.1% error, shape ρ=0.54' },
  { label: 'COVID DEEP CRASH — MAR 2020', date: '2020-03-11', description: 'March 11, 2020 · Mean=-19.40% vs actual=-15.20%, WHO declares pandemic' },
  { label: 'CALM MARKET — JUN 2023', date: '2023-06-15', description: 'June 15, 2023 · Stable market, symmetric fan, no fixed-node' },
];

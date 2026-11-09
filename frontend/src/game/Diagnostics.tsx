import type { Diagnostic } from '../types';
import { fmt, Empty } from '../components/ui';
export function Diagnostics({ diag, names, actions, label }: { diag?: Diagnostic; names?: string[]; actions?: Record<string, string>; label: string }) {
  if (!diag) return <Empty>No decision on this tick for {label}.</Empty>;
  return <div className="diagnostics"><div className="diagnostic-action"><span>Chosen action</span><strong>{actions?.[String(diag.action)] ?? `Action ${diag.action}`}</strong></div>
    {diag.q_values && <section aria-label="Q-values"><h3>Q-values</h3>{diag.q_values.map((v, i) => <div className="diagnostic-row" key={i}><span>{actions?.[String(i)] ?? `Action ${i}`}</span><b>{fmt(v, 5)}</b></div>)}</section>}
    {diag.action_probs && <section aria-label="Action probabilities"><h3>Action probabilities</h3>{diag.action_probs.map((v, i) => <div className="probability-row" key={i}><div><span>{actions?.[String(i)] ?? `Action ${i}`}</span><b>{fmt(v * 100, 2)}%</b></div><progress aria-label={actions?.[String(i)] ?? `Action ${i}`} value={v} max={1}/></div>)}</section>}
    <details open><summary>Observation vector</summary>{names ? <dl className="observation-grid">{diag.obs.map((v, i) => <div key={i}><dt>{names[i] ?? `Observation ${i}`}</dt><dd>{fmt(v, 4)}</dd></div>)}</dl> : <p className="muted">Observation labels are unavailable. Retry the game configuration request.</p>}</details>
  </div>;
}

import { useState, type ReactNode } from 'react';
import { BarChart3, GitCompareArrows } from 'lucide-react';
import { query, useResource } from '../api';
import type { Benchmark, Benchmarks, Comparison, Leaderboard, Pair } from '../types';
import { Badge, Empty, ErrorMessage, Field, fmt, PageHeader, Panel, plural, ResourceView, Table } from '../components/ui';
import { Chart, chartColors } from '../components/Chart';
import { adaptCompare, adaptLeaderboard, confidenceInterval } from './adapters';
export const ciText = (value: unknown) => { const ci = confidenceInterval(value); return ci ? `[${fmt(ci[0],3)}, ${fmt(ci[1],3)}]` : 'Not available'; };
function Benchmarked({ children }: { children: (b: Benchmark, metrics: Record<string,string>) => ReactNode }) {
  const benchmarks = useResource<Benchmarks>('/api/benchmarks');
  const [choice, setChoice] = useState('');
  return <ResourceView resource={benchmarks}>{data => {
    if (!data.benchmarks.length) return <Panel><Empty>No benchmarks yet. Complete training runs to create a comparison.</Empty></Panel>;
    const key = (b: Benchmark) => `${b.env_version}:${b.total_steps}`;
    const selected = data.benchmarks.find(b => key(b) === choice) ?? data.benchmarks[0];
    return <><div className="benchmark-selector"><BarChart3 size={20}/><Field label="Benchmark"><select value={key(selected)} onChange={e => setChoice(e.target.value)}>{data.benchmarks.map(b => <option key={key(b)} value={key(b)}>{fmt(b.total_steps,0)} {plural(b.total_steps, 'step')} · {b.env_version} · {b.runs} {plural(b.runs, 'run')}</option>)}</select></Field><span>One environment.<br/>One training budget.</span></div><div key={key(selected)}>{children(selected, data.metrics)}</div></>;
  }}</ResourceView>;
}
function ComparisonView({ benchmark }: { benchmark: Benchmark }) {
  const resource = useResource<Comparison>(`/api/compare?${query({ total_steps: benchmark.total_steps, env_version: benchmark.env_version })}`);
  const [a,setA] = useState(''), [b,setB] = useState(''), [pairPath,setPairPath] = useState<string | null>(null);
  const pair = useResource<Pair>(pairPath);
  return <ResourceView resource={resource}>{data => {
    let groups: ReturnType<typeof adaptCompare>['groups'];
    try { groups = adaptCompare(data).groups; } catch (e) { return <ErrorMessage message={e instanceof Error ? e.message : 'Invalid benchmark data.'}/>; }
    if (!groups.length) return <Empty>No groups in this benchmark yet.</Empty>;
    const aValue = a || groups[0].group_key, bValue = b || groups[1]?.group_key || groups[0].group_key;
    return <>
      <Panel title="How quickly do they learn?" action={<Badge tone="mint">95% bootstrap CI</Badge>}><p className="muted">{data.benchmark.validation ?? 'Validation score across training seeds. Shading shows the confidence interval.'}</p><Chart yLabel="Validation score" series={groups.map((g,i) => ({ name: g.label, color: chartColors[i % chartColors.length], points: g.points }))}/></Panel>
      <Panel title="Final-policy comparison"><Table headers={['Policy group', 'Seeds', 'Mean score', '95% CI', 'AUC', 'Wall time', 'Steps / s']}>{groups.map((g,i) => <tr key={g.group_key}><td><span className="group-swatch" style={{ background: chartColors[i % chartColors.length] }}/><strong>{g.label}</strong><small>{g.kind} · {g.group_key}</small></td><td>{g.n_seeds}<small>{g.seeds.join(', ')}</small></td><td className="number-emphasis">{fmt(g.final.mean)}</td><td>{ciText(g.final.ci)}</td><td>{fmt(g.auc.mean,4)}</td><td>{fmt(g.wall_seconds_mean)} s</td><td>{fmt(g.sps_mean,0)}</td></tr>)}</Table></Panel>
      <Panel title="Compare two groups" action={<GitCompareArrows size={17}/>}><form className="inline-form" onSubmit={e => { e.preventDefault(); setPairPath(`/api/compare/pair?${query({ a: aValue, b: bValue })}`); if (pairPath === `/api/compare/pair?${query({ a: aValue, b: bValue })}`) pair.refresh(); }}><Field label="Group A"><select value={aValue} onChange={e => { setA(e.target.value); setPairPath(null); }}>{groups.map(g => <option value={g.group_key} key={g.group_key}>{g.label}</option>)}</select></Field><Field label="Group B"><select value={bValue} onChange={e => { setB(e.target.value); setPairPath(null); }}>{groups.map(g => <option value={g.group_key} key={g.group_key}>{g.label}</option>)}</select></Field><button className="button" disabled={aValue === bValue || pair.loading}>Compare</button></form>{pairPath && <ResourceView resource={pair}>{p => <><p className="muted">Difference is A − B: {p.a} compared with {p.b}.</p><Table headers={['Metric','A mean','B mean','Difference','95% CI','P(improvement)','Seeds A / B']}>{(['final_mean','auc'] as const).map(key => <tr key={key}><td>{key === 'auc' ? 'AUC' : 'Final mean'}</td><td>{fmt(p[key].mean_a,3)}</td><td>{fmt(p[key].mean_b,3)}</td><td>{fmt(p[key].diff,3)}</td><td>{ciText(p[key].ci)}</td><td>{fmt(p[key].prob_improvement,3)}</td><td>{fmt(p[key].n_a,0)} / {fmt(p[key].n_b,0)}</td></tr>)}</Table><p className="muted">Wall-clock ratio A / B: {fmt(p.wall_ratio,3)}</p></>}</ResourceView>}</Panel>
      <div className="two-columns"><Panel title="In-distribution test" action={<Badge>Confirmatory only</Badge>}>{groups.some(g => g.inDistribution) ? <Table headers={['Group','Default score','95% CI','Evaluated values']}>{groups.map(g => <tr key={g.group_key}><td>{g.label}</td><td>{fmt(g.inDistribution?.mean)}</td><td>{ciText(g.inDistribution?.ci)}</td><td>{g.inDistribution?.values?.length ?? '—'}</td></tr>)}</Table> : <Empty>No confirmatory in-distribution evaluations yet.</Empty>}</Panel><Panel title="Out-of-distribution test" action={<Badge tone="amber">Changed conditions</Badge>}>{groups.some(g => g.outOfDistribution.length) ? <Table headers={['Group','Scenario','Score','95% CI']}>{groups.flatMap(g => g.outOfDistribution.length ? g.outOfDistribution.map(s => <tr key={`${g.group_key}-${s.name}`}><td>{g.label}</td><td>{s.name}</td><td>{fmt(s.mean)}</td><td>{ciText(s.ci)}</td></tr>) : [<tr key={g.group_key}><td>{g.label}</td><td colSpan={3}>Not evaluated</td></tr>])}</Table> : <Empty>No confirmatory OOD evaluations yet.</Empty>}</Panel></div>
    </>;
  }}</ResourceView>;
}
export function ComparisonPage() { return <><PageHeader title="Put the evidence side by side." description="Baselines and experimental policies, compared within the same benchmark."/><Benchmarked>{b => <ComparisonView benchmark={b}/>}</Benchmarked></>; }
function LeaderboardView({ benchmark, metrics }: { benchmark: Benchmark; metrics: Record<string,string> }) {
  const [metric,setMetric] = useState('final_mean');
  const resource = useResource<Leaderboard>(`/api/leaderboard?${query({ metric, total_steps: benchmark.total_steps, env_version: benchmark.env_version })}`);
  return <><div className="filter-row"><Field label="Rank by"><select value={metric} onChange={e => setMetric(e.target.value)}>{Object.entries(metrics).map(([id,label]) => <option value={id} key={id}>{id.replaceAll('_',' ')} · {label}</option>)}</select></Field></div><ResourceView resource={resource}>{response => {
    const data = adaptLeaderboard(response);
    return <Panel title="The leaderboard" action={<Badge>{data.lower_is_better ? 'Lower is better' : 'Higher is better'}</Badge>}><p className="muted">{data.description}</p>{data.rows.length ? <Table headers={['Rank','Policy','Training seeds','Value','95% CI','Final mean','Wall time']}>
      {data.rows.map(r => <tr key={r.group_key} className={r.rank === 1 ? 'leader-row' : ''}><td className="rank">{r.rank === null ? <span className="muted">Unranked</span> : String(r.rank).padStart(2,'0')}</td><td><strong>{r.label}</strong><small>{r.kind} · {r.group_key}</small></td><td>{r.n_seeds}<small>Runs {r.run_ids.join(', ')}</small></td><td className="number-emphasis">{r.value === null ? 'No data' : fmt(r.value,4)}</td><td>{ciText(r.ci)}</td><td>{fmt(r.final_mean)}</td><td>{fmt(r.wall_seconds_mean)} s</td></tr>)}</Table> : <Empty>No groups for this benchmark.</Empty>}<p className="table-note">Training seed counts describe the group. Held-out evaluations may cover fewer models. Missing measurements remain unranked.</p></Panel>;
  }}</ResourceView></>;
}
export function LeaderboardPage() { return <><PageHeader title="Results earn their place." description="Rank policies by a measured metric. Missing evidence stays visible."/><Benchmarked>{(b,metrics) => <LeaderboardView benchmark={b} metrics={metrics}/>}</Benchmarked></>; }

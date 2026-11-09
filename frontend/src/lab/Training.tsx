import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Play, Square, RotateCcw } from 'lucide-react';
import { errorText, post, request, useAction, useResource, type Resource } from '../api';
import type { Algorithms, Evaluation, Metrics, Profiles, Run } from '../types';
import { ActionFeedback, Badge, DataDetails, Empty, Field, fmt, humanize, PageHeader, Panel, plural, ResourceView, Stat, Table, time } from '../components/ui';
import { Chart, chartColors } from '../components/Chart';
import { EventFeed } from './Overview';

export function mergeMetrics(old: Metrics, incoming: Metrics): Metrics {
  const merged = { ...old };
  Object.entries(incoming).forEach(([key, points]) => {
    const byStep = new Map(old[key] ?? []);
    points.forEach(([step, value]) => byStep.set(step, value));
    merged[key] = [...byStep.entries()].sort((a, b) => a[0]-b[0]);
  });
  return merged;
}
function useMetrics(id: number | null): Resource<Metrics> {
  const [value, setValue] = useState<{ id: number | null; data: Metrics | null; loading: boolean; error: string | null; updated: number | null }>({ id, data: null, loading: !!id, error: null, updated: null });
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    if (id === null) return;
    let cursor = 0, all: Metrics = {};
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout>;
    setValue({ id, data: null, loading: true, error: null, updated: null });
    const load = async () => {
      try {
        const incoming = await request<Metrics>(`/api/training/runs/${id}/metrics?since_step=${Math.max(-1,cursor - 1)}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        all = mergeMetrics(all, incoming);
        const steps = Object.values(incoming).flatMap(points => points.map(p => p[0]));
        if (steps.length) cursor = Math.max(cursor, ...steps);
        setValue({ id, data: all, loading: false, error: null, updated: Date.now() });
      } catch (e) { if (!controller.signal.aborted) setValue(old => ({ ...old, loading: false, error: errorText(e) })); }
      finally { if (!controller.signal.aborted) timer = setTimeout(load, 1000); }
    };
    void load(); return () => { controller.abort(); clearTimeout(timer); };
  }, [id, revision]);
  return { ...(value.id === id ? value : { data: null, loading: !!id, error: null, updated: null }), refresh: () => setRevision(x => x+1) };
}
function RunDetail({ id }: { id: number }) {
  const detail = useResource<Run>(`/api/training/runs/${id}`, 1500);
  const metrics = useMetrics(id);
  const evals = useResource<Evaluation[]>(`/api/training/runs/${id}/evals`, 3000);
  const logs = useResource<{ run_id: number; lines: string[] }>(`/api/training/runs/${id}/log?tail=40`, 3000);
  const action = useAction();
  return <><ResourceView resource={detail}>{run => <>
    <Panel title={`Run #${run.id} · ${run.algo.toUpperCase()}${run.variant ? ` / ${run.variant}` : ''}`} action={<Badge>{run.status}</Badge>}><div className="run-progress"><progress aria-label="Training step progress" value={run.steps} max={run.total_steps}/><span>{fmt(run.steps,0)} / {fmt(run.total_steps,0)} {plural(run.total_steps, 'step')}</span></div><div className="button-row"><button className="button danger" disabled={!['running','starting'].includes(run.status) || !!run.stop_requested || action.pending} onClick={() => void action.execute(() => post(`/api/training/runs/${id}/stop`), detail.refresh, 'Stop requested. The trainer will checkpoint before exiting.')}><Square size={14}/>{['running','starting'].includes(run.status) && run.stop_requested ? 'Stopping safely…' : 'Stop safely'}</button><button className="button" disabled={!['stopped','interrupted','limit'].includes(run.status) || action.pending} onClick={() => void action.execute(() => post(`/api/training/runs/${id}/resume`), detail.refresh, 'Resume requested.')}><RotateCcw size={14}/>Resume run</button><span className="muted">Seed {run.seed} · {run.profile} · {run.device ?? 'Device pending'}</span></div><ActionFeedback action={action}/>{run.error && <div className="notice error"><pre>{run.error}</pre></div>}<DataDetails title="Active configuration, hardware and code version" value={{ config: run.config, hardware: run.hardware, code_version: run.code_version, env_version: run.env_version }}/></Panel>
    {run.summary && <div className="stat-strip"><Stat label="Final validation mean" value={fmt(run.summary.final_mean)}/><Stat label="Best validation mean" value={fmt(run.summary.best_mean)}/><Stat label="Sample efficiency (AUC)" value={fmt(run.summary.auc,4)}/><Stat label="Wall time" value={`${fmt(run.wall_seconds)} s`}/></div>}
    <ResourceView resource={metrics}>{data => <div className="two-columns">{[
      { title: 'Episode score', keys: ['ep_score_mean'], unit: 'Obstacles cleared' },
      { title: 'Episode return', keys: ['ep_return_mean'], unit: 'Return' },
      { title: 'Validation', keys: ['val_mean', 'val_median'], unit: 'Score' },
      { title: 'Loss', keys: ['loss', 'value_loss', 'policy_gradient_loss'], unit: 'Loss' },
      { title: 'Throughput', keys: ['sps'], unit: 'Steps / second' },
    ].map(c => { const keys = c.keys.filter(k => run.metric_keys?.includes(k)); return <Panel key={c.title} title={c.title}>{keys.length ? <Chart compact yLabel={c.unit} series={keys.map((key,i) => ({ name: humanize(key), color: chartColors[i], points: (data[key] ?? []).map(([x,y]) => ({ x,y })) }))}/> : <Empty>This run has not reported {c.title.toLowerCase()} metrics.</Empty>}</Panel>; })}<Panel title="Available measurements"><p className="muted">Only metrics logged by this algorithm are shown.</p><DataDetails title="All measured series" value={data}/></Panel></div>}</ResourceView>
  </>}</ResourceView><Panel title="Validation evaluations"><ResourceView resource={evals} empty="No evaluations yet. They appear at the configured evaluation interval.">{rows => <Table headers={['Step', 'Purpose', 'Scenario', 'Episodes', 'Mean', 'Median', 'Maximum']}>{rows.map(e => <tr key={e.id}><td>{fmt(e.step,0)}</td><td>{e.purpose}</td><td>{e.scenario}</td><td>{e.n}</td><td>{fmt(e.mean)}</td><td>{fmt(e.median)}</td><td>{fmt(e.max)}</td></tr>)}</Table>}</ResourceView></Panel><Panel title="Trainer output"><ResourceView resource={logs}>{log => log.lines.length ? <pre className="log-output">{log.lines.join('\n')}</pre> : <Empty>No log output yet.</Empty>}</ResourceView></Panel><Panel title="Errors & activity"><EventFeed runId={id}/></Panel></>;
}
export function Training() {
  const [search, setSearch] = useSearchParams();
  const algorithms = useResource<Algorithms>('/api/algorithms');
  const profiles = useResource<Profiles>('/api/profiles');
  const runs = useResource<Run[]>('/api/training/runs?limit=100', 2000);
  const [algo, setAlgo] = useState('dqn'), [variant, setVariant] = useState('control'), [seed,setSeed] = useState(0), [steps,setSteps] = useState('5000'), [profile,setProfile] = useState('quick_smoke');
  const [device,setDevice] = useState('auto');
  const action = useAction();
  const requestedId = Number(search.get('run'));
  const selectedId = requestedId > 0 ? requestedId : runs.data?.[0]?.id ?? null;
  return <><PageHeader title="Learning, in real time." description="Start a bounded run and follow the policy from its first decision."/>
    <Panel title="New training run"><ResourceView resource={algorithms}>{a => <ResourceView resource={profiles}>{p => <form onSubmit={e => { e.preventDefault(); void action.execute(() => post<Run>('/api/training/runs', { algo, ...(algo === 'xdqn' ? { variant } : {}), seed, profile, ...(steps !== '' ? { total_steps: Number(steps) } : {}), device }), r => { setSearch({ run: String(r.id) }); runs.refresh(); }, 'Training run created.'); }}><div className="form-grid training-form"><Field label="Algorithm"><select value={algo} onChange={e => setAlgo(e.target.value)}>{a.algorithms.map(row => <option key={row.id} value={row.id}>{row.name}</option>)}</select></Field>{a.algorithms.find(row => row.id === algo)?.supports_variants && <Field label="Variant"><select value={variant} onChange={e => setVariant(e.target.value)}>{a.variants.map(v => <option key={v.id} value={v.id}>{v.id}</option>)}</select></Field>}<Field label="Profile"><select value={profile} onChange={e => { setProfile(e.target.value); setSteps(''); }}>{Object.entries(p.profiles).map(([id]) => <option key={id} value={id}>{humanize(id)}</option>)}</select></Field><Field label="Step override" help={`Blank uses profile: ${fmt(p.profiles[profile]?.steps,0)} ${plural(p.profiles[profile]?.steps, 'step')}.`}><input type="number" min={p.overridable.steps.min} max={p.overridable.steps.max} step={1} value={steps} onChange={e => setSteps(e.target.value)} placeholder="Use profile"/></Field><Field label="Seed"><input type="number" required min={0} value={seed} onChange={e => setSeed(Number(e.target.value))}/></Field><Field label="Device"><select value={device} onChange={e => setDevice(e.target.value)}><option value="auto">Automatic</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></Field></div><p className="muted">{p.profiles[profile]?.description}</p><button className="button primary" disabled={action.pending}><Play size={16}/>{action.pending ? 'Starting…' : 'Start training'}</button></form>}</ResourceView>}</ResourceView><ActionFeedback action={action}/></Panel>
    <Panel title="Run directory"><ResourceView resource={runs} empty="No runs yet. Start your first bounded training run above.">{rows => <><Field label="Inspect run"><select value={selectedId ?? ''} onChange={e => setSearch({ run: e.target.value })}>{requestedId && !rows.some(r=>r.id === requestedId) ? <option value={requestedId}>Run #{requestedId}</option> : null}{rows.map(r => <option key={r.id} value={r.id}>#{r.id} · {r.algo}{r.variant ? ` / ${r.variant}` : ''} · seed {r.seed} · {r.status} · {time(r.created_at)}</option>)}</select></Field><p className="muted">The latest 100 runs. Select a run to inspect its configuration and measurements.</p></>}</ResourceView></Panel>
    {selectedId !== null && <RunDetail key={selectedId} id={selectedId}/>}
  </>;
}

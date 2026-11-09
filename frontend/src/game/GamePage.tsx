import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Play, Pause, RotateCcw, ArrowUpRight, Radio, Trophy, SlidersHorizontal } from 'lucide-react';
import { motion } from 'motion/react';
import { post, useAction, useResource } from '../api';
import type { Difficulty, GameConfig, MatchRow, MatchStats, Mode, Model, ReadyMessage, Series } from '../types';
import { ActionFeedback, Badge, DataDetails, Empty, ErrorMessage, Field, fmt, Loading, PageHeader, Panel, plural, ResourceView, Table, time } from '../components/ui';
import { useSettings } from '../settings';
import { useGame } from './useGame';
import { bestKey } from './state';
import { World } from './World';
import { Diagnostics } from './Diagnostics';
import { BirdArt } from './art';
import { pilotVerb } from './copy';
import { readStageSize, useStageSize, viewWidthChanged, viewWidthForStage } from './viewport';

export const modes: Record<Mode, { title: string; description: string; caption: string }> = {
  human: { title: 'Find your flow.', description: 'A little lift. A clear gap. One more flight.', caption: 'Human' },
  ai: { title: 'Watch intelligence take flight.', description: 'Choose a trained policy and see every decision as it happens.', caption: 'AI flight' },
  human_vs_ai: { title: 'Instinct meets intelligence.', description: 'You and an agent. The same canyon. See who goes further.', caption: 'Human vs AI' },
  ai_vs_ai: { title: 'May the best policy fly.', description: 'Two agents, one shared world. Compare a flight or run a seeded series.', caption: 'AI vs AI' },
};
export function ModelSelect({ label, models, value, onChange }: { label: string; models: Model[]; value: string; onChange: (id: string) => void }) {
  return <Field label={label}><select required value={value} onChange={e => onChange(e.target.value)}><option value="" disabled>Choose a model</option>{models.map(m => <option value={m.id} key={m.id}>{m.id} · {m.algo.toUpperCase()}{m.variant ? ` / ${m.variant}` : ''} · {fmt(m.train_steps, 0)} {plural(m.train_steps, 'step')} · {m.status}</option>)}</select></Field>;
}
export function MatchList({ mode, revision = 0 }: { mode?: Mode; revision?: number }) {
  const matches = useResource<MatchRow[]>(`/api/matches?limit=20${mode ? `&mode=${mode}` : ''}`);
  useEffect(() => { if (revision) matches.refresh(); }, [revision, matches.refresh]);
  return <ResourceView resource={matches} empty="No saved flights yet. Finish a flight to see it here.">{rows => <Table headers={['Flight', 'Pilots', 'Score', 'Outcome', 'Recorded', '']}>
    {rows.map(m => <tr key={m.id}><td className="mono">#{m.id}<small>{m.difficulty} / seed {m.seed}</small></td><td>{m.result.players.map(p => p.label).join(' / ')}</td><td className="mono">{m.result.players.map(p => fmt(p.score, 0)).join(' : ')}</td><td>{m.result.draw ? 'Draw' : m.result.winner !== null ? pilotVerb(m.result.players[m.result.winner].label, 'win') : 'Solo flight'}</td><td>{time(m.created_at)}</td><td><Link className="button small ghost" to={`/replays?match=${m.id}`}><Play size={13}/>Replay</Link></td></tr>)}
  </Table>}</ResourceView>;
}
function SeriesPanel({ a, b, difficulty }: { a: string; b: string; difficulty: Difficulty }) {
  const action = useAction();
  const [n, setN] = useState(5); const [seed, setSeed] = useState(30000); const [limit, setLimit] = useState(4600);
  const [result, setResult] = useState<Series | null>(null);
  return <Panel title="Seeded competition" action={<Badge>AI vs AI</Badge>}><p className="muted">Run both selected models on identical fields. Results are saved as replayable matches.</p><form onSubmit={e => { e.preventDefault(); setResult(null); void action.execute(() => post<Series>('/api/competitions/series', { model_a: a, model_b: b, difficulty, n_seeds: n, first_seed: seed, limit_ticks: limit }), setResult, 'Series completed.'); }}><div className="form-grid"><Field label="Seeds"><input type="number" required min={1} max={25} value={n} onChange={e => setN(Number(e.target.value))}/></Field><Field label="First seed"><input type="number" required min={0} value={seed} onChange={e => setSeed(Number(e.target.value))}/></Field><Field label="Tick limit per flight"><input type="number" required min={1} max={108000} value={limit} onChange={e => setLimit(Number(e.target.value))}/></Field></div><button className="button" disabled={!a || !b || action.pending}><Trophy size={16}/>{action.pending ? 'Running series…' : 'Run series'}</button></form><ActionFeedback action={action}/>{result && <><p className="series-summary">{result.model_a}: <b>{result.wins_a} {plural(result.wins_a, 'win')}</b> · {result.model_b}: <b>{result.wins_b} {plural(result.wins_b, 'win')}</b> · {result.draws} {plural(result.draws, 'draw')}</p><p className="muted">Mean scores: {fmt(result.mean_score_a)} / {fmt(result.mean_score_b)} across {result.n} {plural(result.n, 'seed')}.</p><Table headers={['Seed', result.model_a, result.model_b, 'Winner', '']}>
    {result.matches.map(m => <tr key={m.match_id}><td>{m.seed}</td><td>{fmt(m.scores[0])}</td><td>{fmt(m.scores[1])}</td><td>{m.draw ? 'Draw' : m.winner === 0 ? result.model_a : result.model_b}</td><td><Link to={`/replays?match=${m.match_id}`}>Replay #{m.match_id}</Link></td></tr>)}</Table></>}</Panel>;
}
export function GamePage({ mode = 'human', replay = false }: { mode?: Mode; replay?: boolean }) {
  const [search] = useSearchParams();
  const modelResource = useResource<Model[]>(replay || mode === 'human' ? null : '/api/models');
  const config = useResource<GameConfig>('/api/game/config');
  const [includeArchived, setIncludeArchived] = useState(() => !!search.get('model'));
  const available = (modelResource.data ?? []).filter(m => includeArchived || m.status !== 'archived');
  const [modelA, setModelA] = useState(search.get('model') ?? '');
  const [modelB, setModelB] = useState('');
  const [difficulty, setDifficulty] = useState<Difficulty>('normal');
  const [seed, setSeed] = useState('');
  const [sameSeed, setSameSeed] = useState(false);
  const [pulse, setPulse] = useState(-1000);
  const [selectedReplay, setSelectedReplay] = useState(search.get('match') ?? '');
  const [diagPlayer, setDiagPlayer] = useState(0);
  const replays = useResource<MatchRow[]>(replay ? '/api/matches?limit=100' : null);
  const replayDetail = useResource<MatchRow>(replay && selectedReplay ? `/api/matches/${encodeURIComponent(selectedReplay)}` : null);
  const { state, send, reconnect } = useGame();
  const { ready, snapshot, result } = state;
  const stage = useStageSize();
  const replacement = useRef<{ ready: ReadyMessage; pause: boolean } | null>(null);
  const humanInput = useRef<ReadyMessage | null>(null);
  const { settings, reduce } = useSettings();
  const human = !replay && (mode === 'human' || mode === 'human_vs_ai');
  const needsAI = mode !== 'human' && !replay;
  const validA = available.some(m => m.id === modelA) ? modelA : available.find(m => m.status === 'promoted')?.id ?? available[0]?.id ?? '';
  const validB = available.some(m => m.id === modelB) ? modelB : available.find(m => m.id !== validA)?.id ?? validA;
  const connected = state.connection === 'connected';
  const canSetSpeed = !human || (!!ready && !!snapshot && !result
    && ['running', 'paused'].includes(snapshot.phase)
    && ready.players.some(p => p.kind === 'human')
    && ready.players.every((p, i) => p.kind !== 'human' || snapshot.players[i]?.alive === false));
  const canFlap = human && connected && !!snapshot && ['ready', 'running'].includes(snapshot.phase) && !!snapshot.players[0]?.alive && !state.pending;
  const flap = useCallback(() => { if (canFlap && send({ type: 'flap' })) { humanInput.current = ready; setPulse(performance.now()); } }, [canFlap, send, ready]);
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (e.repeat || e.ctrlKey || e.altKey || e.metaKey || (e.target instanceof HTMLElement && (e.target.closest('input,select,textarea,button:not(.world-input),a,[contenteditable="true"]')))) return;
      if (e.code === 'Space' || e.key.toLowerCase() === settings.flapKey) { e.preventDefault(); flap(); }
    };
    window.addEventListener('keydown', listener); return () => window.removeEventListener('keydown', listener);
  }, [flap, settings.flapKey]);
  useEffect(() => { if (replay) setSelectedReplay(search.get('match') ?? ''); }, [search, replay]);
  const stats = useResource<MatchStats>(mode === 'human_vs_ai' && !replay ? '/api/matches/stats' : null);
  useEffect(() => { if (result) { stats.refresh(); replays.refresh(); } }, [result, stats.refresh, replays.refresh]);
  const start = () => {
    if (!config.data) return;
    setDiagPlayer(mode === 'human_vs_ai' ? 1 : 0);
    replacement.current = null;
    const replayDifficulty = replayDetail.data?.difficulty ?? replays.data?.find(m => String(m.id) === selectedReplay)?.difficulty ?? 'normal';
    const world = config.data.difficulties[replay ? replayDifficulty : difficulty];
    const size = stage.ref.current ? readStageSize(stage.ref.current) : stage.size;
    const view_width = viewWidthForStage(size, world, config.data.view_width_range);
    const sent = replay
      ? !!selectedReplay && send({ type: 'replay', match_id: Number(selectedReplay), view_width })
      : send({ type: 'start', mode, models: mode === 'ai_vs_ai' ? [validA, validB] : needsAI ? [validA] : [], difficulty, view_width, ...(seed.trim() ? { seed: Number(seed) } : {}) });
    if (sent && ready) replacement.current = { ready, pause: false };
  };
  const restart = useCallback((keepSeed: boolean, resizing = false) => {
    if (!ready || !config.data) return;
    const size = stage.ref.current ? readStageSize(stage.ref.current) : stage.size;
    const view_width = viewWidthForStage(size, ready.config, config.data.view_width_range);
    // Restart keeps the old width on the server. A new start must use the acknowledged
    // match's settings, not unsubmitted edits in the setup form.
    const sent = ready.replay_of !== null
      ? send({ type: 'replay', match_id: ready.replay_of, view_width, speed: snapshot?.speed ?? 1 })
      : send({ type: 'start', mode: ready.mode, models: ready.players.flatMap(p => p.model_id === null ? [] : [p.model_id]),
        difficulty: ready.difficulty, limit_ticks: ready.limit_ticks, view_width,
        ...(keepSeed ? { seed: ready.seed } : {}), ...(!human ? { speed: snapshot?.speed ?? 1 } : {}) });
    if (sent) replacement.current = { ready, pause: resizing && snapshot?.phase === 'paused' };
  }, [ready, config.data, stage.ref, stage.size, send, snapshot?.speed, snapshot?.phase, human]);
  useEffect(() => {
    const pending = replacement.current;
    if (!connected || state.error) { replacement.current = null; return; }
    if (pending && ready !== pending.ready) {
      replacement.current = null;
      if (ready && pending.pause) send({ type: 'pause' });
    }
  }, [ready, connected, state.error, send]);
  const resizedView = !!ready && !!config.data && viewWidthChanged(ready.config.view_width,
    viewWidthForStage(stage.size, ready.config, config.data.view_width_range));
  useEffect(() => {
    if (!resizedView || !ready || !snapshot || !connected || state.pending || state.error || result || replacement.current) return;
    if (snapshot.phase === 'over') return;
    // A human's recorded inputs cannot be resumed by start. Keep the whole match,
    // including the surviving AI after a crash, intact until an explicit rematch.
    if (human && (humanInput.current === ready || snapshot.phase !== 'ready' || snapshot.tick !== 0)) return;
    restart(true, true);
  }, [resizedView, ready, snapshot, connected, state.pending, state.error, result, human, restart]);
  const currentAI = ready?.players.map((p, i) => ({ ...p, i })).filter(p => p.kind === 'ai') ?? [];
  const diagnosticIndex = currentAI.some(p => p.i === diagPlayer) ? diagPlayer : currentAI[0]?.i;
  const bestDistance = snapshot ? Math.max(...snapshot.players.map(p => p.distance), 1) : 1;
  const aiNeedsModel = needsAI && (!validA || (mode === 'ai_vs_ai' && !validB));
  const outcome = result ? result.draw ? 'A perfectly matched flight.' : result.winner === null ? 'Flight complete.' : `${pilotVerb(result.players[result.winner].label, 'win')}.` : '';
  return <><section className="game-workspace"><PageHeader title={replay ? 'Every flight has a story.' : modes[mode].title} description={replay ? 'Replay a recorded flight from its original inputs.' : modes[mode].description}><Badge tone={connected ? 'mint' : 'coral'}>{state.connection}</Badge></PageHeader>
    {state.error && <ErrorMessage message={state.error}/>} {state.connection === 'disconnected' && <div className="notice warning"><span>Flight disconnected. The last snapshot is frozen.</span><button className="button small" onClick={reconnect}>Reconnect</button></div>}
    {ready?.in_distribution === false && <div className="notice warning">Out of distribution: {ready.difficulty} differs from the model’s training difficulty. These results are not a training benchmark.</div>}
    <div className="game-layout"><div className="flight-column">
      <div className="scoreboard">{ready && snapshot ? ready.players.map((p, i) => <div key={i} className={`pilot-score ${i === 1 ? 'rival' : ''} ${!snapshot.players[i]?.alive ? 'crashed' : ''}`}><div className="pilot-identity"><BirdArt rival={i === 1}/><div><span>{p.kind === 'human' ? 'Human pilot' : `AI pilot ${ready.players.length > 1 ? i === 0 ? 'A' : 'B' : ''}`}</span><strong title={p.label}>{p.label}</strong></div><b className="flight-score">{fmt(snapshot.players[i]?.score, 0)}</b></div><div className="pilot-meta"><span>{snapshot.players[i]?.alive ? 'In flight' : `Crashed · ${snapshot.players[i]?.death}`}</span><span>{fmt(snapshot.players[i]?.distance)} px</span></div><progress aria-label={`${p.label} relative distance`} value={snapshot.players[i]?.distance ?? 0} max={bestDistance}/></div>) : <div className="preflight-score"><Radio size={19}/><span>{replay ? 'Choose a recording to begin' : 'Your flight is waiting'}</span></div>}</div>
      {snapshot && snapshot.players.length === 2 && <div className="relative-progress">{snapshot.players[0].distance === snapshot.players[1].distance ? 'Flying level' : `${pilotVerb(ready!.players[snapshot.players[0].distance > snapshot.players[1].distance ? 0 : 1].label, 'lead')} by ${fmt(Math.abs(snapshot.players[0].distance - snapshot.players[1].distance))} px`}</div>}
      <div className={`flight-stage ${ready && stage.size.width > 0 && 480 * ready.config.view_width / ready.config.world_height > stage.size.width ? 'width-constrained' : ''}`} ref={stage.ref}><World state={state} flapPulse={pulse} onFlap={flap} canFlap={canFlap}/>
        {!snapshot && <div className="world-message"><span className="wordmark">aerie</span><p>{state.pending ? 'Preparing your flight…' : replay ? 'Relive the flight.' : 'A little courage. A little lift.'}</p></div>}
        {snapshot?.phase === 'ready' && <div className="world-message ready-message"><strong>Ready when you are.</strong><p>Press Space, {settings.flapKey.toUpperCase()} or tap to take flight.</p></div>}
        {snapshot?.phase === 'paused' && <div className="world-message"><strong>Take a breath.</strong><p>Your flight is paused.</p></div>}
      {result && <motion.section className="result-panel result-overlay" initial={reduce ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} role="status"><Trophy size={27}/><div><span>{result.new_record ? 'New personal record' : result.limit_reached ? 'Flight limit reached' : replay ? 'Replay complete' : 'The results are in'}</span><h2>{outcome}</h2></div><Table headers={['Pilot', 'Score', 'Distance', 'Ticks', 'Flaps', 'Finish']}>{result.players.map((p, i) => <tr key={i}><td>{p.label}</td><td>{p.score}</td><td>{fmt(p.distance)} px</td><td>{p.ticks}</td><td>{p.flaps}</td><td>{p.death ?? 'Limit reached'}</td></tr>)}</Table><div className="button-row"><button className="button primary" disabled={!connected || state.pending} onClick={() => restart(sameSeed)}><RotateCcw size={16}/>{replay ? 'Watch again' : ready?.players.length === 2 ? 'Rematch' : 'Fly again'}</button>{!replay && <Link className="button ghost" to={`/replays?match=${result.match_id}`}>Watch replay<ArrowUpRight size={15}/></Link>}</div></motion.section>}
      </div>
      <div className="flight-toolbar"><div className="button-row"><button className="button ghost" disabled={!connected || state.pending || !snapshot || !['running', 'paused'].includes(snapshot.phase)} onClick={() => send({ type: snapshot?.phase === 'paused' ? 'resume' : 'pause' })}>{snapshot?.phase === 'paused' ? <Play size={16}/> : <Pause size={16}/>} {snapshot?.phase === 'paused' ? 'Resume' : 'Pause'}</button><button className="button ghost" disabled={!connected || state.pending || !ready} onClick={() => restart(sameSeed)}><RotateCcw size={16}/>{replay ? 'Replay again' : 'Restart'}</button></div>{canSetSpeed && <Field label="Playback"><select aria-label="Playback speed" value={snapshot?.speed ?? 1} disabled={!ready || !connected || state.pending} onChange={e => send({ type: 'set_speed', speed: Number(e.target.value) })}>{[.25, .5, 1, 2, 4, 8, 16].map(s => <option key={s} value={s}>{s}×</option>)}</select></Field>}{canFlap && <span className="key-hint"><kbd>Space</kbd> or <kbd>{settings.flapKey.toUpperCase()}</kbd> to flap</span>}</div>
      {ready && snapshot && <div className="flight-footer"><span>Seed {ready.seed} · {ready.difficulty} · {fmt(snapshot.time)} s</span><span>{replay ? `Replay #${ready.replay_of}` : `Tick ${snapshot.tick}`}</span></div>}
    </div><aside className="game-rail"><Panel title={replay ? 'Flight recorder' : 'Flight setup'} action={<SlidersHorizontal size={16}/>}>
      <form onSubmit={e => { e.preventDefault(); start(); }}>
      {replay ? <ResourceView resource={replays} empty="No recordings yet. Complete a flight first.">{rows => <Field label="Stored match"><select value={selectedReplay} required onChange={e => setSelectedReplay(e.target.value)}><option value="" disabled>Choose a match</option>{!rows.some(m => String(m.id) === selectedReplay) && selectedReplay && <option value={selectedReplay}>Match #{selectedReplay}</option>}{rows.map(m => <option key={m.id} value={m.id}>#{m.id} · {m.mode.replaceAll('_', ' ')} · {m.result.players.map(p => p.label).join(' / ')}</option>)}</select></Field>}</ResourceView> : <>
        {needsAI && <><ResourceView resource={modelResource} empty="No registered models. Train a policy in the lab to fly with AI.">{() => available.length ? <><ModelSelect label={mode === 'ai_vs_ai' ? 'Pilot A' : 'AI model'} models={available} value={validA} onChange={setModelA}/>{mode === 'ai_vs_ai' && <ModelSelect label="Pilot B" models={available} value={validB} onChange={setModelB}/>}</> : <Empty>No non-archived models. Include archived models or train a new policy.</Empty>}</ResourceView><label className="check-row compact-check"><input type="checkbox" checked={includeArchived} onChange={e => setIncludeArchived(e.target.checked)}/>Include archived models</label></>}
        <Field label="Difficulty"><select value={difficulty} onChange={e => setDifficulty(e.target.value as Difficulty)}><option value="easy">Easy · wider gaps</option><option value="normal">Normal · training conditions</option><option value="hard">Hard · tighter gaps</option></select></Field><Field label="World seed" help="Leave blank for a fresh field."><input type="number" min={0} step={1} value={seed} onChange={e => setSeed(e.target.value)} placeholder="Random"/></Field>
        <label className="check-row compact-check"><input type="checkbox" checked={sameSeed} onChange={e => setSameSeed(e.target.checked)}/>Keep seed on rematch</label>
      </>}
      <button className="button primary full" disabled={!connected || state.pending || !config.data || aiNeedsModel || (replay && !selectedReplay)}><Play size={17}/>{state.pending ? 'Preparing…' : replay ? 'Play recording' : ready ? 'Start new flight' : 'Launch flight'}</button></form>
      {ready && !replay && <p className="muted small-text">Setup changes apply when you start a new flight.</p>}
      {human && resizedView && snapshot && snapshot.tick > 0 && <p className="muted small-text">Resized view applies on your next flight.</p>}
      {human && ready && <div className="record-row"><span>Personal best · {ready.difficulty}</span><b>{fmt(result?.personal_best ?? ready.personal_best, 0)}</b></div>}
      {!replay && ready && ready.players.map((p, i) => p.kind === 'ai' ? <div className="record-row" key={i}><span>{p.label} · session best</span><b>{fmt(state.sessionBest[bestKey(ready, i)], 0)}</b></div> : null)}
      </Panel>
      {config.error && <ErrorMessage message={config.error} retry={config.refresh}/>}
      {config.loading && <Loading label="Loading observation labels…"/>}
      {currentAI.length > 0 && <Panel title="Inside the policy" action={<Badge>Live decisions</Badge>}>{currentAI.length > 1 && <Field label="Inspect pilot"><select value={diagnosticIndex} onChange={e => setDiagPlayer(Number(e.target.value))}>{currentAI.map(p => <option key={p.i} value={p.i}>{p.label}</option>)}</select></Field>}<Diagnostics label={ready!.players[diagnosticIndex!]?.label ?? 'AI'} diag={snapshot?.players[diagnosticIndex!]?.diag} names={config.data?.observation.names} actions={config.data?.actions}/></Panel>}
      {replay && selectedReplay && <Panel title="Recording details"><ResourceView resource={replayDetail}>{m => <><p className="muted">{time(m.created_at)} · {m.difficulty}</p><DataDetails title="Recorded inputs and result" value={m}/></>}</ResourceView></Panel>}
      {mode === 'human' && !ready && <Panel title="A small field guide"><div className="field-guide"><BirdArt/><p>Give your kite-bird a lift to thread the mineral gates. Stay between the ceiling and floor. Every cleared gate counts.</p></div></Panel>}
    </aside></div></section>
    {mode === 'ai_vs_ai' && !replay && <SeriesPanel a={validA} b={validB} difficulty={difficulty}/>}
    {mode === 'human_vs_ai' && !replay && <><Panel title="Your rivalry"><ResourceView resource={stats}>{s => <div className="rivalry-stats"><span><b>{s.human.vs_ai.wins}</b> {plural(s.human.vs_ai.wins, 'win')}</span><span><b>{s.human.vs_ai.losses}</b> {plural(s.human.vs_ai.losses, 'loss', 'losses')}</span><span><b>{s.human.vs_ai.draws}</b> {plural(s.human.vs_ai.draws, 'draw')}</span><span><b>{s.human.games}</b> human {plural(s.human.games, 'flight')}</span></div>}</ResourceView></Panel><Panel title="Recent head-to-head flights"><MatchList mode="human_vs_ai" revision={result?.match_id}/></Panel></>}
    {replay && <Panel title="Flight archive"><MatchList/></Panel>}
  </>;
}

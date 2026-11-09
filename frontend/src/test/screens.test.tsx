import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import type { GameConfig, MatchResult, Model, ReadyMessage, Run, Series, ServerMessage, Snapshot } from '../types';
import { fixture } from './fixtures';
class TestSocket {
  static CONNECTING=0; static OPEN=1; static instances:TestSocket[]=[];
  readyState=0; sent:unknown[]=[];
  onopen:(()=>void)|null=null; onclose:(()=>void)|null=null; onerror:(()=>void)|null=null; onmessage:((e:{data:string})=>void)|null=null;
  constructor(){TestSocket.instances.push(this);}
  open(){this.readyState=1;this.onopen?.();}
  emit(msg:ServerMessage){this.onmessage?.({data:JSON.stringify(msg)});}
  send(value:string){this.sent.push(JSON.parse(value));}
  close(){this.readyState=3;this.onclose?.();}
}
const routes:Record<string,string>={
  '/api/health':'health','/api/game/config':'game_config','/api/matches/stats':'matches_stats','/api/matches':'matches',
  '/api/models':'models','/api/status':'status','/api/events':'events','/api/algorithms':'algorithms','/api/profiles':'profiles',
  '/api/training/runs':'runs','/api/benchmarks':'benchmarks','/api/compare':'compare','/api/compare/pair':'compare_pair',
  '/api/leaderboard':'leaderboard_final_mean','/api/research/hypotheses':'hypotheses','/api/research/experiments':'experiments',
  '/api/research/sessions':'sessions','/api/research/queue':'queue','/api/research/observations':'observations',
};
let requests:{path:string;options?:RequestInit}[]=[];
let mutationError=false;
function mockApi(input:RequestInfo|URL,options?:RequestInit) {
  const raw=String(input), path=raw.split('?')[0]; requests.push({path:raw,options});
  if(options?.method==='POST') {
    if(mutationError) return Promise.resolve(new Response(JSON.stringify(fixture('error_409')),{status:409}));
    const value = path.endsWith('/series') ? fixture('series') : path.startsWith('/api/models/') ? fixture('model_detail') : path.startsWith('/api/training/') ? fixture('run_detail') : fixture('session_detail');
    return Promise.resolve(new Response(JSON.stringify(value),{status:200}));
  }
  let name=routes[path];
  if(path==='/api/game/config') return Promise.resolve(new Response(JSON.stringify({ ...fixture<GameConfig>('game_config'), view_width_range: [432, 1296] })));
  if(/^\/api\/models\/.+/.test(path))name='model_detail';
  if(/^\/api\/matches\/\d+$/.test(path))name='match_detail';
  if(/^\/api\/training\/runs\/\d+$/.test(path))name='run_detail';
  if(path.endsWith('/metrics'))name='run_metrics';
  if(path.endsWith('/evals'))name='run_evals';
  if(path.endsWith('/log'))return Promise.resolve(new Response(JSON.stringify({run_id:1,lines:[]})));
  if(/^\/api\/research\/sessions\/\d+$/.test(path))name='session_detail';
  if(/^\/api\/research\/experiments\/\d+$/.test(path))name='experiment_detail';
  if(!name)return Promise.resolve(new Response(JSON.stringify({detail:`Unknown test endpoint ${raw}`}),{status:404}));
  return Promise.resolve(new Response(JSON.stringify(fixture(name)),{status:200}));
}
function visit(path:string){window.location.hash=`#${path}`;return render(<App/>);}
function trace(name:string){return fixture<{dir:string;msg:ServerMessage}[]>(name).filter(t=>t.dir==='server').map(t=>t.msg);}
beforeEach(()=>{
  requests=[];mutationError=false;TestSocket.instances=[];localStorage.clear();
  vi.stubGlobal('fetch',vi.fn(mockApi));vi.stubGlobal('WebSocket',TestSocket);
  vi.stubGlobal('matchMedia',vi.fn(()=>({matches:false,addListener:vi.fn(),removeListener:vi.fn(),addEventListener:vi.fn(),removeEventListener:vi.fn()})));
  vi.spyOn(window,'scrollTo').mockImplementation(()=>{});
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockImplementation(()=>null);
});
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});
describe('all screens are navigable and consume captured API responses',()=>{
  it.each([
    ['/','Find yournext horizon.','/api/matches/stats'],
    ['/play/human','Find your flow.','/api/game/config'],
    ['/play/ai','Watch intelligence take flight.','/api/models'],
    ['/play/human_vs_ai','Instinct meets intelligence.','/api/matches?mode=human_vs_ai'],
    ['/play/ai_vs_ai','May the best policy fly.','/api/models'],
    ['/replays','Every flight has a story.','/api/matches'],
    ['/settings','Make yourself at home.',null],
    ['/lab','A better flight starts here.','/api/status'],
    ['/lab/training','Learning, in real time.','/api/training/runs'],
    ['/lab/compare','Put the evidence side by side.','/api/compare'],
    ['/lab/leaderboard','Results earn their place.','/api/leaderboard'],
    ['/lab/models','Meet the pilots.','/api/models'],
    ['/lab/history','Every idea leaves a trace.','/api/research/hypotheses'],
    ['/lab/control','Give curiosity a boundary.','/api/research/queue'],
  ])('renders %s',async(path,heading,endpoint)=>{
    visit(path);
    expect(screen.getByRole('heading',{level:1}).textContent?.replace(/\s+/g,' ')).toContain(heading);
    const navigation=screen.getByRole('navigation',{name:'Main navigation'});
    expect(within(navigation).getAllByRole('link').length).toBe(13);
    await waitFor(()=>expect(screen.queryByText('This view could not be displayed')).toBeNull());
    if(endpoint) await waitFor(()=>expect(requests.some(r=> endpoint.includes('?mode=') ? r.path.includes('mode=human_vs_ai') : r.path.startsWith(endpoint))).toBe(true));
    await waitFor(()=>expect(screen.getByText('Backend connected')).toBeTruthy());
    expect(screen.queryByText('This view could not be displayed')).toBeNull();
  });
  it('surfaces real list errors and permits retry',async()=>{
    vi.stubGlobal('fetch',vi.fn(()=>Promise.resolve(new Response(JSON.stringify({detail:'Registry unavailable'}),{status:503}))));
    visit('/lab/models');
    await screen.findByText('503: Registry unavailable');
    const count=requests.length;
    vi.stubGlobal('fetch',vi.fn(mockApi));
    fireEvent.click(screen.getAllByRole('button',{name:'Retry'}).at(-1)!);
    await screen.findByText('ppo-r8');
    expect(requests.length).toBeGreaterThan(count);
  });
  it('renders explicit empty model state',async()=>{
    vi.stubGlobal('fetch',vi.fn((input:RequestInfo|URL,options?:RequestInit)=>String(input).startsWith('/api/models')?Promise.resolve(new Response('[]')):mockApi(input,options)));
    visit('/lab/models');
    await screen.findByText('No models match these filters. Try another filter or start training.');
  });
});
describe('game controls and flagship match',()=>{
  it.each([['human','ws_human'],['ai','ws_ai_ppo'],['human_vs_ai','ws_human_vs_ai'],['ai_vs_ai','ws_ai_vs_ai']])('launches %s with the documented model count',async(mode,recording)=>{
    visit(`/play/${mode}`);
    const socket=TestSocket.instances[0];act(()=>socket.open());
    const launch=screen.getByRole('button',{name:'Launch flight'});
    await waitFor(()=>expect(launch.hasAttribute('disabled')).toBe(false));
    fireEvent.click(launch);
    expect(socket.sent[0]).toMatchObject({type:'start',mode,difficulty:'normal'});
    const command=socket.sent[0] as {models:string[]};
    expect(command.models.length).toBe(mode==='human'?0:mode==='ai_vs_ai'?2:1);
    const records=trace(recording);act(()=>{records.slice(0,2).forEach(m=>socket.emit(m));});
    if(mode.startsWith('human')) {
      expect(document.activeElement?.getAttribute('aria-label')).toBe('Flap');
      fireEvent.keyDown(document.activeElement!,{key:' ',code:'Space'});
      expect(socket.sent.at(-1)).toEqual({type:'flap'});
      const count=socket.sent.length;
      fireEvent.keyDown(document.activeElement!,{key:' ',code:'Space',repeat:true});
      expect(socket.sent.length).toBe(count);
    }
  });
  it('keeps distinct pilots visible after the human crashes and displays the server winner with rematch',async()=>{
    visit('/play/human_vs_ai');const socket=TestSocket.instances[0];act(()=>socket.open());
    await waitFor(()=>expect(screen.getByRole('button',{name:'Launch flight'}).hasAttribute('disabled')).toBe(false));
    fireEvent.click(screen.getByRole('button',{name:'Launch flight'}));
    const messages=trace('ws_human_vs_ai');
    const ready=messages.find(m=>m.type==='ready') as ReadyMessage;
    const mixed=messages.find(m=>m.type==='state'&&!m.players[0].alive&&m.players[1].alive) as Snapshot;
    act(()=>{socket.emit(ready);socket.emit(mixed);});
    expect(screen.getByText('Human pilot')).toBeTruthy();expect(screen.getByText('AI pilot B')).toBeTruthy();
    expect(screen.getByText(`Crashed · ${mixed.players[0].death}`)).toBeTruthy();
    expect(screen.getByText('In flight')).toBeTruthy();expect(screen.queryByRole('button',{name:'Rematch'})).toBeNull();
    expect(screen.getAllByRole('progressbar').length).toBeGreaterThanOrEqual(2);
    const gameOver=messages.find(m=>m.type==='game_over')!;
    act(()=>socket.emit(gameOver));
    const button=screen.getByRole('button',{name:'Rematch'});fireEvent.click(button);
    expect(socket.sent.at(-1)).toEqual({type:'start',mode:ready.mode,models:ready.players.flatMap(p=>p.model_id===null?[]:[p.model_id]),difficulty:ready.difficulty,limit_ticks:ready.limit_ticks,view_width:432});
  });
  it('honors an explicitly selected archived model from the model viewer',async()=>{
    const archived=fixture<Model[]>('models').find(m=>m.status==='archived')!;
    visit(`/play/ai?model=${encodeURIComponent(archived.id)}`);
    const socket=TestSocket.instances[0];act(()=>socket.open());
    await waitFor(()=>expect(screen.getByRole('button',{name:'Launch flight'}).hasAttribute('disabled')).toBe(false));
    fireEvent.click(screen.getByRole('button',{name:'Launch flight'}));
    expect(socket.sent[0]).toMatchObject({models:[archived.id]});
  });
  it('replays a stored match and supports the speed protocol',async()=>{
    visit('/replays?match=5');const socket=TestSocket.instances[0];act(()=>socket.open());
    await screen.findByRole('combobox',{name:'Stored match'});
    fireEvent.click(screen.getByRole('button',{name:'Play recording'}));
    expect(socket.sent[0]).toEqual({type:'replay',match_id:5,view_width:432});
    act(()=>trace('ws_replay').slice(0,2).forEach(m=>socket.emit(m)));
    fireEvent.change(screen.getByRole('combobox',{name:'Playback speed'}),{target:{value:'4'}});
    expect(socket.sent.at(-1)).toEqual({type:'set_speed',speed:4});
  });
});
describe('lab mutations',()=>{
  it('posts a bounded training run with selected algorithm and seed',async()=>{
    visit('/lab/training');const button=await screen.findByRole('button',{name:'Start training'});
    fireEvent.change(screen.getByRole('spinbutton',{name:'Seed'}),{target:{value:'7'}});
    fireEvent.click(button);
    await waitFor(()=>expect(requests.some(r=>r.path==='/api/training/runs'&&r.options?.method==='POST')).toBe(true));
    const request=requests.find(r=>r.path==='/api/training/runs'&&r.options?.method==='POST')!;
    expect(JSON.parse(String(request.options?.body))).toMatchObject({algo:'dqn',seed:7,total_steps:5000,profile:'quick_smoke'});
  });
  it('posts session controls and surfaces a 409 without faking success',async()=>{
    visit('/lab/control');const pause=await screen.findByRole('button',{name:'Pause scheduling'});fireEvent.click(pause);
    await waitFor(()=>expect(requests.some(r=>r.path==='/api/research/sessions/1/pause'&&r.options?.method==='POST')).toBe(true));
    mutationError=true;
    fireEvent.click(screen.getByRole('button',{name:'Start research session'}));
    await screen.findByText(`409: ${fixture<{detail:string}>('error_409').detail}`);
    expect(screen.queryByText('Research session created.')).toBeNull();
  });
  it('inspects metadata then archives using the model status endpoint',async()=>{
    const model=fixture<Model>('model_detail');
    visit(`/lab/models?id=${model.id}`);
    const archive=await screen.findByRole('button',{name:'Archive'});fireEvent.click(archive);
    await waitFor(()=>expect(requests.some(r=>r.path===`/api/models/${model.id}/status`&&r.options?.method==='POST')).toBe(true));
    expect(screen.getByRole('link',{name:'Watch it play'}).getAttribute('href')).toContain(model.id);
  });
});


describe('round-two game regressions', () => {
  it('offers fast-forward only after every human is dead, sends speed and shows server refusal', async () => {
    visit('/play/human_vs_ai');
    const socket = TestSocket.instances[0];
    act(() => socket.open());
    const records = trace('ws_human_vs_ai');
    const ready = records.find(m => m.type === 'ready') as ReadyMessage;
    const alive = records.find(m => m.type === 'state' && m.players.every(p => p.alive)) as Snapshot;
    const crashed = records.find(m => m.type === 'state' && !m.players[0].alive && m.players[1].alive) as Snapshot;
    expect(screen.queryByRole('combobox', { name: 'Playback speed' })).toBeNull();
    act(() => { socket.emit(ready); socket.emit(alive); });
    expect(screen.queryByRole('combobox', { name: 'Playback speed' })).toBeNull();
    act(() => socket.emit({ ...alive, phase: 'paused' }));
    expect(screen.queryByRole('combobox', { name: 'Playback speed' })).toBeNull();
    act(() => socket.emit(crashed));
    const playback = screen.getByRole('combobox', { name: 'Playback speed' });
    fireEvent.change(playback, { target: { value: '16' } });
    expect(socket.sent.at(-1)).toEqual({ type: 'set_speed', speed: 16 });
    act(() => socket.emit({ type: 'error', message: 'Speed refused by server' }));
    expect(screen.getByRole('alert').textContent).toContain('Speed refused by server');
    act(() => socket.emit({ ...crashed, phase: 'paused', speed: 4 }));
    expect((screen.getByRole('combobox', { name: 'Playback speed' }) as HTMLSelectElement).value).toBe('4');
    act(() => socket.emit({ ...crashed, phase: 'over' }));
    expect(screen.queryByRole('combobox', { name: 'Playback speed' })).toBeNull();
    act(() => { socket.emit(ready); socket.emit(alive); });
    expect(screen.queryByRole('combobox', { name: 'Playback speed' })).toBeNull();
    await screen.findByText('Backend connected');
  });
  it.each([['mouse', 'human', 'ws_human'], ['touch', 'human', 'ws_human'], ['touch', 'human_vs_ai', 'ws_human_vs_ai']])('sends one flap per %s press in %s', async (pointerType, mode, recording) => {
    visit(`/play/${mode}`);
    const socket = TestSocket.instances[0];
    act(() => { socket.open(); trace(recording).slice(0, 2).forEach(m => socket.emit(m)); });
    const input = screen.getByRole('button', { name: 'Flap' });
    const press = new MouseEvent('pointerdown', { bubbles: true, cancelable: true, button: 0 });
    Object.defineProperty(press, 'pointerType', { value: pointerType });
    fireEvent(input, press);
    fireEvent.click(input, { detail: 1 });
    expect(socket.sent).toEqual([{ type: 'flap' }]);
    fireEvent(input, new MouseEvent('pointerdown', { bubbles: true, button: 2 }));
    expect(socket.sent).toHaveLength(1);
    const snapshot = trace(recording).find(m => m.type === 'state') as Snapshot;
    act(() => socket.emit({ ...snapshot, phase: 'paused' }));
    expect(screen.queryByRole('button', { name: 'Flap' })).toBeNull();
    act(() => socket.emit({ ...snapshot, phase: 'running', players: snapshot.players.map(p => ({ ...p, alive: false })) }));
    expect(screen.queryByRole('button', { name: 'Flap' })).toBeNull();
    await screen.findByText('Backend connected');
  });
  it.each([0, 1])('renders winner and lead agreement for pilot %s', async winner => {
    visit('/play/human_vs_ai');
    const socket = TestSocket.instances[0];
    const records = trace('ws_human_vs_ai');
    const ready = records.find(m => m.type === 'ready') as ReadyMessage;
    const snapshot = records.find(m => m.type === 'state') as Snapshot;
    const result = (records.find(m => m.type === 'game_over') as { result: MatchResult }).result;
    act(() => { socket.open(); socket.emit(ready); socket.emit({ ...snapshot, players: snapshot.players.map((p, i) => ({ ...p, distance: i === winner ? 100 : 50 })) }); });
    expect(screen.getByText(`${ready.players[winner].label} ${winner === 0 ? 'lead' : 'leads'} by 50 px`)).toBeTruthy();
    act(() => socket.emit({ type: 'game_over', result: { ...result, winner, draw: false } }));
    expect(screen.getByRole('heading', { name: `${result.players[winner].label} ${winner === 0 ? 'win' : 'wins'}.` })).toBeTruthy();
    await screen.findByText('Backend connected');
  });
  it.each([1, 2])('renders %s win/seed series totals with correct plurality', async count => {
    const result = { ...fixture<Series>('series'), wins_a: count, wins_b: 0, draws: count, n: count };
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, options?: RequestInit) => String(input) === '/api/competitions/series'
      ? Promise.resolve(new Response(JSON.stringify(result))) : mockApi(input, options)));
    const { container } = visit('/play/ai_vs_ai');
    const run = screen.getByRole('button', { name: 'Run series' });
    await waitFor(() => expect(run.hasAttribute('disabled')).toBe(false));
    fireEvent.click(run);
    await screen.findByText('Series completed.');
    expect(container.querySelector('.series-summary')?.textContent).toBe(`${result.model_a}: ${count} ${count === 1 ? 'win' : 'wins'} · ${result.model_b}: 0 wins · ${count} ${count === 1 ? 'draw' : 'draws'}`);
    expect(screen.getByText(new RegExp(`across ${count} ${count === 1 ? 'seed' : 'seeds'}\\.`))).toBeTruthy();
  });
});

describe('safe stop completion', () => {
  it('returns to the disabled normal stop button after the stopped status is polled', async () => {
    let run = { ...fixture<Run>('run_detail'), status: 'running', stop_requested: 0, error: null };
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, options?: RequestInit) => {
      const path = String(input);
      if (path === `/api/training/runs/${run.id}/stop` && options?.method === 'POST') {
        requests.push({ path, options });
        run = { ...run, stop_requested: 1 };
        return Promise.resolve(new Response(JSON.stringify(run)));
      }
      if (path === `/api/training/runs/${run.id}`) return Promise.resolve(new Response(JSON.stringify(run)));
      return mockApi(input, options);
    }));
    visit(`/lab/training?run=${run.id}`);
    const stop = await screen.findByRole('button', { name: 'Stop safely' });
    expect(stop.hasAttribute('disabled')).toBe(false);
    fireEvent.click(stop);
    await screen.findByRole('button', { name: 'Stopping safely…' });
    run = { ...run, status: 'stopped' };
    const finished = await screen.findByRole('button', { name: 'Stop safely' }, { timeout: 3000 });
    expect(finished.hasAttribute('disabled')).toBe(true);
    expect(screen.queryByRole('button', { name: 'Stopping safely…' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Resume run' }).hasAttribute('disabled')).toBe(false);
    expect(requests.filter(r => r.path.endsWith('/stop') && r.options?.method === 'POST')).toHaveLength(1);
  });
});

describe('tablet navigation drawer', () => {
  it('manages focus, scrolling, route changes and a resize back to desktop', async () => {
    let onResize: (() => void) | undefined;
    const media = { matches: true, addEventListener: (_: string, listener: () => void) => { onResize = listener; }, removeEventListener: vi.fn() };
    vi.stubGlobal('matchMedia', vi.fn((query: string) => query === '(max-width: 1000px)' ? media : { matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }));
    const { container } = visit('/play/human');
    const menu = screen.getByRole('button', { name: 'Open navigation' });
    const sidebar = container.querySelector('#main-navigation')!;
    expect(sidebar.hasAttribute('inert')).toBe(true);
    fireEvent.click(menu);
    expect(sidebar.hasAttribute('inert')).toBe(false);
    expect(container.querySelector('.workspace')?.hasAttribute('inert')).toBe(true);
    expect(document.body.style.overflow).toBe('hidden');
    const first = sidebar.querySelector('a')!;
    const last = sidebar.querySelectorAll('a').item(sidebar.querySelectorAll('a').length - 1);
    expect(document.activeElement).toBe(first);
    fireEvent.keyDown(first, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(last);
    fireEvent.keyDown(last, { key: 'Tab' });
    expect(document.activeElement).toBe(first);
    fireEvent.keyDown(first, { key: 'Escape' });
    expect(document.activeElement).toBe(menu);
    expect(document.body.style.overflow).toBe('');
    fireEvent.click(menu);
    fireEvent.click(within(screen.getByRole('navigation')).getByRole('link', { name: 'AI flight' }));
    await screen.findByRole('heading', { name: 'Watch intelligence take flight.' });
    expect(sidebar.hasAttribute('inert')).toBe(true);
    fireEvent.click(menu);
    act(() => { media.matches = false; onResize?.(); });
    expect(sidebar.hasAttribute('inert')).toBe(false);
    expect(container.querySelector('.nav-scrim')).toBeNull();
    expect(container.querySelector('.workspace')?.hasAttribute('inert')).toBe(false);
    expect(document.body.style.overflow).toBe('');
    await screen.findByText('Backend connected');
  });
});


describe('stage-aware game requests', () => {
  function measureStage(width: number, height: number) {
    const size = { width, height };
    let notify: (() => void) | undefined;
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function (this: HTMLElement) { return this.classList.contains('flight-stage') ? size.width : 0; });
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function (this: HTMLElement) { return this.classList.contains('flight-stage') ? size.height : 0; });
    vi.stubGlobal('ResizeObserver', class {
      constructor(callback: () => void) { notify = callback; }
      observe() {} disconnect() {}
    });
    return (width: number, height: number) => { size.width = width; size.height = height; notify?.(); };
  }
  const cases = [
    ['human', 'ws_human'], ['ai', 'ws_ai_ppo'], ['human_vs_ai', 'ws_human_vs_ai'],
    ['ai_vs_ai', 'ws_ai_vs_ai'], ['replay', 'ws_replay'],
  ] as const;
  afterEach(() => vi.useRealTimers());
  describe.each([
    ['1648 × 760 desktop', 984, 480, 1049], ['1440 × 900 desktop', 810, 560, 740],
    ['server maximum', 2000, 480, 1296], ['tablet / server minimum', 360, 600, 432],
  ] as const)('%s (modeled content box)', (_name, width, height, expected) => {
    it.each(cases)('requests the measured, clamped width in %s', async (mode, recording) => {
      measureStage(width, height);
      const { container } = visit(mode === 'replay' ? '/replays?match=5' : `/play/${mode}`);
      const socket = TestSocket.instances[0]; act(() => socket.open());
      const launch = screen.getByRole('button', { name: mode === 'replay' ? 'Play recording' : 'Launch flight' });
      await waitFor(() => expect(launch.hasAttribute('disabled')).toBe(false));
      fireEvent.click(launch);
      expect(socket.sent[0]).toMatchObject(mode === 'replay'
        ? { type: 'replay', match_id: 5, view_width: expected }
        : { type: 'start', mode, view_width: expected });
      const records = trace(recording);
      const ready = records.find(m => m.type === 'ready') as ReadyMessage;
      act(() => {
        socket.emit({ ...ready, config: { ...ready.config, view_width: expected } });
        socket.emit(records.find(m => m.type === 'state')!);
      });
      expect((container.querySelector('.world') as HTMLElement).style.aspectRatio).toBe(`${expected} / 512`);
      expect(container.querySelector('.flight-stage .scoreboard')).toBeNull();
      expect(socket.sent).toHaveLength(1);
      await screen.findByText('Backend connected');
    });
  });
  it('resizes AI playback once, keeping acknowledged settings, seed, speed and pause', async () => {
    const resize = measureStage(1000, 512);
    visit('/play/ai');
    const socket = TestSocket.instances[0]; act(() => socket.open());
    const launch = screen.getByRole('button', { name: 'Launch flight' });
    await waitFor(() => expect(launch.hasAttribute('disabled')).toBe(false));
    fireEvent.click(launch);
    const records = trace('ws_ai_ppo');
    const original = records.find(m => m.type === 'ready') as ReadyMessage;
    const ready = { ...original, config: { ...original.config, view_width: 1000 } };
    const snapshot = records.find(m => m.type === 'state') as Snapshot;
    act(() => { socket.emit(ready); socket.emit({ ...snapshot, phase: 'paused', speed: 4 }); });
    fireEvent.change(screen.getByRole('combobox', { name: 'Difficulty' }), { target: { value: 'hard' } });
    vi.useFakeTimers();
    act(() => { resize(1010, 512); vi.advanceTimersByTime(300); });
    expect(socket.sent).toHaveLength(1);
    act(() => { resize(800, 512); vi.advanceTimersByTime(100); resize(900, 512); vi.advanceTimersByTime(300); });
    expect(socket.sent).toHaveLength(2);
    expect(socket.sent.at(-1)).toEqual({ type: 'start', mode: ready.mode, models: [ready.players[0].model_id], difficulty: ready.difficulty,
      limit_ticks: ready.limit_ticks, seed: ready.seed, speed: 4, view_width: 900 });
    // A late state from the previous match must not trigger duplicate replacements.
    act(() => socket.emit({ ...snapshot, phase: 'paused', speed: 4 }));
    expect(socket.sent).toHaveLength(2);
    act(() => { socket.emit({ ...ready, config: { ...ready.config, view_width: 900 } }); socket.emit({ ...snapshot, phase: 'running', speed: 4 }); });
    expect(socket.sent.at(-1)).toEqual({ type: 'pause' });
    act(() => socket.emit({ ...snapshot, phase: 'paused', speed: 4 }));
    expect(screen.getByRole('button', { name: 'Resume' })).toBeTruthy();
  });
  it('uses the width range and world height returned by config rather than local constants', async () => {
    measureStage(1200, 600);
    const config = fixture<GameConfig>('game_config');
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, options?: RequestInit) => String(input) === '/api/game/config'
      ? Promise.resolve(new Response(JSON.stringify({ ...config, view_width_range: [600, 1000],
        difficulties: { ...config.difficulties, normal: { ...config.difficulties.normal, world_height: 450 } } })))
      : mockApi(input, options)));
    visit('/play/human');
    const socket = TestSocket.instances[0]; act(() => socket.open());
    const launch = screen.getByRole('button', { name: 'Launch flight' });
    await waitFor(() => expect(launch.hasAttribute('disabled')).toBe(false));
    fireEvent.click(launch);
    expect(socket.sent[0]).toMatchObject({ type: 'start', view_width: 900 });
  });
  it('defers resizing through a human flight and surviving AI, then rematches with the selected seed policy', async () => {
    const resize = measureStage(1000, 512);
    visit('/play/human_vs_ai');
    const socket = TestSocket.instances[0]; act(() => socket.open());
    const launch = screen.getByRole('button', { name: 'Launch flight' });
    await waitFor(() => expect(launch.hasAttribute('disabled')).toBe(false));
    fireEvent.click(launch);
    const records = trace('ws_human_vs_ai');
    const original = records.find(m => m.type === 'ready') as ReadyMessage;
    const ready = { ...original, config: { ...original.config, view_width: 1000 } };
    const initial = records.find(m => m.type === 'state') as Snapshot;
    act(() => { socket.emit(ready); socket.emit({ ...initial, phase: 'ready', tick: 0 }); });
    fireEvent.keyDown(screen.getByRole('button', { name: 'Flap' }), { key: ' ', code: 'Space' });
    vi.useFakeTimers();
    // Even before the first flap is acknowledged, resizing must not lose that input.
    act(() => { resize(800, 512); vi.advanceTimersByTime(300); });
    expect(socket.sent).toHaveLength(2);
    expect(document.querySelector('.flight-stage.width-constrained')).not.toBeNull();
    const alive = { ...initial, tick: 10, phase: 'running' as const };
    const crashed = records.find(m => m.type === 'state' && !m.players[0].alive && m.players[1].alive) as Snapshot;
    for (const snapshot of [alive, { ...alive, phase: 'paused' as const }, crashed]) {
      act(() => socket.emit(snapshot));
      expect(socket.sent).toHaveLength(2);
    }
    expect(screen.getByText('Resized view applies on your next flight.')).toBeTruthy();
    expect(screen.getByRole('combobox', { name: 'Playback speed' })).toBeTruthy();
    act(() => socket.emit(records.find(m => m.type === 'game_over')!));
    expect(socket.sent).toHaveLength(2);
    fireEvent.change(screen.getByRole('combobox', { name: 'Difficulty' }), { target: { value: 'hard' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Keep seed on rematch' }));
    fireEvent.click(screen.getByRole('button', { name: 'Rematch' }));
    expect(socket.sent.at(-1)).toEqual({ type: 'start', mode: ready.mode, models: [ready.players[1].model_id], difficulty: ready.difficulty,
      limit_ticks: ready.limit_ticks, seed: ready.seed, view_width: 800 });
  });
  it('adapts an untouched human preflight and recalculates on the restart button before debounce', async () => {
    const resize = measureStage(1000, 512);
    visit('/play/human');
    const socket = TestSocket.instances[0]; act(() => socket.open());
    const launch = screen.getByRole('button', { name: 'Launch flight' });
    await waitFor(() => expect(launch.hasAttribute('disabled')).toBe(false));
    fireEvent.click(launch);
    const records = trace('ws_human');
    const original = records.find(m => m.type === 'ready') as ReadyMessage;
    const ready = { ...original, config: { ...original.config, view_width: 1000 } };
    const snapshot = records.find(m => m.type === 'state') as Snapshot;
    act(() => { socket.emit(ready); socket.emit({ ...snapshot, phase: 'ready', tick: 0 }); });
    vi.useFakeTimers();
    act(() => { resize(900, 512); vi.advanceTimersByTime(300); });
    expect(socket.sent.at(-1)).toMatchObject({ type: 'start', seed: ready.seed, view_width: 900 });
    act(() => { socket.emit({ ...ready, config: { ...ready.config, view_width: 900 } }); socket.emit({ ...snapshot, phase: 'ready', tick: 0 }); });
    act(() => resize(800, 512));
    fireEvent.click(screen.getByRole('button', { name: 'Restart' }));
    expect(socket.sent.at(-1)).toMatchObject({ type: 'start', view_width: 800 });
    expect(socket.sent.at(-1)).not.toHaveProperty('seed');
  });
  it('re-requests the active recording on resize, replay again and watch again', async () => {
    const resize = measureStage(1000, 512);
    visit('/replays?match=5');
    const socket = TestSocket.instances[0]; act(() => socket.open());
    const launch = screen.getByRole('button', { name: 'Play recording' });
    await waitFor(() => expect(launch.hasAttribute('disabled')).toBe(false));
    fireEvent.click(launch);
    const records = trace('ws_replay');
    const original = records.find(m => m.type === 'ready') as ReadyMessage;
    const ready = { ...original, replay_of: 5, config: { ...original.config, view_width: 1000 } };
    const snapshot = records.find(m => m.type === 'state') as Snapshot;
    act(() => { socket.emit(ready); socket.emit({ ...snapshot, phase: 'running', speed: 8 }); });
    vi.useFakeTimers();
    act(() => { resize(900, 512); vi.advanceTimersByTime(300); });
    expect(socket.sent.at(-1)).toEqual({ type: 'replay', match_id: 5, speed: 8, view_width: 900 });
    act(() => { socket.emit({ ...ready, config: { ...ready.config, view_width: 900 } }); socket.emit({ ...snapshot, phase: 'running', speed: 8 }); });
    act(() => resize(850, 512));
    fireEvent.click(screen.getByRole('button', { name: 'Replay again' }));
    expect(socket.sent.at(-1)).toEqual({ type: 'replay', match_id: 5, speed: 8, view_width: 850 });
    act(() => { socket.emit({ ...ready, config: { ...ready.config, view_width: 850 } }); socket.emit({ ...snapshot, phase: 'over', speed: 8 }); socket.emit(records.find(m => m.type === 'game_over')!); });
    const count = socket.sent.length;
    act(() => { resize(800, 512); vi.advanceTimersByTime(300); });
    expect(socket.sent).toHaveLength(count);
    fireEvent.click(screen.getByRole('button', { name: 'Watch again' }));
    expect(socket.sent.at(-1)).toEqual({ type: 'replay', match_id: 5, speed: 8, view_width: 800 });
  });
});

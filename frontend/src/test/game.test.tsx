import { act, cleanup, render, renderHook, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Diagnostic, GameConfig, ReadyMessage, ServerMessage, Snapshot } from '../types';
import { bestKey, gameReducer, initialGameState, parseServerMessage } from '../game/state';
import { useGame } from '../game/useGame';
import { Diagnostics } from '../game/Diagnostics';
import { fixture } from './fixtures';
interface Trace { dir: 'server' | 'client'; msg: ServerMessage; note?: string }
const serverMessages = (name:string) => fixture<Trace[]>(name).filter(e=>e.dir === 'server').map(e=>e.msg);
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('authoritative WebSocket traces', () => {
  it.each(['ws_human','ws_ai_ppo','ws_ai_dqn','ws_human_vs_ai','ws_ai_vs_ai','ws_replay'])('handles the complete %s trace without deriving a game outcome', name => {
    let state = initialGameState;
    const messages = serverMessages(name);
    let snapshots = 0;
    messages.forEach((message,i) => {
      state = gameReducer(state,{ type:'server',message,at:i*33 });
      if (message.type === 'ready') { expect(state.ready).toEqual(message); expect(state.result).toBeNull(); expect(state.snapshot).toBeNull(); }
      if (message.type === 'state') { snapshots++; expect(state.snapshot).toEqual(message); expect(state.ready?.players.length).toBe(message.players.length); }
      if (message.type === 'game_over') expect(state.result).toEqual(message.result);
    });
    expect(snapshots).toBeGreaterThan(10);
    expect(state.result).not.toBeNull();
    expect(state.snapshot?.phase).toBe('over');
    if (name === 'ws_replay') { expect(state.ready?.replay_of).not.toBeNull(); expect(state.sessionBest).toEqual({}); }
    else {
      const ready = state.ready!;
      ready.players.forEach((_,i) => expect(state.sessionBest[bestKey(ready,i)]).toBeGreaterThanOrEqual(state.result!.players[i].score));
    }
  });
  it.each(['ws_human_vs_ai','ws_ai_vs_ai'])('%s retains a crashed bird while its opponent keeps moving', name => {
    let state = initialGameState;
    const messages = serverMessages(name);
    let death:{ index:number; y:number; distance:number; tick:number; otherDistance:number } | null = null;
    let continuation = false;
    for (const message of messages) {
      state = gameReducer(state,{ type:'server',message,at:0 });
      if (message.type !== 'state') continue;
      const index = message.players.findIndex(p=>!p.alive);
      if (index < 0 || !message.players.some(p=>p.alive)) continue;
      expect(state.result).toBeNull();
      if (message.phase === 'over') {
        expect(message.tick).toBe(state.ready?.limit_ticks);
        continue;
      }
      expect(state.snapshot?.phase).toBe('running');
      if (!death) death = { index, y:message.players[index].y, distance:message.players[index].distance, tick:message.tick, otherDistance:message.players[1-index].distance };
      if (message.tick > death.tick) {
        expect(state.snapshot?.players[death.index].y).toBe(death.y);
        expect(state.snapshot?.players[death.index].distance).toBe(death.distance);
        expect(state.snapshot?.players[1-death.index].distance).toBeGreaterThan(death.otherDistance);
        continuation = true;
      }
    }
    expect(continuation).toBe(true);
  });
  it('replaces records on ready, clears a stopped match, and preserves server errors', () => {
    let state = serverMessages('ws_human').reduce((s,message) => gameReducer(s,{type:'server',message,at:1}),initialGameState);
    expect(state.result).not.toBeNull();
    const ready = serverMessages('ws_ai_dqn').find(m=>m.type==='ready')!;
    state = gameReducer(state,{type:'server',message:ready,at:2});
    expect(state.result).toBeNull(); expect(state.snapshot).toBeNull();
    state = gameReducer(state,{type:'pending'});
    const error = serverMessages('ws_error')[0];
    state = gameReducer(state,{type:'server',message:error,at:3});
    expect(state.error).toBe('unknown model does-not-exist'); expect(state.pending).toBe(false);
    state = gameReducer(state,{type:'server',message:{type:'stopped'},at:4});
    expect(state.ready).toBeNull(); expect(state.snapshot).toBeNull();
  });
  it('preserves pause and resume exactly as sent and removes absent diagnostics', () => {
    const ready = serverMessages('ws_ai_ppo').find(m=>m.type==='ready') as ReadyMessage;
    const snapshot = serverMessages('ws_ai_ppo').find(m=>m.type==='state' && m.players[0].diag) as Snapshot;
    let state = gameReducer(initialGameState,{type:'server',message:ready,at:0});
    state = gameReducer(state,{type:'server',message:{...snapshot,phase:'paused'},at:1});
    expect(state.snapshot?.phase).toBe('paused');
    state = gameReducer(state,{type:'server',message:{...snapshot,phase:'running',players:snapshot.players.map(({diag: _diag,...p})=>p)},at:2});
    expect(state.snapshot?.players[0].diag).toBeUndefined();
  });
  it('rejects malformed or unknown messages', () => {
    expect(()=>parseServerMessage('bad json')).toThrow();
    expect(()=>parseServerMessage('{"type":"invented"}')).toThrow('Unrecognized');
    expect(()=>parseServerMessage('null')).toThrow();
  });
});

describe('real policy diagnostics', () => {
  const config = fixture<GameConfig>('game_config');
  it.each([['ws_ai_dqn','Q-values','Action probabilities'],['ws_ai_ppo','Action probabilities','Q-values']])('%s renders only the returned diagnostic', (name,present,absent) => {
    const state = serverMessages(name).find(m=>m.type==='state' && m.players[0].diag) as Snapshot;
    const diag = state.players[0].diag as Diagnostic;
    render(<Diagnostics label="Policy" diag={diag} names={config.observation.names} actions={config.actions}/>);
    expect(screen.getByRole('heading',{name:present})).toBeTruthy();
    expect(screen.queryByRole('heading',{name:absent})).toBeNull();
    config.observation.names.forEach(name=>expect(screen.getByText(name)).toBeTruthy());
    expect(screen.getByText('Chosen action')).toBeTruthy();
  });
  it('does not invent values when no AI acted on the tick', () => {
    render(<Diagnostics label="Crashed model" names={config.observation.names}/>);
    expect(screen.getByText('No decision on this tick for Crashed model.')).toBeTruthy();
    expect(screen.queryByText('Q-values')).toBeNull(); expect(screen.queryByText('Action probabilities')).toBeNull();
  });
});

export class SocketMock {
  static CONNECTING=0; static OPEN=1; static instances:SocketMock[]=[];
  readyState=0; sent:string[]=[];
  onopen:(()=>void)|null=null; onclose:(()=>void)|null=null; onerror:(()=>void)|null=null; onmessage:((e:{data:string})=>void)|null=null;
  constructor(public url:string){SocketMock.instances.push(this);}
  open(){this.readyState=1;this.onopen?.();}
  message(msg:ServerMessage){this.onmessage?.({data:JSON.stringify(msg)});}
  send(value:string){this.sent.push(value);}
  close(){this.readyState=3;this.onclose?.();}
}
describe('WebSocket transport', () => {
  it('sends real protocol commands and keeps the last snapshot on connection loss', () => {
    SocketMock.instances=[]; vi.stubGlobal('WebSocket',SocketMock);
    const {result,unmount}=renderHook(()=>useGame());
    const socket=SocketMock.instances[0];
    expect(socket.url).toContain('/ws/game');
    act(()=>{socket.open();});
    act(()=>{result.current.send({type:'start',mode:'human',models:[],difficulty:'normal'});});
    expect(JSON.parse(socket.sent[0])).toEqual({type:'start',mode:'human',models:[],difficulty:'normal'});
    act(()=>{serverMessages('ws_human').slice(0,2).forEach(m=>socket.message(m));});
    const snapshot=result.current.state.snapshot;
    expect(snapshot).not.toBeNull();
    act(()=>{result.current.send({type:'flap'});socket.close();});
    expect(result.current.state.connection).toBe('disconnected');
    expect(result.current.state.snapshot).toEqual(snapshot);
    let success=true; act(()=>{success=result.current.send({type:'restart'});});
    expect(success).toBe(false); expect(result.current.state.error).toContain('disconnected');
    unmount(); expect(socket.onmessage).toBeNull();
  });
});

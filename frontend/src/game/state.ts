import type { MatchResult, ReadyMessage, ServerMessage, Snapshot } from '../types';
export interface GameState {
  connection: 'connecting' | 'connected' | 'disconnected'; ready: ReadyMessage | null;
  snapshot: Snapshot | null; previous: Snapshot | null; receivedAt: number;
  result: MatchResult | null; error: string | null; pending: boolean; sessionBest: Record<string, number>;
}
export const initialGameState: GameState = { connection: 'connecting', ready: null, snapshot: null, previous: null, receivedAt: 0, result: null, error: null, pending: false, sessionBest: {} };
export type GameEvent = { type: 'server'; message: ServerMessage; at: number }
  | { type: 'connection'; value: GameState['connection'] }
  | { type: 'error'; message: string } | { type: 'pending' } | { type: 'reset' };
export function gameReducer(state: GameState, event: GameEvent): GameState {
  if (event.type === 'reset') return { ...initialGameState, sessionBest: state.sessionBest };
  if (event.type === 'connection') return { ...state, connection: event.value, pending: false };
  if (event.type === 'error') return { ...state, error: event.message, pending: false };
  if (event.type === 'pending') return { ...state, error: null, pending: true };
  const message = event.message;
  switch (message.type) {
    case 'ready': return { ...state, ready: message, snapshot: null, previous: null, result: null, error: null, pending: false };
    case 'state': {
      if (!state.ready) return state;
      const best = { ...state.sessionBest };
      // Replays do not set live session records; identity and difficulty keep records separate.
      if (state.ready.replay_of === null) message.players.forEach((p, i) => {
        const key = bestKey(state.ready!, i);
        best[key] = Math.max(best[key] ?? 0, p.score);
      });
      return { ...state, previous: state.snapshot, snapshot: message, receivedAt: event.at, sessionBest: best, pending: false };
    }
    case 'game_over': return { ...state, result: message.result, pending: false };
    case 'error': return { ...state, error: message.message, pending: false };
    case 'stopped': return { ...state, ready: null, snapshot: null, previous: null, result: null, error: null, pending: false };
  }
}
export function bestKey(ready: ReadyMessage, index: number) { return `${ready.players[index]?.model_id ?? 'human'}:${ready.difficulty}`; }
export function parseServerMessage(raw: string): ServerMessage {
  const msg: unknown = JSON.parse(raw);
  if (!msg || typeof msg !== 'object' || !('type' in msg) || !['ready', 'state', 'game_over', 'error', 'stopped'].includes(String(msg.type))) throw new Error('Unrecognized game message. Reconnect to the server.');
  return msg as ServerMessage;
}

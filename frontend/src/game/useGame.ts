import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { gameReducer, initialGameState, parseServerMessage } from './state';
import type { GameCommand } from '../types';
export function useGame() {
  const [state, dispatch] = useReducer(gameReducer, initialGameState);
  const [revision, setRevision] = useState(0);
  const socket = useRef<WebSocket | null>(null);
  const commandTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    dispatch({ type: 'reset' });
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/game`);
    socket.current = ws;
    const timeout = setTimeout(() => {
      if (ws.readyState === WebSocket.CONNECTING) { dispatch({ type: 'error', message: 'Connection timed out. Check the backend and reconnect.' }); ws.close(); }
    }, 12000);
    ws.onopen = () => { clearTimeout(timeout); dispatch({ type: 'connection', value: 'connected' }); };
    ws.onmessage = e => {
      try {
        const message = parseServerMessage(String(e.data));
        clearTimeout(commandTimer.current);
        dispatch({ type: 'server', message, at: performance.now() });
      } catch (e) { dispatch({ type: 'error', message: e instanceof Error ? e.message : 'Invalid game message.' }); }
    };
    ws.onerror = () => dispatch({ type: 'error', message: 'Cannot reach the game server. Check the backend, then reconnect.' });
    ws.onclose = () => { clearTimeout(timeout); clearTimeout(commandTimer.current); dispatch({ type: 'connection', value: 'disconnected' }); };
    return () => {
      clearTimeout(timeout); clearTimeout(commandTimer.current);
      ws.onopen = null; ws.onmessage = null; ws.onerror = null; ws.onclose = null;
      ws.close(); socket.current = null;
    };
  }, [revision]);
  const send = useCallback((command: GameCommand) => {
    if (socket.current?.readyState !== WebSocket.OPEN) {
      dispatch({ type: 'error', message: 'The game is disconnected. Reconnect before sending an input.' }); return false;
    }
    if (command.type !== 'flap') {
      dispatch({ type: 'pending' }); clearTimeout(commandTimer.current);
      commandTimer.current = setTimeout(() => dispatch({ type: 'error', message: 'The server has not acknowledged this command. Reconnect if it remains unresponsive.' }), 15000);
    }
    socket.current.send(JSON.stringify(command)); return true;
  }, []);
  return { state, send, reconnect: () => setRevision(x => x + 1) };
}

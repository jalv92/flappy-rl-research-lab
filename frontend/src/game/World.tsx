import { useEffect, useRef, type CSSProperties } from 'react';
import type { GameState } from './state';
import { useSettings } from '../settings';
import { CanyonArt } from './art';

// Rendering only. All geometry, positions, deaths and scores come from the server.
export function World({ state, flapPulse, onFlap, canFlap }: { state: GameState; flapPulse: number; onFlap: () => void; canFlap: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const input = useRef<HTMLButtonElement>(null);
  const live = useRef({ state, flapPulse });
  live.current = { state, flapPulse };
  const { reduce } = useSettings();
  const hasWorld = !!state.ready && !!state.snapshot;
  useEffect(() => {
    if (hasWorld && canFlap) input.current?.focus({ preventScroll: true });
  }, [state.ready, hasWorld, canFlap]);
  useEffect(() => {
    if (!hasWorld || !canvas.current) return;
    const element = canvas.current;
    const ctx = element.getContext('2d');
    if (!ctx) return;
    let frame = 0;
    const draw = (now: number) => {
      const { state: game, flapPulse: pulse } = live.current;
      const { ready, snapshot: s, previous: p } = game;
      if (!ready || !s) { frame = requestAnimationFrame(draw); return; }
      const c = ready.config, w = c.view_width, h = c.world_height;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      if (element.width !== Math.round(w * ratio) || element.height !== Math.round(h * ratio)) { element.width = Math.round(w * ratio); element.height = Math.round(h * ratio); }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const t = !reduce && s.phase === 'running' && p?.phase === 'running' && game.connection === 'connected' ? Math.min(1, Math.max(0, (now - game.receivedAt) / (1000 / c.tick_rate))) : 1;
      const lerp = (a: number, b: number) => a + (b - a) * t;
      const scroll = p ? lerp(p.scroll_x, s.scroll_x) : s.scroll_x;
      const sky = ctx.createLinearGradient(0, 0, 0, h); sky.addColorStop(0, '#244e58'); sky.addColorStop(.7, '#91c1b2'); sky.addColorStop(1, '#c2cdb0');
      ctx.fillStyle = sky; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#e9e4bb'; ctx.globalAlpha = .75; ctx.beginPath(); ctx.arc(w * .74, h * .23, 43, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
      for (let layer = 0; layer < 2; layer++) {
        const drift = reduce ? 0 : (scroll * (.08 + layer * .05)) % 180;
        ctx.fillStyle = layer ? '#426e70' : '#64948b'; ctx.globalAlpha = layer ? .55 : .65;
        ctx.beginPath(); ctx.moveTo(-180, h);
        for (let i = -1; i <= Math.ceil(w / 180) + 1; i++) { const x = i * 180 - drift; ctx.lineTo(x, h * (.73 + layer * .12)); ctx.lineTo(x + 55, h * (.43 + layer * .16)); ctx.lineTo(x + 103, h * (.63 + layer * .17)); }
        ctx.lineTo(w + 180, h); ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
      for (const obstacle of s.obstacles) {
        const old = p?.obstacles.find(o => o.i === obstacle.i);
        const x = old ? lerp(old.x, obstacle.x) : obstacle.x;
        const ow = c.obstacle_width;
        const paintBlock = (y: number, bh: number, top: boolean) => {
          const rock = ctx.createLinearGradient(x, y, x + ow, y); rock.addColorStop(0, '#b96b60'); rock.addColorStop(.55, '#dd9277'); rock.addColorStop(1, '#f0b690');
          ctx.fillStyle = rock; ctx.fillRect(x, y, ow, bh);
          ctx.fillStyle = '#905358'; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + ow * .3, y); ctx.lineTo(x + ow * .47, y + bh * .65); ctx.lineTo(x, y + bh); ctx.closePath(); ctx.fill();
          ctx.strokeStyle = '#f9d0a0'; ctx.lineWidth = 3; const edge = top ? y + bh : y; ctx.beginPath(); ctx.moveTo(x + 1, edge); ctx.lineTo(x + ow - 1, edge); ctx.stroke();
          ctx.strokeStyle = '#f5bb91'; ctx.lineWidth = 1; ctx.globalAlpha = .35;
          for (let j = 30; j < bh; j += 57) { ctx.beginPath(); ctx.moveTo(x + ow * .4, y + j); ctx.lineTo(x + ow, y + j + 9); ctx.stroke(); }
          ctx.globalAlpha = 1;
        };
        paintBlock(0, obstacle.gap_top, true); paintBlock(obstacle.gap_bottom, h - obstacle.gap_bottom, false);
      }
      s.players.forEach((bird, i) => {
        const old = p?.players[i]; const y = old && bird.alive ? lerp(old.y, bird.y) : bird.y;
        const r = c.bird_radius; const rival = i === 1; const color = !bird.alive ? '#a6adb0' : rival ? '#ffa583' : '#b7f8d9';
        const localFlap = ready.players[i]?.kind === 'human' && now - pulse < 140;
        ctx.save(); ctx.translate(c.bird_x, y);
        // Velocity is used only for a decorative tilt, never to advance position.
        ctx.rotate(reduce || !bird.alive ? 0 : Math.max(-.3, Math.min(.5, bird.vy / 1000)));
        ctx.fillStyle = color; ctx.strokeStyle = '#21454d'; ctx.lineWidth = 1.5;
        ctx.beginPath();
        if (rival) { ctx.moveTo(0, -r); ctx.lineTo(r, 0); ctx.lineTo(0, r); ctx.lineTo(-r, 0); ctx.closePath(); } else ctx.arc(0, 0, r, 0, 2 * Math.PI);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = bird.alive ? (rival ? '#ffe5bd' : '#eaffde') : '#c3c8c9';
        ctx.beginPath(); ctx.moveTo(-r * .85, 0); ctx.lineTo(-r * .2, -r * (localFlap && !reduce ? 1.9 : 1.25)); ctx.lineTo(r * .5, r * .3); ctx.closePath(); ctx.fill();
        ctx.fillStyle = '#244951'; ctx.beginPath(); ctx.arc(r * .5, -r * .17, 2, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = color; ctx.globalAlpha = .8; ctx.lineWidth = 2; ctx.setLineDash(rival ? [3, 3] : []); ctx.beginPath(); ctx.arc(0, 0, r + (rival ? 6 : 3), 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
        ctx.font = 'bold 10px system-ui'; ctx.textAlign = rival ? 'left' : 'right';
        const name = ready.players[i]?.kind === 'human' ? 'YOU' : s.players.length > 1 ? (i === 0 ? 'AI A' : 'AI B') : 'AI';
        const label = bird.alive ? name : `${name} ×`; const lx = c.bird_x + (rival ? 20 : -20); const ly = y + (rival ? 27 : -22);
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = '#16343ee8'; ctx.fillRect(lx - (rival ? 4 : textWidth + 4), ly - 12, textWidth + 8, 18);
        ctx.fillStyle = color; ctx.fillText(label, lx, ly);
        if (!bird.alive) { ctx.strokeStyle = '#e7edef'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(c.bird_x-5,y-5); ctx.lineTo(c.bird_x+5,y+5); ctx.moveTo(c.bird_x+5,y-5); ctx.lineTo(c.bird_x-5,y+5); ctx.stroke(); }
      });
      ctx.fillStyle = '#193f49'; ctx.fillRect(0, h-3, w, 3); ctx.fillRect(0, 0, w, 2);
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [hasWorld, reduce]);
  return <div className={`world ${canFlap ? 'playable' : ''}`} style={state.ready ? { aspectRatio: `${state.ready.config.view_width}/${state.ready.config.world_height}`, '--world-aspect': state.ready.config.view_width / state.ready.config.world_height } as CSSProperties : undefined}>
    {hasWorld ? <canvas ref={canvas} role="img" aria-label="Live flight. Player positions and obstacles streamed by the server."/> : <CanyonArt/>}
    {canFlap && <button ref={input} className="world-input" aria-label="Flap" onPointerDown={e => { if (e.button === 0) { e.preventDefault(); e.currentTarget.focus({ preventScroll: true }); onFlap(); } }} onClick={e => { if (e.detail === 0) onFlap(); }} onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); if (!e.repeat) onFlap(); } }}/>}
  </div>;
}

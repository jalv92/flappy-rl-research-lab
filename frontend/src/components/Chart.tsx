import { useId, useState } from 'react';
import { Empty, fmt, Table } from './ui';
export interface ChartPoint { x: number; y: number; lo?: number; hi?: number }
export interface ChartSeries { name: string; color: string; points: ChartPoint[] }
export const chartColors = ['#9de6ce', '#ee9c7f', '#a5bce8', '#e7cd91', '#cfb1cf', '#8bc9db', '#c1d77a', '#f08fbb'];
const dashes = ['', '8 4', '3 3', '12 3 3 3', '2 5', '12 4', '8 3 2 3', '5 3 5 7'];

// Choose precision for the whole axis, including tiny and constant measured ranges.
export function axisTicks(min: number, max: number) {
  const values = [...new Set(min === max ? [min] : Array.from({ length: 5 }, (_, i) => min + (max - min) * i / 4))];
  const step = (max - min) / 4;
  const scientific = Math.max(Math.abs(min), Math.abs(max)) >= 1e7 || (step > 0 && step < 1e-5);
  const firstPrecision = scientific ? 3 : Math.max(0, Math.min(20, Math.ceil(-Math.log10(step || 1))));
  for (let precision = firstPrecision; precision <= (scientific ? 17 : 20); precision++) {
    const format = new Intl.NumberFormat('en', scientific
      ? { notation: 'scientific', maximumSignificantDigits: precision }
      : { maximumFractionDigits: precision });
    const labels = values.map(value => format.format(value === 0 ? 0 : value));
    // Single values still need enough precision to represent a small nonzero value.
    if (new Set(labels).size === values.length && values.every((value, i) => value === 0 || !['0', '-0'].includes(labels[i]))) {
      return values.map((value, i) => ({ value, label: labels[i] }));
    }
  }
  return values.map(value => ({ value, label: String(value) }));
}
function Marker({ x, y, radius, index, color }: { x: number; y: number; radius: number; index: number; color: string }) {
  if (index % 4 === 1) return <rect x={x-radius} y={y-radius} width={radius*2} height={radius*2} fill={color}/>;
  if (index % 4 === 2) return <path d={`M${x},${y-radius*1.4}l${radius*1.4},${radius*1.4}l${-radius*1.4},${radius*1.4}l${-radius*1.4},${-radius*1.4}Z`} fill={color}/>;
  if (index % 4 === 3) return <path d={`M${x},${y-radius*1.5}l${radius*1.4},${radius*2.5}h${-radius*2.8}Z`} fill={color}/>;
  return <circle cx={x} cy={y} r={radius} fill={color}/>;
}
export function Chart({ series, yLabel, xLabel = 'Environment steps', compact = false }: { series: ChartSeries[]; yLabel: string; xLabel?: string; compact?: boolean }) {
  const [hidden, setHidden] = useState<string[]>([]);
  const [focused, setFocused] = useState('');
  const [hovered, setHovered] = useState('');
  const id = useId();
  // Keep style identity when another series is hidden.
  const visible = series.map((s, index) => ({ ...s, index })).filter(s => !hidden.includes(s.name));
  const focus = [hovered, focused].find(name => visible.some(s => s.name === name)) ?? '';
  const ordered = [...visible].sort((a, b) => Number(a.name === focus) - Number(b.name === focus));
  const all = visible.flatMap(s => s.points).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  const w = 700, h = compact ? 200 : 280, left = 78, right = 30, top = 25, bottom = 45;
  const minX = all.length ? Math.min(...all.map(p => p.x)) : 0;
  const maxX = all.length ? Math.max(...all.map(p => p.x)) : 1;
  const minY = all.length ? Math.min(0, ...all.map(p => p.lo ?? p.y)) : 0;
  const maxY = all.length ? Math.max(...all.map(p => p.hi ?? p.y)) : 1;
  const x = (v: number) => left + (v - minX) / (maxX - minX || 1) * (w - left - right);
  const y = (v: number) => h - bottom - (v - minY) / (maxY - minY || 1) * (h - top - bottom);
  const path = (points: ChartPoint[], key: 'y' | 'lo' | 'hi' = 'y') => points.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p[key] ?? p.y)}`).join(' ');
  return <div className="chart">
    {series.length > 1 && <div className="chart-focus"><label>Focus series <select value={visible.some(s => s.name === focused) ? focused : ''} onChange={e => setFocused(e.target.value)}><option value="">All series</option>{visible.map(s => <option key={s.name} value={s.name}>{s.name}</option>)}</select></label><span>Hover a legend or choose a group to isolate its confidence band.</span></div>}
    <div className="chart-legend">{series.map((s, index) => <label key={s.name} style={{ color: s.color }} onPointerEnter={e => { if (e.pointerType === 'mouse') setHovered(s.name); }} onPointerLeave={() => setHovered('')} onFocus={() => setHovered(s.name)} onBlur={() => setHovered('')}><input type="checkbox" checked={!hidden.includes(s.name)} onChange={() => setHidden(old => old.includes(s.name) ? old.filter(n => n !== s.name) : [...old, s.name])}/><svg viewBox="0 0 36 14" aria-hidden="true"><path d="M0,7H36" fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={dashes[index % dashes.length]}/><Marker x={18} y={7} radius={3} index={index} color={s.color}/></svg><span>{s.name}</span></label>)}</div>
    {!all.length ? <Empty>{series.some(s => s.points.length) ? 'Select a series to show.' : 'No measured points yet. The chart will appear when the trainer logs data.'}</Empty> : <svg className="chart-plot" viewBox={`0 0 ${w} ${h}`} role="img" aria-labelledby={id}><title id={id}>{yLabel} by {xLabel}. Exact values are in the table below.</title>
      {axisTicks(minY, maxY).map(({ value, label }) => <g key={value}><line x1={left} x2={w-right} y1={y(value)} y2={y(value)} stroke="var(--border)"/><text data-axis="y" x={left-10} y={y(value)+4} textAnchor="end">{label}</text></g>)}
      {axisTicks(minX, maxX).map(({ value, label }) => <text data-axis="x" key={value} x={x(value)} y={h-24} textAnchor="middle">{label}</text>)}
      <text x={left} y={13}>{yLabel}</text><text x={w/2} y={h-3} textAnchor="middle">{xLabel}</text>
      {ordered.map(s => <g key={s.name} data-series={s.name} opacity={focus && focus !== s.name ? .16 : 1}>
        {s.points.every(p => p.lo !== undefined && p.hi !== undefined) && s.points.length > 0 && <path className="chart-band" d={`${path(s.points, 'hi')} ${path([...s.points].reverse(), 'lo').replace('M','L')}Z`} fill={s.color} opacity={focus ? (focus === s.name ? .24 : .025) : .07}/>}<path className="chart-line" d={path(s.points)} fill="none" stroke={s.color} strokeWidth={focus === s.name ? 3 : 2.3} strokeDasharray={dashes[s.index % dashes.length]}/>
        {s.points.map((p, n) => <g key={n}><title>{s.name}: {fmt(p.y, 4)} at {fmt(p.x, 0)}{p.lo !== undefined ? `; CI ${fmt(p.lo)}–${fmt(p.hi)}` : ''}</title><Marker x={x(p.x)} y={y(p.y)} radius={s.points.length < 30 ? 3 : 1.5} index={s.index} color={s.color}/></g>)}
      </g>)}
    </svg>}
    {series.some(s => s.points.length > 0) && <details className="chart-data"><summary>View exact data</summary><Table headers={['Series', xLabel, yLabel, 'CI low', 'CI high']}>{series.flatMap(s => s.points.map((p, i) => <tr key={`${s.name}-${i}`}><td>{s.name}</td><td>{fmt(p.x, 0)}</td><td>{fmt(p.y, 5)}</td><td>{fmt(p.lo, 5)}</td><td>{fmt(p.hi, 5)}</td></tr>))}</Table></details>}
  </div>;
}

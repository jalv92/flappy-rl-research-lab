import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { axisTicks, Chart, chartColors } from '../components/Chart';
import { pilotVerb } from '../game/copy';
import { plural } from '../components/ui';
afterEach(cleanup);

describe('axis labels preserve small measured differences', () => {
  it.each([[0, .01], [-1, -.99], [.009, .011], [1, 2], [0, 0], [-1, -1], [1e-10, 2e-10], [1e9, 1e9+1]])('formats %s to %s without repeated labels', (min, max) => {
    const ticks = axisTicks(min, max);
    expect(new Set(ticks.map(t => t.label)).size).toBe(ticks.length);
    expect(ticks[0].value).toBe(min);
    expect(ticks.at(-1)?.value).toBe(max);
    if (min !== max) expect(ticks).toHaveLength(5);
  });
  it.each([[0, .01], [-1, -.99], [0, 0]])('renders unique labels on both axes for %s to %s', (min, max) => {
    const { container } = render(<Chart yLabel="Loss" series={[{ name: 'Loss', color: chartColors[0], points: [{ x: 1, y: min }, { x: min === max ? 1 : 2, y: max }] }]}/>);
    for (const axis of ['x', 'y']) {
      const labels = [...container.querySelectorAll(`[data-axis="${axis}"]`)].map(el => el.textContent);
      expect(labels.length).toBeGreaterThan(0);
      expect(new Set(labels).size).toBe(labels.length);
    }
  });
});

describe('eight-group comparison readability', () => {
  const series = chartColors.map((color, i) => ({ name: `Policy ${i+1}`, color, points: [{ x: 0, y: 1, lo: 0, hi: 2 }, { x: 10, y: 2, lo: 1, hi: 3 }] }));
  it('keeps distinct colour and dash styles when another group is hidden', () => {
    const { container } = render(<Chart yLabel="Score" series={series}/>);
    const styles = [...container.querySelectorAll('.chart-line')].map(el => `${el.getAttribute('stroke')}/${el.getAttribute('stroke-dasharray')}`);
    expect(new Set(styles).size).toBe(8);
    const before = container.querySelector('[data-series="Policy 8"] .chart-line')?.getAttribute('stroke-dasharray');
    fireEvent.click(screen.getByRole('checkbox', { name: 'Policy 1' }));
    expect(container.querySelector('[data-series="Policy 8"] .chart-line')?.getAttribute('stroke-dasharray')).toBe(before);
    expect(container.querySelector('[data-series="Policy 1"]')).toBeNull();
    expect(container.querySelectorAll('tbody tr')).toHaveLength(16);
  });
  it('lets touch and keyboard users focus a band, then restore all groups', () => {
    const { container } = render(<Chart yLabel="Score" series={series}/>);
    fireEvent.change(screen.getByRole('combobox', { name: 'Focus series' }), { target: { value: 'Policy 8' } });
    expect(container.querySelector('[data-series="Policy 8"]')?.getAttribute('opacity')).toBe('1');
    expect(container.querySelector('[data-series="Policy 1"]')?.getAttribute('opacity')).toBe('0.16');
    expect(container.querySelector('[data-series="Policy 8"] .chart-band')?.getAttribute('opacity')).toBe('0.24');
    expect(container.querySelector('.chart-plot')?.lastElementChild?.getAttribute('data-series')).toBe('Policy 8');
    fireEvent.change(screen.getByRole('combobox', { name: 'Focus series' }), { target: { value: '' } });
    expect([...container.querySelectorAll('[data-series]')].every(el => el.getAttribute('opacity') === '1')).toBe(true);
  });
});

describe('pilot and count grammar', () => {
  it.each([['You', 'You win', 'You lead'], ['ppo-r8', 'ppo-r8 wins', 'ppo-r8 leads']])('uses correct agreement for %s', (name, win, lead) => {
    expect(pilotVerb(name, 'win')).toBe(win);
    expect(pilotVerb(name, 'lead')).toBe(lead);
  });
  it.each([[0, 'wins', 'losses', 'draws'], [1, 'win', 'loss', 'draw'], [2, 'wins', 'losses', 'draws']])('pluralizes %s results', (count, win, loss, draw) => {
    expect(plural(count, 'win')).toBe(win);
    expect(plural(count, 'loss', 'losses')).toBe(loss);
    expect(plural(count, 'draw')).toBe(draw);
  });
});

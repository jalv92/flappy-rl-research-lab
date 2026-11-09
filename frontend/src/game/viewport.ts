import { useLayoutEffect, useRef, useState } from 'react';
import type { WorldConfig } from '../types';

export interface StageSize { width: number; height: number }

export function readStageSize(stage: HTMLDivElement): StageSize {
  const rect = stage.getBoundingClientRect();
  const css = getComputedStyle(stage);
  // Measure the intended content box, not temporary scrollbars from a deferred
  // human resize. Otherwise those scrollbars can keep the new view too wide.
  return {
    width: rect.width > 0 ? rect.width - (parseFloat(css.borderLeftWidth) || 0) - (parseFloat(css.borderRightWidth) || 0) : stage.clientWidth,
    height: rect.height > 0 ? rect.height - (parseFloat(css.borderTopWidth) || 0) - (parseFloat(css.borderBottomWidth) || 0) : stage.clientHeight,
  };
}

// Only the visible window changes; simulation geometry remains server-owned.
export function viewWidthForStage(size: StageSize, world: WorldConfig, range?: [number, number]) {
  const [min, max] = range ?? [world.view_width, world.view_width];
  const width = size.width > 0 && size.height > 0
    ? Math.floor(world.world_height * size.width / size.height) : world.view_width;
  return Math.max(min, Math.min(max, width));
}

export function viewWidthChanged(current: number, next: number) {
  return Math.abs(next - current) >= Math.max(24, current * .05);
}

export function useStageSize() {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<StageSize>({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const stage = ref.current;
    if (!stage) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const measure = () => setSize(previous => {
      const next = readStageSize(stage);
      return previous.width === next.width && previous.height === next.height ? previous : next;
    });
    const schedule = () => { clearTimeout(timer); timer = setTimeout(measure, 250); };
    measure();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedule);
    observer?.observe(stage);
    window.addEventListener('resize', schedule);
    return () => { clearTimeout(timer); observer?.disconnect(); window.removeEventListener('resize', schedule); };
  }, []);
  return { ref, size };
}

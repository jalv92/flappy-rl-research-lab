import { useCallback, useEffect, useRef, useState } from 'react';

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}
export function errorText(error: unknown): string { return error instanceof Error ? error.message : String(error); }
export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  const text = await response.text();
  let body: unknown;
  try { body = text ? JSON.parse(text) : null; } catch { throw new ApiError(response.status, `The server returned an unreadable response (${response.status}).`); }
  if (!response.ok) {
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : response.statusText;
    const message = typeof detail === 'string' ? detail : JSON.stringify(detail);
    throw new ApiError(response.status, `${response.status}: ${message}`);
  }
  return body as T;
}
export function post<T>(path: string, body?: unknown): Promise<T> { return request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) }); }
export interface Resource<T> { data: T | null; loading: boolean; error: string | null; refresh: () => void; updated: number | null }
export function useResource<T>(path: string | null, pollMs = 0): Resource<T> {
  const [value, setValue] = useState<{ path: string | null; data: T | null; loading: boolean; error: string | null; updated: number | null }>({ path, data: null, loading: !!path, error: null, updated: null });
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision(x => x + 1), []);
  useEffect(() => {
    if (!path) { setValue({ path, data: null, loading: false, error: null, updated: null }); return; }
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setValue(old => ({ path, data: old.path === path ? old.data : null, loading: true, error: null, updated: old.path === path ? old.updated : null }));
    const load = async () => {
      try {
        const data = await request<T>(path, { signal: controller.signal });
        if (!controller.signal.aborted) setValue({ path, data, loading: false, error: null, updated: Date.now() });
      } catch (error) {
        if (!controller.signal.aborted) setValue(old => ({ ...old, loading: false, error: errorText(error) }));
      } finally {
        if (pollMs && !controller.signal.aborted) timer = setTimeout(load, pollMs);
      }
    };
    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [path, pollMs, revision]);
  return { ...(value.path === path ? value : { data: null, loading: !!path, error: null, updated: null }), refresh };
}
export function useAction() {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const locked = useRef(false);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const execute = async <T,>(task: () => Promise<T>, onSuccess?: (value: T) => void, message = 'Saved.') => {
    if (locked.current) return;
    locked.current = true; setPending(true); setError(null); setSuccess(null);
    try { const value = await task(); if (mounted.current) { onSuccess?.(value); setSuccess(message); } }
    catch (e) { if (mounted.current) setError(errorText(e)); }
    finally { locked.current = false; if (mounted.current) setPending(false); }
  };
  return { pending, error, success, execute };
}
export function query(values: Record<string, string | number | undefined>) {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => { if (value !== undefined && value !== '') params.set(key, String(value)); });
  return params.toString();
}

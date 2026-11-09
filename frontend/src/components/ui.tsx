import type { ReactNode } from 'react';
import { AlertCircle, ArrowUpRight, RefreshCw } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { Resource } from '../api';

export const fmt = (value: number | null | undefined, digits = 1) => value == null || !Number.isFinite(value) ? '—' : new Intl.NumberFormat('en', { maximumFractionDigits: digits }).format(value);
export const compact = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
export const time = (value: number | null | undefined) => value == null ? '—' : new Date(value * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
export const humanize = (value: string) => value.replaceAll('_', ' ');
export const plural = (count: number | null | undefined, singular: string, multiple = `${singular}s`) => count === 1 ? singular : multiple;
export function PageHeader({ title, description, children }: { title: string; description: string; children?: ReactNode }) {
  return <header className="page-heading"><div><h1>{title}</h1><p>{description}</p></div>{children && <div className="heading-actions">{children}</div>}</header>;
}
export function Panel({ title, children, action, className = '' }: { title?: string; children: ReactNode; action?: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{title && <div className="panel-heading"><h2>{title}</h2>{action}</div>}{children}</section>;
}
export function Empty({ children = 'No records yet.' }: { children?: ReactNode }) { return <div className="empty-state"><span className="empty-mark" aria-hidden="true">◇</span><p>{children}</p></div>; }
export function ErrorMessage({ message, retry }: { message: string; retry?: () => void }) { return <div className="notice error" role="alert"><AlertCircle size={18}/><span>{message}</span>{retry && <button className="button small ghost" onClick={retry}><RefreshCw size={14}/>Retry</button>}</div>; }
export function Loading({ label = 'Loading from the lab…' }: { label?: string }) { return <div className="loading-state" role="status"><span className="loading-dot"/>{label}</div>; }
export function ResourceView<T>({ resource, children, empty }: { resource: Resource<T>; children: (data: T) => ReactNode; empty?: string }) {
  return <>{resource.error && <ErrorMessage message={`${resource.data ? 'Refresh failed; last received data is shown. ' : ''}${resource.error}`} retry={resource.refresh}/>}{resource.loading && resource.data === null && <Loading/>}{resource.data !== null && (Array.isArray(resource.data) && resource.data.length === 0 ? <Empty>{empty}</Empty> : children(resource.data))}{!resource.loading && !resource.error && resource.data === null && <Empty>{empty}</Empty>}</>;
}
export function ActionFeedback({ action }: { action: { pending: boolean; error: string | null; success: string | null } }) { return <>{action.pending && <p role="status" className="muted">Waiting for the server…</p>}{action.error && <ErrorMessage message={action.error}/>} {action.success && <p className="success-text" role="status">{action.success}</p>}</>; }
export function Badge({ children, tone }: { children: string; tone?: string }) {
  const good = ['running', 'completed', 'confirmed', 'promoted', 'evaluated', 'connected'];
  const bad = ['failed', 'errored', 'rejected', 'implementation_failed', 'disconnected'];
  return <span className={`badge ${tone ?? (good.includes(children) ? 'mint' : bad.includes(children) ? 'coral' : '')}`}>{humanize(children)}</span>;
}
export function Stat({ label, value, detail, accent }: { label: string; value: ReactNode; detail?: ReactNode; accent?: boolean }) { return <div className={`stat ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>; }
export function Meter({ label, value, max, unit = '', detail }: { label: string; value: number | null; max: number | null; unit?: string; detail?: string }) { return <div className="meter"><div><span>{label}</span><span className="mono">{fmt(value)}{unit}{max !== null ? ` / ${fmt(max)}${unit}` : ''}</span></div>{value !== null && max !== null && max > 0 && <progress aria-label={label} value={value} max={max}/>} {detail && <small>{detail}</small>}</div>; }
export function DataDetails({ title = 'Raw metadata', value, open = false }: { title?: string; value: unknown; open?: boolean }) { return <details className="data-details" open={open}><summary>{title}</summary><pre>{JSON.stringify(value, null, 2) ?? 'No data returned.'}</pre></details>; }
export function Field({ label, children, help }: { label: string; children: ReactNode; help?: string }) { return <label className="field"><span>{label}</span>{children}{help && <small>{help}</small>}</label>; }
export function TextLink({ to, children }: { to: string; children: ReactNode }) { return <Link className="text-link" to={to}>{children}<ArrowUpRight size={15}/></Link>; }
export function Table({ headers, children }: { headers: string[]; children: ReactNode }) { return <div className="table-scroll"><table><thead><tr>{headers.map((h, i) => <th scope="col" key={`${h}-${i}`}>{h}</th>)}</tr></thead><tbody>{children}</tbody></table></div>; }

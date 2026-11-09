// The application never imports this module. These are recorded backend responses, read-only.
const files = import.meta.glob('../../../docs/api-fixtures/*.json', { eager: true, import: 'default' });
export function fixture<T>(name: string): T {
  const data = files[`../../../docs/api-fixtures/${name}.json`];
  if (data === undefined) throw new Error(`Missing fixture: ${name}`);
  return structuredClone(data) as T;
}

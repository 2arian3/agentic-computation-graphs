export function fmtNs(ns?: number | null): string {
  if (ns == null) return "—";
  const s = ns / 1e9;
  if (s < 0.001) return `${(ns / 1e3).toFixed(0)} µs`;
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  return `${s.toFixed(2)} s`;
}

export function fmtS(s?: number | null): string {
  if (s == null) return "—";
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  return `${s.toFixed(2)} s`;
}

export function fmtInt(n?: number | null): string {
  if (n == null) return "—";
  return n.toLocaleString();
}

export function fmtTime(ts?: number | null): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

export function truncate(s: string, n: number): string {
  if (!s) return "";
  return s.length <= n ? s : s.slice(0, n) + "…";
}

export function prettyJSON(v: any): string {
  if (v == null) return "";
  if (typeof v === "string") {
    try {
      return JSON.stringify(JSON.parse(v), null, 2);
    } catch {
      return v;
    }
  }
  return JSON.stringify(v, null, 2);
}

export function fmtBytes(n?: number): string {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

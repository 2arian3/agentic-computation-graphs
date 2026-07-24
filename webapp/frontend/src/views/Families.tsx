import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Card } from "../components/Card";

type FamStats = {
  n: number; accuracy: number; short_circuit_frac: number;
  nodes_mean: number; nodes_std: number; depth_mean: number;
  width_mean: number; width_max: number; width_gt1_frac: number;
  width_exec_gt1_frac: number; tokens_mean: number;
  distinct_shapes: number; modal_shape_frac: number; ext_tools: string[];
};
type Rollup = {
  files: string[]; families: string[];
  by_file: Record<string, Record<string, FamStats>>;
  metric_help: Record<string, string>;
};

const pct = (x: number) => `${Math.round((x ?? 0) * 100)}%`;
const label = (f: string) => f.replace(/^families_?/, "") || "families";

export default function Families() {
  const [data, setData] = useState<Rollup | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = () => { setErr(null); api.families().then(setData).catch((e) => setErr(String(e))); };
  useEffect(() => { load(); }, []);

  if (err) return <div className="empty">Failed to load family rollup: {err}</div>;
  if (!data) return <div className="empty">Loading family rollup…</div>;

  const rows: { fam: string; file: string; s: FamStats; first: boolean }[] = [];
  for (const fam of data.families) {
    let first = true;
    for (const file of data.files) {
      const s = data.by_file[file]?.[fam];
      if (!s) continue;
      rows.push({ fam, file, s, first });
      first = false;
    }
  }

  return (
    <div>
      <div className="spread" style={{ marginBottom: 14 }}>
        <div>
          <h2 style={{ margin: 0 }}>Family rollup — structure by application type × backbone</h2>
          <div className="tiny faint">
            From archived sweep traces ({data.files.map(label).join(", ")}). Mirrors <span className="mono">docs/12</span>.
          </div>
        </div>
        <button className="btn" onClick={load}>⟳ Refresh</button>
      </div>

      <Card title="Per-family × backbone">
        <div style={{ overflow: "auto" }}>
          <table className="tbl">
            <thead>
              <tr>
                <th>family</th><th>backbone</th><th>n</th><th>acc</th>
                <th title={data.metric_help?.short_circuit_frac}>short-circuit</th>
                <th>nodes</th><th>depth</th><th>width m/max</th>
                <th title={data.metric_help?.width_gt1_frac}>width&gt;1</th>
                <th>tokens</th><th>#shapes</th><th>modal</th><th>ext tools used</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} style={r.first && i > 0 ? { borderTop: "2px solid var(--line, #8884)" } : undefined}>
                  <td>{r.first ? <b>{r.fam}</b> : ""}</td>
                  <td><span className="tag">{label(r.file)}</span></td>
                  <td className="mono">{r.s.n}</td>
                  <td className="mono">{r.s.accuracy.toFixed(2)}</td>
                  <td className="mono">
                    <span style={r.s.short_circuit_frac >= 0.3 ? { color: "#e0685a", fontWeight: 700 } : undefined}>
                      {pct(r.s.short_circuit_frac)}
                    </span>
                  </td>
                  <td className="mono">{r.s.nodes_mean.toFixed(1)}±{r.s.nodes_std.toFixed(1)}</td>
                  <td className="mono">{r.s.depth_mean.toFixed(1)}</td>
                  <td className="mono">{r.s.width_mean.toFixed(2)}/{r.s.width_max}</td>
                  <td className="mono">{pct(r.s.width_gt1_frac)}</td>
                  <td className="mono">{Math.round(r.s.tokens_mean)}</td>
                  <td className="mono">{r.s.distinct_shapes}</td>
                  <td className="mono">{r.s.modal_shape_frac.toFixed(2)}</td>
                  <td className="tiny">{r.s.ext_tools.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="tiny faint" style={{ marginTop: 10 }}>
          <b>short-circuit</b> = fraction of runs finishing with ≤1 tool call (a ≤1-node ACG) — the failure
          signature. On tool-composition families, accuracy ≈ 1 − short-circuit. Values ≥ 30% are highlighted.
        </div>
      </Card>
    </div>
  );
}

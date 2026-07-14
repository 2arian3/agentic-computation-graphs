import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Doc } from "../api/types";
import { Card } from "../components/Card";

export default function Documents() {
  const [kind, setKind] = useState<"corpus" | "distractors">("corpus");
  const [docs, setDocs] = useState<Doc[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [form, setForm] = useState<Doc>({ id: "", title: "", text: "" });
  const [mode, setMode] = useState<"view" | "edit" | "new">("view");
  const [toast, setToast] = useState<{ msg: string; err?: boolean } | null>(null);
  const [dirty, setDirty] = useState(false);

  // retrieval preview
  const [query, setQuery] = useState("");
  const [noise, setNoise] = useState(0);
  const [results, setResults] = useState<any[] | null>(null);

  const load = (k = kind) => api.corpus(k).then((r) => setDocs(r.docs));
  useEffect(() => { load(); setSel(null); setMode("view"); /* eslint-disable-next-line */ }, [kind]);

  const flash = (msg: string, err = false) => { setToast({ msg, err }); setTimeout(() => setToast(null), 2600); };

  const openDoc = (d: Doc) => { setSel(d.id); setForm({ ...d }); setMode("view"); setDirty(false); };
  const newDoc = () => { setSel(null); setForm({ id: "", title: "", text: "" }); setMode("new"); setDirty(false); };

  const save = async () => {
    try {
      if (mode === "new") {
        const created = await api.createDoc(form, kind);
        flash(`created ${created.id}`);
      } else {
        await api.updateDoc(sel!, form, kind);
        flash(`saved ${sel}`);
      }
      await load();
      setMode("view");
      setSel(form.id);
      setDirty(false);
    } catch (e: any) {
      flash(String(e.message || e), true);
    }
  };

  const del = async (id: string) => {
    if (!confirm(`Delete document ${id}? A .bak is kept.`)) return;
    try {
      await api.deleteDoc(id, kind);
      await load();
      if (sel === id) { setSel(null); setMode("view"); }
      flash(`deleted ${id}`);
    } catch (e: any) { flash(String(e.message || e), true); }
  };

  const reindex = async () => {
    try { const r = await api.reindex(); flash(`reindexed · ${r.num_docs} docs, ${r.total_indexed_terms} terms`); }
    catch (e: any) { flash(String(e.message || e), true); }
  };

  const preview = async () => {
    if (!query.trim()) return;
    const r = await api.corpusSearch(query, 5, noise);
    setResults(r.results);
  };

  const upd = (patch: Partial<Doc>) => { setForm((f) => ({ ...f, ...patch })); setDirty(true); };

  return (
    <div>
      <div className="spread" style={{ marginBottom: 14 }}>
        <div>
          <h2 style={{ margin: 0 }}>Document store</h2>
          <div className="tiny faint">Edits write to <span className="mono">data/{kind === "corpus" ? "corpus.json" : "distractors.json"}</span> (a .bak is kept). The agent loads this file per run.</div>
        </div>
        <div className="btn-row">
          <div className="seg">
            <button className={kind === "corpus" ? "on" : ""} onClick={() => setKind("corpus")}>corpus</button>
            <button className={kind === "distractors" ? "on" : ""} onClick={() => setKind("distractors")}>distractors</button>
          </div>
          <button className="btn" onClick={reindex}>⟳ Rebuild index</button>
          <button className="btn primary" onClick={newDoc}>+ New document</button>
        </div>
      </div>

      <div className="dash-grid" style={{ gridTemplateColumns: "320px 1fr" }}>
        <Card title={`${docs.length} documents`}>
          <div style={{ maxHeight: 620, overflow: "auto" }}>
            {docs.map((d) => (
              <div key={d.id} className="list-item" style={{ marginBottom: 7, cursor: "pointer", borderColor: sel === d.id ? "var(--accent)" : undefined }}
                onClick={() => openDoc(d)}>
                <div className="spread">
                  <div><span className="mono small">{d.id}</span> · <b className="small">{d.title}</b></div>
                  <button className="btn sm danger ghost" onClick={(e) => { e.stopPropagation(); del(d.id); }}>✕</button>
                </div>
                <div className="tiny faint" style={{ marginTop: 3 }}>{d.text.slice(0, 90)}…</div>
              </div>
            ))}
            {!docs.length && <div className="empty">No documents.</div>}
          </div>
        </Card>

        <div className="col-stack">
          <Card
            title={mode === "new" ? "New document" : sel ? `Document ${sel}` : "Document"}
            right={
              mode === "view" && sel ? (
                <button className="btn sm" onClick={() => setMode("edit")}>Edit</button>
              ) : mode !== "view" ? (
                <div className="btn-row">
                  <button className="btn sm ghost" onClick={() => { if (sel) openDoc(docs.find((d) => d.id === sel)!); else setMode("view"); }}>Cancel</button>
                  <button className="btn sm primary" disabled={!form.id.trim() || !dirty} onClick={save}>Save changes</button>
                </div>
              ) : null
            }
          >
            {mode === "view" && !sel ? (
              <div className="empty">Select a document to read, or create a new one.</div>
            ) : mode === "view" ? (
              <div>
                <div className="kv" style={{ marginBottom: 10 }}>
                  <span className="k">id</span><span className="v">{form.id}</span>
                  <span className="k">title</span><span className="v">{form.title}</span>
                </div>
                <pre className="code" style={{ maxHeight: 460 }}>{form.text}</pre>
              </div>
            ) : (
              <div>
                <div className="row2">
                  <div className="field">
                    <label>id</label>
                    <input type="text" value={form.id} disabled={mode === "edit"}
                      onChange={(e) => upd({ id: e.target.value })} placeholder="D17" />
                  </div>
                  <div className="field">
                    <label>title</label>
                    <input type="text" value={form.title} onChange={(e) => upd({ title: e.target.value })} />
                  </div>
                </div>
                <div className="field">
                  <label>text</label>
                  <textarea style={{ minHeight: 220 }} value={form.text} onChange={(e) => upd({ text: e.target.value })} />
                </div>
              </div>
            )}
          </Card>

          <Card title="Retrieval preview" collapsible right={<span className="tiny faint">runs the real corpus.search()</span>}>
            <div className="row2" style={{ alignItems: "end" }}>
              <div className="field" style={{ margin: 0 }}>
                <label>query</label>
                <input type="text" value={query} onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && preview()} placeholder="Aurora-9 blade alloy" />
              </div>
              <div className="btn-row" style={{ marginBottom: 0 }}>
                <div className="field" style={{ margin: 0, width: 90 }}>
                  <label>noise</label>
                  <input type="number" min={0} max={5} value={noise} onChange={(e) => setNoise(Number(e.target.value))} />
                </div>
                <button className="btn" onClick={preview}>Search</button>
              </div>
            </div>
            {results && (
              <div style={{ marginTop: 12 }}>
                {results.map((r) => (
                  <div key={r.doc_id} className="doc-hit">
                    <div className="spread">
                      <div><span className="mono small">{r.doc_id}</span> · <b className="small">{r.title}</b></div>
                      <span className="score-chip">score {r.score}</span>
                    </div>
                    <div className="tiny faint" style={{ marginTop: 4 }}>{r.snippet}</div>
                  </div>
                ))}
                {!results.length && <div className="empty">No matches.</div>}
              </div>
            )}
          </Card>
        </div>
      </div>

      {toast && <div className={`toast ${toast.err ? "err" : ""}`}>{toast.msg}</div>}
    </div>
  );
}

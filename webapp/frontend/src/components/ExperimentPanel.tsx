import { useEffect, useMemo, useRef, useState } from "react";
import type { Endpoint } from "../App";
import { api, streamSSE } from "../api/client";
import type { HistoryEntry, ModelInfo, PromptPreset, RunConfig } from "../api/types";
import { useRun } from "../state/store";
import { Card } from "./Card";

interface ServingState {
  current: string | null;
  busy: boolean;
  servable: { served: string; label: string; system: string; current: boolean }[];
}

interface Props {
  serverUp: boolean;
  prefill: HistoryEntry | null;
  onPrefillConsumed: () => void;
  endpoint: Endpoint | null;
  setEndpoint: (e: Endpoint) => void;
}

const numField = (v: any) => (v === "" || v == null ? null : Number(v));

export default function ExperimentPanel({ serverUp, prefill, onPrefillConsumed, endpoint, setEndpoint }: Props) {
  const { state, run, abort } = useRun();
  const [cfg, setCfg] = useState<RunConfig | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [presets, setPresets] = useState<PromptPreset[]>([]);
  const [promptMode, setPromptMode] = useState<"preset" | "custom">("preset");
  const [taskId, setTaskId] = useState<string>("");
  const [customPrompt, setCustomPrompt] = useState("");
  const [noise, setNoise] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showEndpoint, setShowEndpoint] = useState(false);
  const [serving, setServing] = useState<ServingState | null>(null);
  const [serveStatus, setServeStatus] = useState<"idle" | "serving" | "done" | "error">("idle");
  const [serveLog, setServeLog] = useState<string[]>([]);
  const serveAbort = useRef<(() => void) | null>(null);

  const loadServing = () => api.serving().then((s) => setServing({ current: s.current, busy: s.busy, servable: s.servable })).catch(() => setServing(null));
  useEffect(() => {
    loadServing();
    const t = setInterval(loadServing, 10000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    api.defaults().then((d) => setCfg(d));
    api.prompts().then((p) => {
      setPresets(p.presets);
      if (p.presets.length) setTaskId(p.presets[0].task_id);
    });
  }, []);

  // Fetch models for the *current* endpoint; auto-select a live served model.
  useEffect(() => {
    api.models(endpoint?.baseUrl).then((m) => {
      setModels(m.models);
      const avail = m.models.filter((x) => x.available).map((x) => x.served);
      setCfg((c) => (c && avail.length && !avail.includes(c.model) ? { ...c, model: avail[0] } : c));
    });
  }, [endpoint?.baseUrl]);

  // Apply a rerun request from History.
  useEffect(() => {
    if (!prefill || !cfg) return;
    setCfg({ ...cfg, ...prefill.config });
    if (prefill.task_id && prefill.task_id !== "CUSTOM") {
      setPromptMode("preset");
      setTaskId(prefill.task_id);
    } else {
      setPromptMode("custom");
      setCustomPrompt(prefill.question || "");
    }
    onPrefillConsumed();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefill, cfg]);

  const set = (patch: Partial<RunConfig>) => setCfg((c) => (c ? { ...c, ...patch } : c));
  const selectedPreset = useMemo(() => presets.find((p) => p.task_id === taskId), [presets, taskId]);
  const running = state.status === "running";

  if (!cfg) return <Card title="Experiment"><div className="empty">Loading configuration…</div></Card>;

  const submit = () => {
    const body: any = {
      ...cfg,
      noise,
      base_url: endpoint?.baseUrl || cfg.base_url,
      gen_ai_system: endpoint?.system || cfg.gen_ai_system,
    };
    if (promptMode === "custom") {
      body.task_id = "CUSTOM";
      body.prompt = customPrompt.trim();
    } else {
      body.task_id = taskId;
    }
    run(body);
  };

  const canRun =
    !running && serveStatus !== "serving" && (promptMode === "preset" ? !!taskId : customPrompt.trim().length > 0);

  // Plain computation (NOT a hook): this runs after the `if (!cfg) return` above, so a
  // useMemo here would violate the rules-of-hooks (React #310).
  const servableSet = new Set((serving?.servable || []).map((s) => s.served));
  const needsServe = !!serving && !!cfg && cfg.model !== serving.current && servableSet.has(cfg.model);
  const serveTarget = (serving?.servable || []).find((s) => s.served === cfg?.model);

  const serveModel = () => {
    if (!cfg || serveStatus === "serving") return;
    if (!confirm(
      `Restart vLLM to serve “${cfg.model}”?\n\nThis stops the current model and reloads the GPU — ` +
      `warmup can take 1–6 minutes (longer for 14B). Any in-flight run finishes first.`
    )) return;
    setServeStatus("serving");
    setServeLog([`requesting swap to ${cfg.model} …`]);
    serveAbort.current = streamSSE(
      "/api/serving/serve",
      { model: cfg.model },
      (ev) => {
        if (ev.kind === "serve_progress") setServeLog((l) => [...l.slice(-40), ev.msg]);
        else if (ev.kind === "serve_done") {
          setServeLog((l) => [...l, ev.already ? "already served." : `✓ now serving ${ev.model}`]);
          setServeStatus("done");
          if (ev.system) setEndpoint({ baseUrl: endpoint?.baseUrl || "", system: ev.system });
          loadServing();
          api.models(endpoint?.baseUrl).then((m) => setModels(m.models));
        } else if (ev.kind === "serve_error") {
          setServeLog((l) => [...l, "✗ " + ev.error]);
          setServeStatus("error");
          loadServing();
        }
      },
      () => {},
      (msg) => { setServeLog((l) => [...l, "✗ " + msg]); setServeStatus("error"); },
    );
  };

  return (
    <Card
      title="Experiment"
      right={
        <span className={`badge ${serverUp ? "correct" : "no_answer"}`}>
          {serverUp ? "server up" : "server down"}
        </span>
      }
    >
      {/* Model */}
      <div className="field">
        <label>Model</label>
        <select value={cfg.model} onChange={(e) => set({ model: e.target.value })}>
          {[...models].sort((a, b) => Number(b.available) - Number(a.available)).map((m) => {
            const servable = servableSet.has(m.served);
            return (
              // Live models are runnable now; cached models are selectable so you can Serve
              // them; only genuinely-unavailable models are disabled.
              <option key={m.served} value={m.served} disabled={serverUp && !m.available && !servable}>
                {m.label}
                {m.available ? " · live" : servable ? " · cached (serve to switch)" : serverUp ? " · not served" : ""}
              </option>
            );
          })}
          {!models.some((m) => m.served === cfg.model) && <option value={cfg.model}>{cfg.model}</option>}
        </select>
        {serverUp && serving && (
          <div style={{ marginTop: 6 }}>
            <div className="hint">
              Currently serving <span className="mono">{serving.current || "—"}</span>. vLLM serves one model at a time.
            </div>
            {needsServe && serveStatus !== "serving" && (
              <button className="btn sm" style={{ marginTop: 6 }} onClick={serveModel}>
                ⚡ Serve “{serveTarget?.label || cfg.model}” on vLLM
              </button>
            )}
            {cfg.model !== serving.current && !servableSet.has(cfg.model) && (
              <div className="tiny faint" style={{ marginTop: 4 }}>
                “{cfg.model}” isn’t in the offline cache — serve it via <span className="mono">docker/serve_vllm.sh</span>.
              </div>
            )}
            {serveStatus === "serving" && (
              <div className="small" style={{ marginTop: 6 }}>
                <span className="spinner" /> swapping model — this can take minutes; keep this tab open.
              </div>
            )}
            {serveLog.length > 0 && serveStatus !== "idle" && (
              <pre className="code" style={{ maxHeight: 150, marginTop: 6 }}>{serveLog.join("\n")}</pre>
            )}
          </div>
        )}
        <div className="spread" style={{ marginTop: 5 }}>
          <span className="hint">
            → <span className={`mono ${serverUp ? "" : "faint"}`}>{endpoint?.baseUrl || cfg.base_url}</span>{" "}
            {serverUp ? <span style={{ color: "var(--ok)" }}>● up</span> : <span style={{ color: "var(--err)" }}>● down</span>}
          </span>
          <span className="tiny mono" style={{ cursor: "pointer", color: "var(--accent)" }} onClick={() => setShowEndpoint((s) => !s)}>
            {showEndpoint ? "hide" : "edit"} endpoint
          </span>
        </div>
      </div>

      {showEndpoint && (
        <div style={{ marginBottom: 12, padding: 10, border: "1px solid var(--border)", borderRadius: 8, background: "var(--bg-elev2)" }}>
          <div className="field" style={{ marginBottom: 8 }}>
            <label>base_url <span className="hint">· OpenAI-compatible /v1 endpoint</span></label>
            <input type="text" value={endpoint?.baseUrl || ""} placeholder="http://localhost:8001/v1"
              onChange={(e) => setEndpoint({ baseUrl: e.target.value, system: endpoint?.system || "" })} />
          </div>
          <div className="field" style={{ margin: 0 }}>
            <label>gen_ai_system <span className="hint">· “qwen” / “llama” (telemetry tag)</span></label>
            <input type="text" value={endpoint?.system || ""} placeholder="llama"
              onChange={(e) => setEndpoint({ baseUrl: endpoint?.baseUrl || "", system: e.target.value })} />
          </div>
          <div className="tiny faint" style={{ marginTop: 8 }}>
            Saved locally · applied to the next run and to the live-status check — no restart needed.
          </div>
        </div>
      )}

      {/* Decode params */}
      <div className="field">
        <label>Temperature <span className="hint">· sampling randomness (variance driver)</span></label>
        <div className="slider-row">
          <input type="range" min={0} max={1.5} step={0.05} value={cfg.temperature}
            onChange={(e) => set({ temperature: Number(e.target.value) })} />
          <span className="val">{cfg.temperature.toFixed(2)}</span>
        </div>
      </div>
      <div className="row2">
        <div className="field">
          <label>top_p</label>
          <input type="number" min={0} max={1} step={0.01} value={cfg.top_p}
            onChange={(e) => set({ top_p: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>max_tokens</label>
          <input type="number" min={16} step={16} value={cfg.max_tokens}
            onChange={(e) => set({ max_tokens: Number(e.target.value) })} />
        </div>
      </div>
      <div className="row2">
        <div className="field">
          <label>seed <span className="hint">· blank = none</span></label>
          <input type="number" value={cfg.seed ?? ""} placeholder="none"
            onChange={(e) => set({ seed: numField(e.target.value) })} />
        </div>
        <div className="field">
          <label>max_steps</label>
          <input type="number" min={1} max={20} value={cfg.max_steps}
            onChange={(e) => set({ max_steps: Number(e.target.value) })} />
        </div>
      </div>

      <div className="spread" style={{ cursor: "pointer" }} onClick={() => setShowAdvanced((s) => !s)}>
        <span className="small dim mono">{showAdvanced ? "▾" : "▸"} advanced</span>
        <span className="tiny faint">retrieval · concurrency · reasoning · branch tool</span>
      </div>
      {showAdvanced && (
        <div style={{ marginTop: 10 }}>
          <div className="row2">
            <div className="field">
              <label>search_top_k</label>
              <input type="number" min={1} max={10} value={cfg.search_top_k}
                onChange={(e) => set({ search_top_k: Number(e.target.value) })} />
            </div>
            <div className="field">
              <label>max_tool_workers</label>
              <input type="number" min={1} max={16} value={cfg.max_tool_workers}
                onChange={(e) => set({ max_tool_workers: Number(e.target.value) })} />
            </div>
          </div>
          <div className="row2">
            <div className="field">
              <label>retrieval noise <span className="hint">· distractors</span></label>
              <input type="number" min={0} max={5} value={noise} onChange={(e) => setNoise(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>sub_agent_max_steps</label>
              <input type="number" min={1} max={12} value={cfg.sub_agent_max_steps}
                onChange={(e) => set({ sub_agent_max_steps: Number(e.target.value) })} />
            </div>
          </div>
          <label className="switch" style={{ marginBottom: 8 }}>
            <input type="checkbox" checked={cfg.elicit_reasoning}
              onChange={(e) => set({ elicit_reasoning: e.target.checked })} />
            Elicit reasoning <span className="faint tiny">(adds a required “thought” to each tool call)</span>
          </label>
          <label className="switch">
            <input type="checkbox" checked={cfg.enable_sub_agent}
              onChange={(e) => set({ enable_sub_agent: e.target.checked })} />
            Enable sub_agent branch tool <span className="faint tiny">(lets the graph fan out)</span>
          </label>
        </div>
      )}

      <div className="divider" />

      {/* Prompt */}
      <div className="field">
        <div className="seg" style={{ marginBottom: 10 }}>
          <button className={promptMode === "preset" ? "on" : ""} onClick={() => setPromptMode("preset")}>Preset prompt</button>
          <button className={promptMode === "custom" ? "on" : ""} onClick={() => setPromptMode("custom")}>Custom prompt</button>
        </div>
        {promptMode === "preset" ? (
          <>
            <select value={taskId} onChange={(e) => setTaskId(e.target.value)}>
              {["multi-hop", "branching"].map((group) => (
                <optgroup key={group} label={group}>
                  {presets.filter((p) => p.group === group).map((p) => (
                    <option key={p.task_id} value={p.task_id}>
                      {p.task_id} · {p.hops}-hop · {p.question.slice(0, 46)}…
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            {selectedPreset && (
              <div className="doc-hit" style={{ marginTop: 10 }}>
                <div className="small">{selectedPreset.question}</div>
                <div className="tiny faint" style={{ marginTop: 6 }}>
                  gold: {selectedPreset.answers.join(", ") || "—"} · supporting: {selectedPreset.supporting.join(", ")}
                </div>
              </div>
            )}
          </>
        ) : (
          <textarea placeholder="Ask a multi-hop question grounded in the document store…"
            value={customPrompt} onChange={(e) => setCustomPrompt(e.target.value)} />
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 6 }}>
        {running ? (
          <button className="btn danger" style={{ flex: 1 }} onClick={abort}>
            ■ Stop
          </button>
        ) : (
          <button className="btn primary" style={{ flex: 1 }} disabled={!canRun} onClick={submit} title={serverUp ? "" : "Model server is down — use Replay, or start vLLM on :8000"}>
            ▶ Run experiment
          </button>
        )}
      </div>
      {!serverUp && !running && (
        <div className="tiny faint" style={{ marginTop: 8 }}>
          Live runs call the model server on :8000 (currently down). You can still explore the full
          visualization via <b>Replay</b> below.
        </div>
      )}
    </Card>
  );
}

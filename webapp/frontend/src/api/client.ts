// REST + SSE client for the dashboard backend.
import type {
  Doc, HistoryEntry, ModelInfo, PromptPreset, RunConfig, StreamEvent, TraceInfo,
} from "./types";

async function jget<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
async function jsend<T>(url: string, method: string, body?: any): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

const q = (baseUrl?: string) => (baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : "");

export const api = {
  health: (baseUrl?: string) =>
    jget<{ server_up: boolean; base_url: string; served_models: string[]; default_model: string }>(`/api/health${q(baseUrl)}`),
  models: (baseUrl?: string) =>
    jget<{ server_up: boolean; default_model: string; models: ModelInfo[] }>(`/api/models${q(baseUrl)}`),
  defaults: () => jget<RunConfig>("/api/defaults"),
  prompts: () => jget<{ presets: PromptPreset[] }>("/api/prompts"),

  traces: () => jget<{ traces: TraceInfo[] }>("/api/traces"),
  traceRuns: (file: string) => jget<{ runs: any[] }>(`/api/traces/runs?file=${encodeURIComponent(file)}`),

  corpus: (kind = "corpus") => jget<{ kind: string; docs: Doc[] }>(`/api/corpus?kind=${kind}`),
  corpusSearch: (query: string, topK: number, noise: number) =>
    jget<{ results: any[] }>(`/api/corpus/search?query=${encodeURIComponent(query)}&top_k=${topK}&noise=${noise}`),
  createDoc: (doc: Doc, kind = "corpus") => jsend<Doc>(`/api/corpus?kind=${kind}`, "POST", doc),
  updateDoc: (id: string, doc: Partial<Doc>, kind = "corpus") => jsend<Doc>(`/api/corpus/${id}?kind=${kind}`, "PUT", doc),
  deleteDoc: (id: string, kind = "corpus") => jsend<{ ok: boolean }>(`/api/corpus/${id}?kind=${kind}`, "DELETE"),
  reindex: () => jsend<any>("/api/corpus/reindex", "POST"),

  history: () => jget<{ entries: HistoryEntry[] }>("/api/history"),
  historyGet: (id: string) => jget<any>(`/api/history/${id}`),
  historyDelete: (id: string) => jsend<{ ok: boolean }>(`/api/history/${id}`, "DELETE"),

  serving: () => jget<{ base_url: string; server_up: boolean; current: string | null; busy: boolean; servable: any[] }>("/api/serving"),
};

// Generic SSE-over-fetch reader (POST + streamed text/event-stream). Returns an abort fn.
export function streamSSE(
  path: string,
  body: any,
  onEvent: (ev: any) => void,
  onDone: () => void,
  onError: (msg: string) => void,
): () => void {
  const controller = new AbortController();
  (async () => {
    try {
      const resp = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) {
        onError(`${resp.status} ${await resp.text()}`);
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        // Normalize CRLF/CR to LF: sse_starlette separates lines and frames with "\r\n",
        // so a naive "\n\n" split would never find a frame boundary.
        buf += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
        // SSE frames are separated by a blank line.
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const dataLines = frame
            .split("\n")
            .filter((l) => l.startsWith("data:"))
            .map((l) => l.slice(5).trim());
          if (!dataLines.length) continue;
          try {
            onEvent(JSON.parse(dataLines.join("\n")));
          } catch {
            /* ignore malformed frame */
          }
        }
      }
      onDone();
    } catch (e: any) {
      if (e.name !== "AbortError") onError(String(e?.message || e));
      else onDone();
    }
  })();
  return () => controller.abort();
}

// Typed convenience for run/replay streams.
export function streamRun(
  path: "/api/runs" | "/api/replay",
  body: any,
  onEvent: (ev: StreamEvent) => void,
  onDone: () => void,
  onError: (msg: string) => void,
): () => void {
  return streamSSE(path, body, onEvent as (ev: any) => void, onDone, onError);
}

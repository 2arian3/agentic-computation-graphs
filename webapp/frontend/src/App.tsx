import { useEffect, useState } from "react";
import { api } from "./api/client";
import { useTheme } from "./state/theme";
import Dashboard from "./views/Dashboard";
import Documents from "./views/Documents";
import History from "./views/History";
import Families from "./views/Families";
import type { HistoryEntry } from "./api/types";

type View = "dashboard" | "documents" | "history" | "families";

export interface Endpoint {
  baseUrl: string;
  system: string;
}

export default function App() {
  const { theme, toggle } = useTheme();
  const [view, setView] = useState<View>("dashboard");
  const [prefill, setPrefill] = useState<HistoryEntry | null>(null);
  const [health, setHealth] = useState<{ server_up: boolean; base_url: string } | null>(null);
  const [endpoint, setEndpointState] = useState<Endpoint | null>(() => {
    try {
      const s = localStorage.getItem("acg-endpoint");
      return s ? JSON.parse(s) : null;
    } catch {
      return null;
    }
  });
  const setEndpoint = (e: Endpoint) => {
    setEndpointState(e);
    localStorage.setItem("acg-endpoint", JSON.stringify(e));
  };

  // Seed the endpoint from the backend's configured default on first load.
  useEffect(() => {
    if (!endpoint) api.defaults().then((d) => setEndpoint({ baseUrl: d.base_url || "", system: d.gen_ai_system || "" })).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const poll = () => api.health(endpoint?.baseUrl).then(setHealth).catch(() => setHealth(null));
    poll();
    const t = setInterval(poll, 8000);
    return () => clearInterval(t);
  }, [endpoint?.baseUrl]);

  const nav: { id: View; icon: string; label: string }[] = [
    { id: "dashboard", icon: "◈", label: "Experiment" },
    { id: "documents", icon: "▤", label: "Documents" },
    { id: "families", icon: "⊞", label: "Families" },
    { id: "history", icon: "◷", label: "History" },
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">ACG</span>
          <span>Experiment Dashboard</span>
        </div>
        <div className="spacer" />
        <span className="status-pill" title={health?.base_url}>
          <span className={`dot ${health == null ? "" : health.server_up ? "up" : "down"}`} />
          {health == null ? "backend…" : health.server_up ? "model server up" : "model server down · replay ready"}
        </span>
        <button className="icon-btn" onClick={toggle} title="Toggle theme">
          {theme === "dark" ? "☀ Light" : "☾ Dark"}
        </button>
      </header>

      <aside className="sidebar">
        {nav.map((n) => (
          <div key={n.id} className={`nav-item ${view === n.id ? "active" : ""}`} onClick={() => setView(n.id)}>
            <span className="ni-ico">{n.icon}</span>
            {n.label}
          </div>
        ))}
        <div className="side-note">
          Reuses the <span className="mono">acg</span> pipeline unchanged. Live runs need the model server on
          <span className="mono"> :8000</span>; otherwise replay archived traces.
        </div>
      </aside>

      <main className="main">
        {view === "dashboard" && (
          <Dashboard
            serverUp={!!health?.server_up}
            prefill={prefill}
            onPrefillConsumed={() => setPrefill(null)}
            endpoint={endpoint}
            setEndpoint={setEndpoint}
          />
        )}
        {view === "documents" && <Documents />}
        {view === "families" && <Families />}
        {view === "history" && (
          <History
            onRerun={(entry) => {
              setPrefill(entry);
              setView("dashboard");
            }}
          />
        )}
      </main>
    </div>
  );
}

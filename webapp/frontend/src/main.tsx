import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ThemeProvider } from "./state/theme";
import { RunProvider } from "./state/store";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <RunProvider>
        <App />
      </RunProvider>
    </ThemeProvider>
  </React.StrictMode>,
);

import React, { useState } from "react";

export function Card({
  title,
  right,
  children,
  collapsible = false,
  defaultCollapsed = false,
  bodyClass,
}: {
  title: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  bodyClass?: string;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  return (
    <div className={`card ${collapsed ? "collapsed" : ""}`}>
      <div className="card-head">
        {collapsible && (
          <span className="chev" onClick={() => setCollapsed((c) => !c)}>
            ▾
          </span>
        )}
        <h3>{title}</h3>
        <div className="ch-right">{right}</div>
      </div>
      <div className={`card-body ${bodyClass || ""}`}>{children}</div>
    </div>
  );
}

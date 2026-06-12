import React, { useState } from "react";
import FoldForm from "./FoldForm.jsx";
import DesignForm from "./DesignForm.jsx";

const TASKS = [
  { id: "predict", t: "Fold & Affinity", s: "Predict 3D structure and binding affinity (Boltz-2, ESMFold-2)." },
  { id: "design", t: "Drug Design", s: "Generate de-novo binders, nanobodies & antibodies (BoltzGen)." },
];

export default function NewJob({ catalog, onSubmitted, onError }) {
  const [task, setTask] = useState("predict");
  if (!catalog) return <div className="empty">Loading…</div>;

  return (
    <div>
      <div className="panel">
        <h2>New prediction</h2>
        <p className="desc">
          Improved open successors to AlphaFold 3, running on Tenstorrent — fully in-region,
          your sequences never leave the cluster.
        </p>
        <div className="cardgrid">
          {TASKS.map((x) => (
            <button
              key={x.id}
              className={`selcard ${task === x.id ? "active" : ""}`}
              onClick={() => setTask(x.id)}
            >
              <div className="t">{x.t}</div>
              <div className="s">{x.s}</div>
            </button>
          ))}
        </div>
        {catalog.demo_note && (
          <div className="demo-banner mt16">
            <span className="demo-badge">Free demo</span>
            <span>{catalog.demo_note}</span>
          </div>
        )}
      </div>

      {task === "predict" ? (
        <FoldForm catalog={catalog} onSubmitted={onSubmitted} onError={onError} />
      ) : (
        <DesignForm catalog={catalog} onSubmitted={onSubmitted} onError={onError} />
      )}
    </div>
  );
}

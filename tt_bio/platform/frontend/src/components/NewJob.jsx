import React, { useState } from "react";
import FoldForm from "./FoldForm.jsx";
import DesignForm from "./DesignForm.jsx";
import EmbedForm from "./EmbedForm.jsx";

const TASKS = [
  { id: "predict", t: "Fold & Affinity", s: "Predict 3D structure and binding affinity." },
  { id: "design", t: "Drug Design", s: "Generate de-novo binders, nanobodies & antibodies with BoltzGen." },
  { id: "embed", t: "Protein Embeddings", s: "Compute ESMC language-model embeddings for search, clustering & ML features." },
];

export default function NewJob({ catalog, onSubmitted, onError }) {
  const [task, setTask] = useState("predict");
  if (!catalog) return <div className="empty">Loading…</div>;

  return (
    <div>
      <div className="panel">
        <h2>New job</h2>
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
      ) : task === "design" ? (
        <DesignForm catalog={catalog} onSubmitted={onSubmitted} onError={onError} />
      ) : (
        <EmbedForm catalog={catalog} onSubmitted={onSubmitted} onError={onError} />
      )}
    </div>
  );
}

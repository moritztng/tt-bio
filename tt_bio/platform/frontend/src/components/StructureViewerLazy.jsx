import React, { Suspense, lazy } from "react";

// Mol* is large (~1 MB gzip), so load it as a separate chunk only when a result
// structure is actually viewed — keeps the initial app load light.
const StructureViewer = lazy(() => import("./StructureViewer.jsx"));

export default function StructureViewerLazy(props) {
  return (
    <Suspense fallback={<div className="viewer-wrap"><div className="viewer-canvas"><div className="viewer-overlay">Loading viewer…</div></div></div>}>
      <StructureViewer {...props} />
    </Suspense>
  );
}

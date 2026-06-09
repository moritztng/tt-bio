import React, { useEffect, useRef, useState } from "react";
import * as $3Dmol from "3dmol";

const CHAIN_PALETTE = ["#147598", "#c98a00", "#2f8f4e", "#7048e8", "#c70007", "#0f5d79", "#d6336c", "#1a73c4"];

export default function StructureViewer({ url, format = "cif", downloadName }) {
  const hostRef = useRef(null);
  const viewerRef = useRef(null);
  const [scheme, setScheme] = useState("plddt");
  const [surface, setSurface] = useState(false);
  const [spin, setSpin] = useState(false);
  const [status, setStatus] = useState("loading");
  const brange = useRef([0, 100]);
  const chains = useRef([]);

  // Create the viewer once.
  useEffect(() => {
    const v = $3Dmol.createViewer(hostRef.current, { backgroundColor: "white" });
    viewerRef.current = v;
    return () => { try { v.clear(); } catch { /* noop */ } };
  }, []);

  // (Re)load the model when the URL changes.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !url) return;
    let cancelled = false;
    setStatus("loading");
    fetch(url)
      .then((r) => r.text())
      .then((text) => {
        if (cancelled) return;
        v.clear();
        v.addModel(text, format);
        const atoms = v.getModel().selectedAtoms({});
        const bs = atoms.map((a) => a.b).filter((b) => typeof b === "number" && !Number.isNaN(b));
        brange.current = bs.length ? [Math.min(...bs), Math.max(...bs)] : [0, 100];
        chains.current = [...new Set(atoms.map((a) => a.chain))].filter(Boolean);
        applyStyle();
        v.zoomTo();
        v.render();
        v.resize();
        setStatus("ok");
      })
      .catch(() => !cancelled && setStatus("error"));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, format]);

  // Restyle on scheme / surface change.
  useEffect(() => { if (status === "ok") applyStyle(); /* eslint-disable-next-line */ }, [scheme, surface]);

  // Spin toggle.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || status !== "ok") return;
    v.spin(spin ? "y" : false);
  }, [spin, status]);

  function applyStyle() {
    const v = viewerRef.current;
    if (!v) return;
    v.setStyle({}, {});
    const [mn, mx] = brange.current;
    if (scheme === "plddt") {
      v.setStyle({}, { cartoon: { colorscheme: { prop: "b", gradient: "roygb", min: mn, max: mx } } });
    } else if (scheme === "spectrum") {
      v.setStyle({}, { cartoon: { color: "spectrum" } });
    } else { // chain
      chains.current.forEach((c, i) =>
        v.setStyle({ chain: c }, { cartoon: { color: CHAIN_PALETTE[i % CHAIN_PALETTE.length] } }));
    }
    // Ligands / hetero atoms as sticks.
    v.addStyle({ hetflag: true }, { stick: { radius: 0.18 }, sphere: { scale: 0.28 } });
    v.removeAllSurfaces();
    if (surface) {
      try { v.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.65, color: "#dfe7ea" }, { hetflag: false }); }
      catch { /* surface can fail on odd inputs */ }
    }
    v.render();
  }

  return (
    <div className="viewer-wrap">
      <div className="viewer-toolbar">
        <select value={scheme} onChange={(e) => setScheme(e.target.value)} className="btn sm" style={{ padding: "5px 9px" }}>
          <option value="plddt">Color: confidence (pLDDT)</option>
          <option value="chain">Color: chain</option>
          <option value="spectrum">Color: spectrum</option>
        </select>
        <button className={`btn sm ${surface ? "primary" : ""}`} onClick={() => setSurface((s) => !s)}>Surface</button>
        <button className={`btn sm ${spin ? "primary" : ""}`} onClick={() => setSpin((s) => !s)}>Spin</button>
        <button className="btn sm" onClick={() => { viewerRef.current?.zoomTo(); viewerRef.current?.render(); }}>Reset view</button>
        <div className="spacer" />
        {scheme === "plddt" && (
          <div className="viewer-legend">
            <span>low</span><span className="legend-bar" /><span>high</span>
          </div>
        )}
        {url && <a className="btn sm" href={url} download={downloadName}>Download</a>}
      </div>
      <div className="viewer-canvas" ref={hostRef}>
        {status === "loading" && <div className="empty">Rendering…</div>}
        {status === "error" && <div className="empty">Could not load structure.</div>}
      </div>
    </div>
  );
}

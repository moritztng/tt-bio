import React, { useEffect, useRef, useState } from "react";
import { createPluginUI } from "molstar/lib/mol-plugin-ui";
import { renderReact18 } from "molstar/lib/mol-plugin-ui/react18";
import { DefaultPluginUISpec } from "molstar/lib/mol-plugin-ui/spec";
import { PluginSpec } from "molstar/lib/mol-plugin/spec";
import { PresetStructureRepresentations } from "molstar/lib/mol-plugin-state/builder/structure/representation-preset";
import { MAQualityAssessment } from "molstar/lib/extensions/model-archive/quality-assessment/behavior";
import "molstar/lib/mol-plugin-ui/skin/light.scss";

// Mol* — the same engine RCSB PDB and the AlphaFold DB use. tt-bio writes real
// per-residue pLDDT into the mmCIF (_ma_qa_metric_local), so the native
// 'plddt-confidence' theme colours predictions exactly like AlphaFold DB.
const THEME = { plddt: "plddt-confidence", chain: "chain-id", spectrum: "sequence-id" };

export default function StructureViewer({ url, format = "cif", downloadName }) {
  const hostRef = useRef(null);
  const pluginRef = useRef(null);
  const structRef = useRef(null);
  const [scheme, setScheme] = useState("plddt");
  const [spin, setSpin] = useState(false);
  const [status, setStatus] = useState("loading");

  // Create the plugin once.
  useEffect(() => {
    let disposed = false;
    (async () => {
      const base = DefaultPluginUISpec();
      const spec = {
        ...base,
        behaviors: [...base.behaviors, PluginSpec.Behavior(MAQualityAssessment)],
        layout: { initial: { isExpanded: false, showControls: false } },
        components: { remoteState: "none" },
      };
      const plugin = await createPluginUI({ target: hostRef.current, spec, render: renderReact18 });
      if (disposed) { plugin.dispose(); return; }
      pluginRef.current = plugin;
      await load();
    })();
    return () => { disposed = true; try { pluginRef.current?.dispose(); } catch { /* noop */ } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { if (pluginRef.current) load(); /* eslint-disable-next-line */ }, [url, format]);
  useEffect(() => { if (pluginRef.current && structRef.current) applyTheme(); /* eslint-disable-next-line */ }, [scheme]);

  useEffect(() => {
    const p = pluginRef.current;
    if (!p?.canvas3d) return;
    p.canvas3d.setProps({
      trackball: { animate: spin ? { name: "spin", params: { speed: 1 } } : { name: "off", params: {} } },
    });
  }, [spin, status]);

  async function load() {
    const plugin = pluginRef.current;
    if (!plugin || !url) return;
    setStatus("loading");
    try {
      await plugin.clear();
      const text = await (await fetch(url)).text();
      const data = await plugin.builders.data.rawData({ data: text, label: downloadName });
      const traj = await plugin.builders.structure.parseTrajectory(data, format === "pdb" ? "pdb" : "mmcif");
      const model = await plugin.builders.structure.createModel(traj);
      const structure = await plugin.builders.structure.createStructure(model);
      structRef.current = structure;
      await applyTheme();
      plugin.managers.camera.reset();
      setStatus("ok");
    } catch (e) {
      console.error("Mol* load failed", e);
      setStatus("error");
    }
  }

  async function applyTheme() {
    const plugin = pluginRef.current;
    const structure = structRef.current;
    if (!plugin || !structure) return;
    const globalName = THEME[scheme];
    await plugin.builders.structure.representation.applyPreset(structure, PresetStructureRepresentations.auto, {
      theme: { globalName, focus: { name: globalName } },
    });
  }

  const resetView = () => { pluginRef.current?.managers.camera.reset(); };

  return (
    <div className="viewer-wrap">
      <div className="viewer-toolbar">
        <select value={scheme} onChange={(e) => setScheme(e.target.value)} className="btn sm" style={{ padding: "5px 9px" }}>
          <option value="plddt">Color: confidence (pLDDT)</option>
          <option value="chain">Color: chain</option>
          <option value="spectrum">Color: spectrum (N→C)</option>
        </select>
        <button className={`btn sm ${spin ? "primary" : ""}`} onClick={() => setSpin((s) => !s)}>Spin</button>
        <button className="btn sm" onClick={resetView}>Reset view</button>
        <div className="spacer" />
        {scheme === "plddt" && (
          <div className="viewer-legend"><span>low</span><span className="legend-bar plddt" /><span>high</span></div>
        )}
        {url && <a className="btn sm" href={url} download={downloadName}>Download</a>}
      </div>
      <div className="viewer-canvas">
        <div ref={hostRef} className="molstar-host" />
        {status === "loading" && <div className="viewer-overlay">Rendering…</div>}
        {status === "error" && <div className="viewer-overlay">Could not load structure.</div>}
      </div>
    </div>
  );
}

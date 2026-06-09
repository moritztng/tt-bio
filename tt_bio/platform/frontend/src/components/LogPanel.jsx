import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

export default function LogPanel({ jobId, live }) {
  const [text, setText] = useState("");
  const preRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      fetch(api.logUrl(jobId)).then((r) => r.text()).then((t) => {
        if (cancelled) return;
        setText(t);
      }).catch(() => {});
    load();
    if (!live) return () => { cancelled = true; };
    const i = setInterval(load, 1500);
    return () => { cancelled = true; clearInterval(i); };
  }, [jobId, live]);

  useEffect(() => {
    if (live && preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight;
  }, [text, live]);

  return <pre className="log" ref={preRef}>{text || "No output yet."}</pre>;
}

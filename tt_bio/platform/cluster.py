"""Cluster orchestration for the ai& Bio platform.

The platform turns one "master" galaxy into the coordinator of a fleet. It owns
two long-lived child processes built from the tt-bio engine's own multi-host
primitives:

* a **controller** — ``tt-bio controller --no-local-workers`` — the persistent
  HTTP scheduler that holds the job queue and tracks every connected worker;
* a **local worker pool** — ``tt-bio worker --connect`` — one worker per local
  device on the master, feeding the same controller.

Other galaxies join by running ``tt-bio worker --connect http://<master>:<port>``
against the controller; their devices simply appear in the pool. Every predict
job the platform submits (``tt-bio predict --controller <url>``) is then fanned
across the whole fleet by the controller's lease scheduler — so 32 users on one
galaxy each grab a free device, and a big batch spreads across galaxies, with no
extra routing logic here.

Design jobs (BoltzGen ``gen run``) are local-multi-device only in the engine, so
they can't ride the controller. Instead they briefly borrow the master's local
devices via :meth:`Cluster.design_slot`, which stops the local worker pool for
the duration (remote galaxies keep serving predict) and restarts it afterwards.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from tt_bio.distributed import ControllerClient

from .jobs import TTBIO


def _public_host(bind_host: str) -> str:
    """Best-effort address other machines can use to reach this host."""
    if bind_host not in ("0.0.0.0", "::", ""):
        return bind_host
    try:
        return socket.gethostname()
    except Exception:
        return "<this-host>"


class Cluster:
    """Owns the master's controller + local worker pool and arbitrates the
    local devices between concurrent predict jobs and exclusive design jobs."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        enabled: bool = True,
        bind_host: str = "0.0.0.0",
        port: int = 8765,
        accelerator: str = "tenstorrent",
        num_devices: int = 0,
        device_ids: str | None = None,
    ):
        self.enabled = enabled
        self.bind_host = bind_host
        self.port = port
        self.accelerator = accelerator
        self.num_devices = num_devices
        self.device_ids = device_ids

        self.controller_url = f"http://127.0.0.1:{port}"
        self.join_url = f"http://{_public_host(bind_host)}:{port}"
        self.client = ControllerClient(self.controller_url, timeout=15.0)

        self._logdir = Path(workspace).expanduser().resolve() / "_cluster"
        self._logdir.mkdir(parents=True, exist_ok=True)

        self._controller_proc: subprocess.Popen | None = None
        self._workers_proc: subprocess.Popen | None = None
        self._lock = threading.RLock()        # guards process start/stop
        self._design_lock = threading.Lock()  # one design job borrows devices at a time
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._started:
            return
        self._started = True
        self._start_controller()
        if self._wait_healthy(timeout=20.0):
            self._start_local_workers()

    def _open_log(self, name: str):
        return open(self._logdir / name, "a")

    def _start_controller(self) -> None:
        cmd = [*TTBIO, "controller", "--listen", f"{self.bind_host}:{self.port}",
               "--no-local-workers", "--accelerator", self.accelerator]
        self._controller_proc = subprocess.Popen(
            cmd, stdout=self._open_log("controller.log"),
            stderr=subprocess.STDOUT, start_new_session=True,
        )

    def _start_local_workers(self) -> None:
        with self._lock:
            if self._workers_proc and self._workers_proc.poll() is None:
                return
            cmd = [*TTBIO, "worker", "--connect", self.controller_url,
                   "--accelerator", self.accelerator]
            if self.num_devices:
                cmd += ["--num_devices", str(self.num_devices)]
            if self.device_ids:
                cmd += ["--device_ids", str(self.device_ids)]
            self._workers_proc = subprocess.Popen(
                cmd, stdout=self._open_log("workers.log"),
                stderr=subprocess.STDOUT, start_new_session=True,
            )

    def _stop_local_workers(self) -> None:
        with self._lock:
            proc = self._workers_proc
            self._workers_proc = None
        if proc and proc.poll() is None:
            # SIGINT → the worker command stops its device workers cleanly and
            # releases the chips; fall back to terminate/kill if it lingers.
            import os
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _wait_healthy(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        url = self.controller_url + "/healthz"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(0.3)
        return False

    def shutdown(self) -> None:
        self._stop_local_workers()
        with self._lock:
            proc = self._controller_proc
            self._controller_proc = None
        if proc and proc.poll() is None:
            import os
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # -- submission helpers ------------------------------------------------
    def controller_alive(self) -> bool:
        return bool(
            self.enabled
            and self._controller_proc is not None
            and self._controller_proc.poll() is None
        )

    def submit_url(self) -> str | None:
        """The URL predict jobs should submit to, or None to run locally."""
        return self.controller_url if self.controller_alive() else None

    @contextmanager
    def design_slot(self):
        """Exclusively borrow the master's local devices for a design job.

        Stops the local predict worker pool so the chips are free for
        ``gen run`` (remote galaxies keep serving predict from the controller),
        then restarts the pool when the design job finishes. Serialised so two
        design jobs never contend for the same devices.
        """
        if not self.controller_alive():
            # No managed pool (cluster disabled) — nothing to free.
            yield
            return
        with self._design_lock:
            self._stop_local_workers()
            # Give the chips a moment to be released before gen run opens them.
            time.sleep(2.0)
            try:
                yield
            finally:
                self._start_local_workers()

    # -- status ------------------------------------------------------------
    def status(self) -> dict:
        """Fleet snapshot for the UI: connected galaxies, devices, run/job
        counts, plus the join command for adding more galaxies."""
        info: dict = {
            "enabled": self.enabled,
            "controller_alive": self.controller_alive(),
            "join_url": self.join_url,
            "join_command": f"tt-bio worker --connect {self.join_url}",
            "local_workers_up": bool(self._workers_proc and self._workers_proc.poll() is None),
        }
        if not self.controller_alive():
            info.update({"hosts": [], "online_workers": 0, "total_workers": 0,
                         "runs": {}, "jobs": {}})
            return info
        try:
            snap = self.client.cluster()
            this_host = _public_host(self.bind_host)
            for h in snap.get("hosts", []):
                h["is_master"] = (h.get("host") == socket.gethostname())
            info.update({
                "hosts": snap.get("hosts", []),
                "online_workers": snap.get("online_workers", 0),
                "total_workers": snap.get("total_workers", 0),
                "runs": snap.get("runs", {}),
                "jobs": snap.get("jobs", {}),
                "master_host": this_host,
            })
        except Exception as e:
            info.update({"hosts": [], "online_workers": 0, "total_workers": 0,
                         "runs": {}, "jobs": {}, "error": str(e)})
        return info

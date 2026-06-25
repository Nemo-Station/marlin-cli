"""Background local engine — start / stop / status + auto-start on find/index.

Ollama-style: the local server (MLX or vLLM) runs detached so the ~40s warmup
is a one-time cost and later commands reuse it. State in ~/.marlin/daemon.json,
logs in ~/.marlin/engine.log. Hosted mode has no daemon (it's a remote URL).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time

from . import engines
from .backend import probe
from .config import CONFIG_DIR
from .logging import get_logger
from .models import Config

DAEMON_FILE = CONFIG_DIR / "daemon.json"
LOG_FILE = CONFIG_DIR / "engine.log"
logger = get_logger("daemon")


def _read() -> dict:
    try:
        return json.loads(DAEMON_FILE.read_text())
    except Exception:
        logger.debug("daemon state file missing or unreadable: {}", DAEMON_FILE)
        return {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pids_on_port(port: int) -> list[int]:
    """Return Marlin engine PIDs holding a TCP port.

    Parameters
    ----------
    port
        TCP port to inspect.
    """
    try:
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        logger.debug("could not inspect port {}", port)
        return []
    ours: list[int] = []
    for tok in out.split():
        try:
            pid = int(tok)
            cmd = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5
            ).stdout
            if any(k in cmd for k in ("sglang", "launch_server", "marlin")):
                ours.append(pid)
        except Exception:
            logger.debug("could not inspect process on port {}: {}", port, tok)
    return ours


def _kill(pid: int) -> None:
    """Terminate a process group, falling back to the process itself."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                logger.debug("process already exited before kill: pid={}", pid)
                return
        if sig is signal.SIGTERM:
            time.sleep(0.5)
        if not _alive(pid):
            return


def status(cfg: Config) -> dict:
    """Return background engine status.

    Parameters
    ----------
    cfg
        Runtime configuration.
    """
    d = _read()
    pid = d.get("pid")
    running = bool(pid and _alive(pid))
    return {
        "engine": d.get("engine") or engines.resolve_engine(cfg),
        "pid": pid if running else None,
        "running": running,
        "reachable": probe(cfg.base_url, cfg.api_key),
        "base_url": cfg.base_url,
        "log": str(LOG_FILE),
    }


def start(cfg: Config, log, port: int = engines.LOCAL_PORT, wait_s: float = 600.0) -> dict:
    """Launch the local engine detached and wait until it answers.

    Parameters
    ----------
    cfg
        Runtime configuration.
    log
        Progress callback.
    port
        TCP port for the local server.
    wait_s
        Maximum startup wait in seconds.

    Returns
    -------
    dict
        Engine status payload.
    """
    engine = engines.resolve_engine(cfg)
    if engine == "hosted":
        raise RuntimeError("hosted mode has no local engine to start")
    if probe(cfg.base_url, cfg.api_key):
        logger.info("engine already reachable at {}", cfg.base_url)
        log("engine already running.")
        return status(cfg)
    if not engines.engine_ready(engine):
        raise RuntimeError(f"{engine} engine not installed — run `marlin engine install`")

    # Reap a stale engine squatting the port (a crashed run that never answered),
    # so we don't fail to bind with "address already in use".
    for p in _pids_on_port(port):
        logger.warning("reaping stale engine process on port {}: pid={}", port, p)
        _kill(p)

    argv, env = engines.serve_command(cfg, engine, port)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(LOG_FILE, "ab")
    log(f"starting {engines.label(engine)} … first run warms up (~40s).")
    proc = subprocess.Popen(argv, env=env, stdout=logf, stderr=logf, start_new_session=True)
    logger.info("engine process started: engine={} pid={} port={}", engine, proc.pid, port)
    DAEMON_FILE.write_text(json.dumps({"pid": proc.pid, "engine": engine, "port": port}))

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.error(
                "engine process exited during startup: pid={} code={}", proc.pid, proc.returncode
            )
            raise RuntimeError(f"engine exited (code {proc.returncode}) — see {LOG_FILE}")
        if probe(cfg.base_url, cfg.api_key):
            logger.info("engine ready: pid={} base_url={}", proc.pid, cfg.base_url)
            log("engine ready.")
            return status(cfg)
        time.sleep(2)
    logger.error("engine startup timed out: pid={} wait_s={}", proc.pid, wait_s)
    raise RuntimeError(f"engine not ready in {wait_s:.0f}s — see {LOG_FILE}")


def stop(log) -> dict:
    """Stop the local engine and reap orphaned engine processes.

    Parameters
    ----------
    log
        Progress callback.

    Returns
    -------
    dict
        Stop result with ``stopped`` and ``pids`` fields.
    """
    killed: list[int] = []
    pid = _read().get("pid")
    if pid and _alive(pid):
        _kill(pid)
        killed.append(pid)
    for p in _pids_on_port(engines.LOCAL_PORT):
        if p not in killed:
            _kill(p)
            killed.append(p)
    DAEMON_FILE.unlink(missing_ok=True)
    if killed:
        logger.info("engine processes stopped: {}", killed)
        log(
            f"stopped engine ({len(killed)} "
            f"process{'es' if len(killed) > 1 else ''}, pid {killed[0]})."
        )
    else:
        logger.info("stop requested but no engine process was running")
        log("no running engine.")
    return {"stopped": bool(killed), "pids": killed}


def ensure_running(cfg: Config, log) -> None:
    """Make the configured local engine reachable.

    Parameters
    ----------
    cfg
        Runtime configuration.
    log
        Progress callback.
    """
    eng = engines.resolve_engine(cfg)
    if eng == "hosted":
        return
    if probe(cfg.base_url, cfg.api_key):
        logger.debug("engine already reachable before ensure_running")
        return  # already serving
    if eng == "mlx" and not engines.engine_ready(eng):
        logger.info("MLX engine missing; building before start")
        from .output import build_spinner

        with build_spinner("building the local engine (one time)") as slog:
            engines.install_mlx(slog)
    if eng == "mlx":
        engines.ensure_weights(
            cfg, log
        )  # pre-download weights (progress bar) before the serve timeout
    start(cfg, log)

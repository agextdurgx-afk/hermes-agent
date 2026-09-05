"""Fail-closed bootstrap for execution-admitted Kanban workers.

This module intentionally uses only the standard library until the marker is
present.  An admitted child blocks on a private stdin pipe, consumes one
durable launch authorization, closes the secret channel, and only then returns
control to normal Hermes startup.
"""

from __future__ import annotations

import atexit
import json
import os
import select
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional


class WorkerLaunchAuthorizationError(RuntimeError):
    """The child could not prove its exact one-use launch authorization."""


_CONSUMED: Optional[dict[str, Any]] = None
_FINISHED = False


_READ_ONLY_KANBAN_ACTIONS = {
    "attachments",
    "context",
    "diagnostics",
    "list",
    "log",
    "ls",
    "runs",
    "show",
    "stats",
    "tail",
}


def _is_read_only_kanban_cli(argv: list[str]) -> bool:
    """Allow inspection-only descendants without allowing agent startup."""
    if "chat" in argv:
        return False
    try:
        index = argv.index("kanban") + 1
    except ValueError:
        return False
    while index < len(argv):
        token = argv[index]
        if token == "--board":
            index += 2
            continue
        if token.startswith("--board="):
            index += 1
            continue
        break
    if index >= len(argv):
        return False
    action = argv[index]
    if action in _READ_ONLY_KANBAN_ACTIONS:
        return True
    if action == "boards" and index + 1 < len(argv):
        return argv[index + 1] in {"list", "ls", "show", "current"}
    if action == "admission" and index + 1 < len(argv):
        return argv[index + 1] == "show"
    return False


def _required_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise WorkerLaunchAuthorizationError(f"missing {name}")
    return value


def _read_startup_secret(timeout_seconds: float = 30.0) -> dict[str, str]:
    try:
        if os.name != "nt":
            ready, _write, _error = select.select(
                [sys.stdin.buffer], [], [], timeout_seconds,
            )
            if not ready:
                raise WorkerLaunchAuthorizationError(
                    "execution launch startup authorization timed out"
                )
        raw = sys.stdin.buffer.readline(4097)
    except WorkerLaunchAuthorizationError:
        raise
    except Exception as exc:
        raise WorkerLaunchAuthorizationError(
            "execution launch startup channel is unreadable"
        ) from exc
    if not raw or len(raw) > 4096 or not raw.endswith(b"\n"):
        raise WorkerLaunchAuthorizationError(
            "execution launch startup authorization is missing or malformed"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerLaunchAuthorizationError(
            "execution launch startup authorization is not valid JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"launch_id", "nonce"}:
        raise WorkerLaunchAuthorizationError(
            "execution launch startup authorization has unexpected fields"
        )
    launch_id = str(payload.get("launch_id") or "").strip()
    nonce = str(payload.get("nonce") or "")
    if not launch_id or not nonce:
        raise WorkerLaunchAuthorizationError(
            "execution launch startup authorization is incomplete"
        )
    return {"launch_id": launch_id, "nonce": nonce}


def _missing_capability_reason(task_id: str, db_path: Path) -> Optional[str]:
    """Inspect the durable board without initializing or mutating it."""
    try:
        uri = db_path.as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            tables = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('execution_admissions','execution_launches')"
                )
            }
            if "execution_admissions" in tables:
                live = conn.execute(
                    "SELECT authorization_id, state FROM execution_admissions "
                    "WHERE live_slot = 1 LIMIT 1"
                ).fetchone()
                if live is not None:
                    return (
                        "live execution admission requires a one-use launch "
                        f"capability ({live['authorization_id']}:{live['state']})"
                    )
            if "execution_launches" in tables:
                launch = conn.execute(
                    "SELECT launch_id, state FROM execution_launches "
                    "WHERE task_id = ? AND state IN "
                    "('reserved','spawning','started','revoking') "
                    "ORDER BY created_at DESC, launch_id DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
                if launch is not None:
                    return (
                        "durable execution launch requires its exact capability "
                        f"({launch['launch_id']}:{launch['state']})"
                    )
        finally:
            conn.close()
    except Exception as exc:
        raise WorkerLaunchAuthorizationError(
            f"cannot verify Kanban launch policy from {db_path}: {exc}"
        ) from exc
    return None


def require_execution_launch() -> Optional[dict[str, Any]]:
    """Consume the startup capability before any agent/plugin/tool startup."""
    global _CONSUMED
    if _CONSUMED is not None:
        return dict(_CONSUMED)
    marker_present = os.environ.get("HERMES_KANBAN_LAUNCH_REQUIRED") == "1"
    if not marker_present:
        task_id = str(os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        if not task_id:
            return None
        if _is_read_only_kanban_cli(sys.argv[1:]):
            return None
        db_raw = str(os.environ.get("HERMES_KANBAN_DB") or "").strip()
        if not db_raw:
            raise WorkerLaunchAuthorizationError(
                "Kanban worker is missing HERMES_KANBAN_DB"
            )
        db_path = Path(db_raw).expanduser().resolve()
        reason = _missing_capability_reason(task_id, db_path)
        if reason is not None:
            raise WorkerLaunchAuthorizationError(reason)
        return None

    task_id = _required_env("HERMES_KANBAN_TASK")
    claim_lock = _required_env("HERMES_KANBAN_CLAIM_LOCK")
    db_path = Path(_required_env("HERMES_KANBAN_DB")).expanduser().resolve()
    try:
        run_id = int(_required_env("HERMES_KANBAN_RUN_ID"))
    except ValueError as exc:
        raise WorkerLaunchAuthorizationError(
            "invalid HERMES_KANBAN_RUN_ID"
        ) from exc
    secret = _read_startup_secret()

    try:
        from hermes_cli import kanban_db as kb

        conn = kb.connect(db_path=db_path)
        try:
            consumed = kb.consume_execution_launch(
                conn,
                launch_id=secret["launch_id"],
                nonce=secret["nonce"],
                task_id=task_id,
                run_id=run_id,
                claim_lock=claim_lock,
                worker_pid=os.getpid(),
            )
        finally:
            conn.close()
    except Exception as exc:
        raise WorkerLaunchAuthorizationError(
            f"execution launch authorization refused: {exc}"
        ) from exc
    finally:
        secret.clear()
        try:
            sys.stdin.close()
        except Exception:
            pass

    _CONSUMED = dict(consumed)
    # Remove the marker and launch identity from descendants. The durable
    # authorization is process-bound; ordinary child tools must not inherit a
    # bootstrap obligation or any reusable capability.
    os.environ.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
    atexit.register(finish_execution_launch)
    return dict(_CONSUMED)


def finish_execution_launch() -> bool:
    """Request external process-exit certification before shutdown."""
    global _FINISHED
    if _FINISHED or _CONSUMED is None:
        return _FINISHED
    db_raw = str(os.environ.get("HERMES_KANBAN_DB") or "").strip()
    if not db_raw:
        return False
    try:
        from hermes_cli import kanban_db as kb

        conn = kb.connect(db_path=Path(db_raw).expanduser().resolve())
        try:
            _FINISHED = kb.finish_execution_launch(
                conn,
                launch_id=str(_CONSUMED["launch_id"]),
                worker_pid=os.getpid(),
            )
        finally:
            conn.close()
    except Exception:
        return False
    return _FINISHED

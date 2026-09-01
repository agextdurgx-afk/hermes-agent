"""Fail-closed lifetime guard for detached Computer Use resources on macOS.

LaunchServices reparents ``CuaDriver.app`` immediately and an isolated Firefox
process intentionally owns a separate process group.  A Kanban worker can be
terminated with ``os._exit`` before Python's atexit handlers run, leaving both
resources alive.  This tiny helper runs outside the worker's process group,
watches the exact owner PID plus its start token, and cleans only the exact
socket or process group it was given when that owner disappears.

This module is internal plumbing, not an agent tool.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


_SOCKET_NAME = re.compile(r"^hc-[0-9a-f]{12}\.sock$")


def process_start_token(pid: int) -> str:
    """Return a stable-enough birth token for one local PID, or ``""``."""
    if pid <= 1:
        return ""
    try:
        proc = subprocess.run(
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def owner_is_same_process(pid: int, expected_start: str) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    if not expected_start:
        return False
    return process_start_token(pid) == expected_start


def validate_socket_path(value: str) -> str:
    path = Path(value).expanduser().resolve(strict=False)
    temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    if path.parent != temp_root or not _SOCKET_NAME.fullmatch(path.name):
        raise ValueError("owner guard socket must be an exact Hermes hc-*.sock endpoint")
    return str(path)


def stop_exact_socket(driver: str, socket_path: str) -> None:
    executable = Path(driver).expanduser().resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("owner guard driver must be an executable file")
    endpoint = validate_socket_path(socket_path)
    try:
        subprocess.run(
            [str(executable), "stop", "--socket", endpoint],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def terminate_exact_process_group(pid: int) -> None:
    if pid <= 1 or pid == os.getpid():
        raise ValueError("owner guard target PID is unsafe")
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return
    # The guarded Firefox launch uses start_new_session=True. Refuse to widen
    # a kill if that invariant is ever changed underneath this helper.
    if pgid != pid:
        raise ValueError("owner guard target is not its own process-group leader")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def guard_until_owner_exits(
    *,
    owner_pid: int,
    owner_start: str,
    target_pid: int | None = None,
    driver: str | None = None,
    socket_path: str | None = None,
    poll_seconds: float = 0.25,
) -> None:
    if owner_pid <= 1 or owner_pid == os.getpid():
        raise ValueError("owner guard requires a distinct positive owner PID")
    if not owner_start:
        raise ValueError("owner guard requires the owner's process birth token")
    process_mode = target_pid is not None
    socket_mode = bool(driver and socket_path)
    if process_mode == socket_mode:
        raise ValueError("owner guard requires exactly one target mode")
    while owner_is_same_process(owner_pid, owner_start):
        time.sleep(max(0.05, poll_seconds))
    if process_mode:
        terminate_exact_process_group(int(target_pid))
    else:
        stop_exact_socket(str(driver), str(socket_path))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-start", default="")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-pid", type=int)
    target.add_argument("--socket")
    parser.add_argument("--driver")
    args = parser.parse_args()
    if args.socket and not args.driver:
        parser.error("--socket requires --driver")
    guard_until_owner_exits(
        owner_pid=args.owner_pid,
        owner_start=args.owner_start,
        target_pid=args.target_pid,
        driver=args.driver,
        socket_path=args.socket,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

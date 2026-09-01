from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use import cua_backend, owner_guard


def test_guard_stops_exact_socket_after_owner_token_disappears():
    socket_path = Path(owner_guard.tempfile.gettempdir()) / "hc-123456789abc.sock"
    with patch.object(
        owner_guard,
        "owner_is_same_process",
        side_effect=[True, False],
    ), patch.object(owner_guard.time, "sleep"), patch.object(
        owner_guard, "stop_exact_socket",
    ) as stop:
        owner_guard.guard_until_owner_exits(
            owner_pid=123,
            owner_start="token",
            driver="/usr/bin/true",
            socket_path=str(socket_path),
        )
    stop.assert_called_once_with("/usr/bin/true", str(socket_path))


def test_guard_terminates_only_process_group_leader():
    with patch.object(owner_guard.os, "getpgid", return_value=999):
        with pytest.raises(ValueError, match="process-group leader"):
            owner_guard.terminate_exact_process_group(777)


def test_guard_refuses_unbound_owner_identity():
    with pytest.raises(ValueError, match="birth token"):
        owner_guard.guard_until_owner_exits(
            owner_pid=123,
            owner_start="",
            target_pid=777,
        )


def test_backend_spawns_detached_guard_with_owner_and_exact_target():
    process = MagicMock()
    process.poll.return_value = None
    with patch.object(cua_backend.sys, "platform", "darwin"), patch.object(
        cua_backend, "cua_driver_child_env", return_value={"PATH": "/usr/bin"},
    ), patch(
        "tools.computer_use.owner_guard.process_start_token", return_value="birth-token",
    ), patch.object(
        cua_backend.subprocess, "Popen", return_value=process,
    ) as popen, patch.object(cua_backend.time, "sleep"):
        result = cua_backend._spawn_owner_guard(target_pid=777)
    assert result is process
    command = popen.call_args.args[0]
    assert command[1].endswith("tools/computer_use/owner_guard.py")
    assert command[command.index("--owner-start") + 1] == "birth-token"
    assert command[command.index("--target-pid") + 1] == "777"
    assert popen.call_args.kwargs["start_new_session"] is True


def test_backend_refuses_guard_that_exits_during_startup():
    process = MagicMock()
    process.poll.return_value = 2
    with patch.object(cua_backend.sys, "platform", "darwin"), patch(
        "tools.computer_use.owner_guard.process_start_token", return_value="birth-token",
    ), patch.object(
        cua_backend.subprocess, "Popen", return_value=process,
    ), patch.object(cua_backend.time, "sleep"):
        with pytest.raises(RuntimeError, match="owner guard exited"):
            cua_backend._spawn_owner_guard(target_pid=777)


def test_backend_refuses_guard_without_owner_birth_token():
    with patch.object(cua_backend.sys, "platform", "darwin"), patch(
        "tools.computer_use.owner_guard.process_start_token", return_value="",
    ):
        with pytest.raises(RuntimeError, match="birth token"):
            cua_backend._spawn_owner_guard(target_pid=777)


def test_backend_stop_closes_direct_isolated_firefox_and_guard():
    backend = cua_backend.CuaDriverBackend(permission_mode="standard")
    process = MagicMock(pid=777)
    process.poll.return_value = None
    guard = MagicMock()
    guard.poll.return_value = None
    backend._isolated_launch_pid = 777
    backend._isolated_launch_process = process
    backend._isolated_launch_guard = guard
    with patch.object(backend._session, "stop"), patch.object(backend._bridge, "stop"):
        backend.stop()
    process.terminate.assert_called_once_with()
    guard.terminate.assert_called_once_with()
    assert backend._isolated_launch_pid is None

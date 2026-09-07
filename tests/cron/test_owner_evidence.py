"""A failed identity lookup must not revoke an active cron owner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("fault", ["unreadable_birth", "denied_birth", "missing_recorded_birth"])
def test_live_owner_can_finish_execution_and_delivery_after_observation_failure(
    tmp_path, monkeypatch, fault
):
    import cron.executions as executions
    import cron.delivery_queue as queue
    import gateway.status as status

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(queue, "DELIVERY_DB", None)
    repo = Path(__file__).resolve().parents[2]
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", """
import json, sys
from cron import executions, delivery_queue as queue
if sys.argv[1] == 'missing_recorded_birth':
    executions._process_start_time = lambda pid: None
    queue._process_start_time = lambda pid: None
row = executions.create_execution('owner-evidence', source='builtin')
assert executions.mark_execution_running(row['id'])
queue.enqueue(row['id'], {'id': 'owner-evidence'}, 'fixture, no external send')
assert queue.claim_next()
print(json.dumps(row), flush=True)
assert sys.stdin.readline().strip() == 'finish'
assert executions.finish_execution(row['id'], success=True)
assert queue._finish(row['id'], error=None)
""", fault],
        cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        row = json.loads(child.stdout.readline())
        assert row["pid"] == child.pid and child.poll() is None
        original = executions.get_execution(row["id"])
        delivery = queue.get_status(row["id"])
        real_birth = status.get_process_start_time

        def observed_birth(pid):
            if pid == child.pid and fault == "denied_birth":
                raise PermissionError("injected process-observation denial")
            if pid == child.pid and fault == "unreadable_birth":
                return None
            return real_birth(pid)

        monkeypatch.setattr(status, "get_process_start_time", observed_birth)
        assert executions.recover_interrupted_executions() == 0
        assert queue.recover_abandoned() == 0
        assert executions.get_execution(row["id"]) == original
        assert queue.get_status(row["id"]) == delivery
        assert queue.claim_next() is None
        out, err = child.communicate("finish\n", timeout=15)
        assert child.returncode == 0, (out, err)
        assert executions.get_execution(row["id"])["status"] == "completed"
        assert queue.get_status(row["id"])["status"] == "delivered"
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=15)


@pytest.mark.parametrize("recorded,observed", [(101, 101), (None, 101), (101, None), (0, 101), (101, 0)])
def test_unproven_death_retains_ownership(monkeypatch, recorded, observed):
    from cron import executions
    import gateway.status as status
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(executions, "_process_start_time", lambda pid: observed)
    assert executions._owner_is_live(12345, recorded)


def test_proven_pid_reuse_or_absence_allows_recovery(monkeypatch):
    from cron import executions
    import gateway.status as status
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(executions, "_process_start_time", lambda pid: 202)
    assert not executions._owner_is_live(12345, 101)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
    assert not executions._owner_is_live(12345, None)

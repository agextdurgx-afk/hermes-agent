"""Fail-closed board execution-admission kernel tests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban as kanban_cli


AUTH = "replay:test:0001"
GENERATION = 4
EVIDENCE = "a" * 64


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _begin(conn):
    return kb.begin_execution_admission(
        conn,
        authorization_id=AUTH,
        generation=GENERATION,
        evidence_sha256=EVIDENCE,
    )


def _task(conn, title: str, *, no_agent: bool = False) -> str:
    body = (
        "Execution: deterministic_no_agent_v1"
        if no_agent
        else f"Run one governed worker: {title}"
    )
    workspace = Path(conn.execute("PRAGMA database_list").fetchone()[2]).parent / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee="governed-worker",
        created_by="coordinator",
        workspace_kind="dir",
        workspace_path=str(workspace),
        idempotency_key=f"admission:{title}",
        max_runtime_seconds=900,
        max_retries=1,
        max_attempts=1,
        model_override="model-1",
        provider_override="provider-1",
        initial_status="scheduled",
    )


def _bind(conn, bindings):
    return kb.bind_execution_admission(
        conn,
        authorization_id=AUTH,
        generation=GENERATION,
        evidence_sha256=EVIDENCE,
        task_bindings=bindings,
    )


def _activate(conn, policy_sha256):
    return kb.activate_execution_admission(
        conn,
        authorization_id=AUTH,
        generation=GENERATION,
        policy_sha256=policy_sha256,
    )


def _activate_one_worker(conn, title="collector", *, lane="ready"):
    _begin(conn)
    worker = _task(conn, title)
    bound = _bind(conn, [{
        "task_id": worker,
        "execution_kind": "worker",
        "allowed_claim_status": lane,
    }])
    _activate(conn, bound["policy_sha256"])
    if lane == "ready":
        assert kb.unblock_task(conn, worker)
    else:
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (worker,))
        conn.commit()
    return worker, bound


def test_existing_launch_ledger_adds_exit_request_column(tmp_path):
    db_path = tmp_path / "legacy-launch.db"
    legacy_schema = kb.SCHEMA_SQL.replace(
        "    exit_requested_at  INTEGER,\n",
        "",
    )
    assert legacy_schema != kb.SCHEMA_SQL
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(legacy_schema)
    finally:
        conn.close()
    kb.init_db(db_path=db_path)
    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(execution_launches)")
        }
    finally:
        conn.close()
    assert "exit_requested_at" in columns


def test_begin_is_deny_all_and_exact_replay_is_idempotent(kanban_home):
    with kb.connect() as conn:
        first = _begin(conn)
        second = _begin(conn)
        assert first == second
        assert first["state"] == "prepared"
        assert first["policy_sha256"] is None

        worker = _task(conn, "collector")
        assert kb.unblock_task(conn, worker)
        assert kb.claim_task(conn, worker, claimer="test:prepared") is None
        assert kb.get_task(conn, worker).status == "ready"
        assert kb.has_spawnable_ready(conn) is False

        rejected = [
            event for event in kb.list_events(conn, worker)
            if event.kind == "execution_admission_rejected"
        ]
        assert len(rejected) == 1
        assert rejected[0].payload["reason"] == "execution_admission_prepared_deny_all"
        assert rejected[0].payload["authorization_id"] == AUTH

        # Repeated dispatcher attempts do not create an unbounded event stream.
        assert kb.claim_task(conn, worker, claimer="test:prepared-again") is None
        rejected_again = [
            event for event in kb.list_events(conn, worker)
            if event.kind == "execution_admission_rejected"
        ]
        assert len(rejected_again) == 1


def test_begin_refuses_a_board_with_executable_work(kanban_home):
    with kb.connect() as conn:
        ready = kb.create_task(conn, title="already ready", assignee="worker")
        with pytest.raises(kb.ExecutionAdmissionError, match="executable work"):
            _begin(conn)
        assert kb.execution_admission_status(conn) is None
        assert kb.get_task(conn, ready).status == "ready"


def test_different_live_authorization_cannot_replace_construction_barrier(
    kanban_home,
):
    with kb.connect() as conn:
        _begin(conn)
        with pytest.raises(kb.ExecutionAdmissionError, match="already holds"):
            kb.begin_execution_admission(
                conn,
                authorization_id="replay:test:0002",
                generation=GENERATION,
                evidence_sha256="b" * 64,
            )
        assert kb.execution_admission_status(conn)["authorization_id"] == AUTH


def test_concurrent_preparers_leave_exactly_one_live_barrier(kanban_home):
    barrier = threading.Barrier(2)

    def begin(authorization_id, evidence):
        with kb.connect() as conn:
            barrier.wait(timeout=5)
            try:
                status = kb.begin_execution_admission(
                    conn,
                    authorization_id=authorization_id,
                    generation=GENERATION,
                    evidence_sha256=evidence,
                )
                return ("ok", status["authorization_id"])
            except kb.ExecutionAdmissionError as exc:
                return ("error", str(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda args: begin(*args),
            [(AUTH, EVIDENCE), ("replay:test:0002", "b" * 64)],
        ))
    assert sorted(result[0] for result in results) == ["error", "ok"]
    with kb.connect() as conn:
        live = kb.execution_admission_status(conn)
        assert live is not None
        assert live["authorization_id"] in {AUTH, "replay:test:0002"}


def test_active_admission_allows_only_exact_worker_identity(kanban_home):
    with kb.connect() as conn:
        _begin(conn)
        worker = _task(conn, "collector")
        deterministic = _task(conn, "registration", no_agent=True)
        bound = _bind(conn, [
            {
                "task_id": worker,
                "execution_kind": "worker",
                "allowed_claim_status": "ready",
            },
            {
                "task_id": deterministic,
                "execution_kind": "deterministic_no_agent",
            },
        ])
        assert bound["state"] == "prepared"
        assert len(bound["tasks"]) == 2
        assert len(bound["policy_sha256"]) == 64

        active = _activate(conn, bound["policy_sha256"])
        assert active["state"] == "active"

        # A foreign card created after activation can become ready, but the
        # claim kernel—not merely the dispatcher—refuses it.
        foreign = _task(conn, "foreign")
        assert kb.unblock_task(conn, foreign)
        assert kb.claim_task(conn, foreign, claimer="test:foreign") is None
        assert kb.get_task(conn, foreign).status == "ready"

        # No-agent cards stay non-inference even if another surface unparks one.
        assert kb.unblock_task(conn, deterministic)
        assert kb.claim_task(conn, deterministic, claimer="test:no-agent") is None
        assert kb.get_task(conn, deterministic).status == "ready"

        assert kb.unblock_task(conn, worker)
        claimed = kb.claim_task(conn, worker, claimer="test:allowed")
        assert claimed is not None
        claimed_event = [
            event for event in kb.list_events(conn, worker)
            if event.kind == "claimed"
        ][-1]
        assert claimed_event.payload["execution_admission"] == {
            "authorization_id": AUTH,
            "generation": GENERATION,
            "policy_sha256": bound["policy_sha256"],
            "state": "active",
        }


def test_board_global_dispatch_spawns_only_the_authorized_worker(
    kanban_home, monkeypatch,
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    with kb.connect() as conn:
        _begin(conn)
        worker = _task(conn, "collector")
        deterministic = _task(conn, "gate", no_agent=True)
        foreign = _task(conn, "foreign")
        bound = _bind(conn, [
            {
                "task_id": worker,
                "execution_kind": "worker",
                "allowed_claim_status": "ready",
            },
            {
                "task_id": deterministic,
                "execution_kind": "deterministic_no_agent",
            },
        ])
        _activate(conn, bound["policy_sha256"])
        assert kb.unblock_task(conn, worker)
        assert kb.unblock_task(conn, deterministic)
        assert kb.unblock_task(conn, foreign)

        dry = kb.dispatch_once(
            conn,
            dry_run=True,
            spawn_fn=lambda *_args, **_kwargs: 123,
        )
        assert [task_id for task_id, _who, _workspace in dry.spawned] == [worker]
        assert dict(dry.admission_guarded) == {
            deterministic: "execution_admission_no_agent_task",
            foreign: "execution_admission_task_not_listed",
        }

        spawned: list[str] = []

        class _Writer:
            def write(self, _payload):
                return None
            def flush(self):
                return None
            def close(self):
                return None

        def spawn(task, _workspace, *, execution_launch=None, **_kwargs):
            assert execution_launch is not None
            spawned.append(task.id)
            return kb.SpawnedWorker(pid=__import__("os").getpid(), startup_writer=_Writer())

        live = kb.dispatch_once(conn, spawn_fn=spawn)
        assert spawned == [worker]
        assert [task_id for task_id, _who, _workspace in live.spawned] == [worker]
        assert kb.get_task(conn, worker).status == "running"
        assert kb.get_task(conn, deterministic).status == "ready"
        assert kb.get_task(conn, foreign).status == "ready"
        assert kb.has_spawnable_ready(conn) is False


def test_identity_or_graph_link_change_is_denied(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="completed parent", assignee="operator")
        assert kb.complete_task(conn, parent, result="done before admission")
        _begin(conn)
        worker = _task(conn, "collector")
        bound = _bind(conn, [
            {
                "task_id": worker,
                "execution_kind": "worker",
                "allowed_claim_status": "ready",
            },
        ])
        _activate(conn, bound["policy_sha256"])

        # A post-bind graph mutation changes the frozen task identity.
        kb.link_tasks(conn, parent, worker)
        assert kb.unblock_task(conn, worker)
        assert kb.claim_task(conn, worker, claimer="test:tampered") is None
        rejection = [
            event for event in kb.list_events(conn, worker)
            if event.kind == "execution_admission_rejected"
        ][-1]
        assert rejection.payload["reason"] == "execution_admission_identity_mismatch"


def test_corrupt_policy_bytes_fail_closed_at_claim(kanban_home):
    with kb.connect() as conn:
        _begin(conn)
        worker = _task(conn, "collector")
        bound = _bind(conn, [{
            "task_id": worker,
            "execution_kind": "worker",
            "allowed_claim_status": "ready",
        }])
        _activate(conn, bound["policy_sha256"])
        conn.execute(
            "UPDATE execution_admissions SET policy_json = ? "
            "WHERE authorization_id = ?",
            ('{"changed":true}', AUTH),
        )
        conn.commit()
        assert kb.unblock_task(conn, worker)
        assert kb.claim_task(conn, worker, claimer="test:corrupt") is None
        rejection = [
            event for event in kb.list_events(conn, worker)
            if event.kind == "execution_admission_rejected"
        ][-1]
        assert rejection.payload["reason"] == "execution_admission_policy_hash_mismatch"


def test_review_claim_is_separately_admitted_and_cannot_change_lane(kanban_home):
    with kb.connect() as conn:
        _begin(conn)
        worker = _task(conn, "worker")
        bound = _bind(conn, [{
            "task_id": worker,
            "execution_kind": "worker",
            "allowed_claim_status": "ready",
        }])
        _activate(conn, bound["policy_sha256"])
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (worker,))
        conn.commit()
        assert kb.claim_review_task(conn, worker, claimer="test:review") is None
        assert kb.get_task(conn, worker).status == "review"
        rejection = [
            event for event in kb.list_events(conn, worker)
            if event.kind == "execution_admission_rejected"
        ][-1]
        assert rejection.payload["reason"] == "execution_admission_claim_lane_mismatch"


def test_bind_rejects_prior_attempts_and_requires_absolute_one_shot(kanban_home):
    with kb.connect() as conn:
        attempted = _task(conn, "attempted")
        assert kb.unblock_task(conn, attempted)
        assert kb.claim_task(conn, attempted, claimer="test:old") is not None
        assert kb.block_task(conn, attempted, reason="old failure")

        _begin(conn)
        with pytest.raises(kb.ExecutionAdmissionError, match="worker attempt"):
            _bind(conn, [{
                "task_id": attempted,
                "execution_kind": "worker",
                "allowed_claim_status": "ready",
            }])

        fresh = kb.create_task(
            conn,
            title="not one-shot",
            assignee="governed-worker",
            initial_status="scheduled",
        )
        with pytest.raises(kb.ExecutionAdmissionError, match="max_attempts=1"):
            _bind(conn, [{
                "task_id": fresh,
                "execution_kind": "worker",
                "allowed_claim_status": "ready",
            }])


def test_seal_denies_all_and_close_preserves_tombstone(kanban_home):
    with kb.connect() as conn:
        _begin(conn)
        worker = _task(conn, "collector")
        bound = _bind(conn, [{
            "task_id": worker,
            "execution_kind": "worker",
            "allowed_claim_status": "ready",
        }])
        _activate(conn, bound["policy_sha256"])

        sealed = kb.seal_execution_admission(
            conn,
            authorization_id=AUTH,
            generation=GENERATION,
            policy_sha256=bound["policy_sha256"],
            reason="replay terminal evidence persisted",
        )
        assert sealed["state"] == "sealed"
        assert sealed["live"] is True
        assert kb.unblock_task(conn, worker)
        assert kb.claim_task(conn, worker, claimer="test:sealed") is None

        closed = kb.close_execution_admission(
            conn,
            authorization_id=AUTH,
            generation=GENERATION,
            policy_sha256=bound["policy_sha256"],
        )
        assert closed["state"] == "closed"
        assert closed["live"] is False
        assert kb.execution_admission_status(conn) is None
        assert kb.execution_admission_status(conn, AUTH)["terminal_reason"] == (
            "replay terminal evidence persisted"
        )

        # Ordinary board behavior returns only after the exact sealed CAS closes.
        assert kb.claim_task(conn, worker, claimer="test:closed") is not None


def test_execution_admission_is_scoped_to_one_board_database(tmp_path):
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    with kb.connect(first_path) as first:
        kb.begin_execution_admission(
            first,
            authorization_id=AUTH,
            generation=GENERATION,
            evidence_sha256=EVIDENCE,
        )
    with kb.connect(second_path) as second:
        task = kb.create_task(second, title="independent", assignee="worker")
        assert kb.claim_task(second, task, claimer="test:other-board") is not None


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_reclaim_during_workspace_resolution_prevents_ready_and_review_spawn(
    kanban_home, monkeypatch, lane,
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, f"race-{lane}", lane=lane)
        real_workspace = kb.resolve_workspace
        spawned = []

        def reclaim_while_resolving(task, *, board=None):
            with kb.connect() as other:
                assert kb.reclaim_task(other, task.id, reason="race probe")
            return real_workspace(task, board=board)

        def spawn(*_args, **_kwargs):
            spawned.append(True)
            raise AssertionError("revoked launch reached subprocess creation")

        monkeypatch.setattr(kb, "resolve_workspace", reclaim_while_resolving)
        result = kb.dispatch_once(conn, spawn_fn=spawn)
        assert result.spawned == []
        assert spawned == []
        launch = conn.execute(
            "SELECT state FROM execution_launches WHERE task_id = ?", (worker,)
        ).fetchone()
        assert launch["state"] == "revoked"


def test_claim_reserves_bound_one_use_launch_without_plaintext_secret(kanban_home):
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn)
        claimed = kb.claim_task(conn, worker, claimer=f"{kb._claimer_id().split(':', 1)[0]}:test")
        assert claimed is not None
        assert claimed.execution_launch_id
        assert claimed.execution_launch_nonce
        row = conn.execute(
            "SELECT * FROM execution_launches WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert row["state"] == "reserved"
        assert row["authorization_id"] == AUTH
        assert row["generation"] == GENERATION
        assert row["policy_sha256"] == bound["policy_sha256"]
        assert row["task_id"] == worker
        assert row["run_id"] == claimed.current_run_id
        assert row["claim_lock"] == claimed.claim_lock
        assert row["board_db_path"] == str(kb._connection_db_path(conn))
        assert claimed.execution_launch_nonce not in json.dumps(dict(row))


def test_failed_process_termination_keeps_launch_and_claim_held(kanban_home):
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn)
        local_lock = f"{kb._claimer_id().split(':', 1)[0]}:termination-test"
        claimed = kb.claim_task(conn, worker, claimer=local_lock)
        assert claimed is not None
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
        )
        try:
            with kb._execution_launch_fence(conn):
                kb._mark_execution_launch_spawning(
                    conn, claimed, str(Path(claimed.workspace_path)),
                )
                kb._set_worker_pid(
                    conn,
                    worker,
                    proc.pid,
                    expected_run_id=claimed.current_run_id,
                    expected_claim_lock=claimed.claim_lock,
                    execution_launch_id=claimed.execution_launch_id,
                )

            def denied(_pid, _sig):
                raise PermissionError("denied")

            assert kb.reclaim_task(
                conn, worker, reason="termination probe", signal_fn=denied,
            ) is False
            assert kb.get_task(conn, worker).status == "running"
            launch = conn.execute(
                "SELECT state FROM execution_launches WHERE task_id = ?", (worker,)
            ).fetchone()
            assert launch["state"] == "revoking"
            with pytest.raises(kb.ExecutionAdmissionError, match="cannot seal"):
                kb.seal_execution_admission(
                    conn,
                    authorization_id=AUTH,
                    generation=GENERATION,
                    policy_sha256=bound["policy_sha256"],
                    reason="must retain uncertain process ownership",
                )
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            assert kb.reclaim_task(conn, worker, reason="test cleanup")


def test_pid_bound_nonce_is_single_use_and_rejects_wrong_token(kanban_home):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn)
        local_lock = f"{kb._claimer_id().split(':', 1)[0]}:consume-test"
        claimed = kb.claim_task(conn, worker, claimer=local_lock)
        assert claimed is not None
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                os.getpid(),
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        with pytest.raises(kb.ExecutionAdmissionError, match="startup proof"):
            kb.consume_execution_launch(
                conn,
                launch_id=claimed.execution_launch_id,
                nonce="wrong",
                task_id=worker,
                run_id=claimed.current_run_id,
                claim_lock=claimed.claim_lock,
                worker_pid=os.getpid(),
            )
        started = kb.consume_execution_launch(
            conn,
            launch_id=claimed.execution_launch_id,
            nonce=claimed.execution_launch_nonce,
            task_id=worker,
            run_id=claimed.current_run_id,
            claim_lock=claimed.claim_lock,
            worker_pid=os.getpid(),
        )
        assert started["launch_id"] == claimed.execution_launch_id
        with pytest.raises(kb.ExecutionAdmissionError, match="expected"):
            kb.consume_execution_launch(
                conn,
                launch_id=claimed.execution_launch_id,
                nonce=claimed.execution_launch_nonce,
                task_id=worker,
                run_id=claimed.current_run_id,
                claim_lock=claimed.claim_lock,
                worker_pid=os.getpid(),
            )
        assert kb.finish_execution_launch(
            conn, launch_id=claimed.execution_launch_id, worker_pid=os.getpid(),
        )
        launch = conn.execute(
            "SELECT state, exit_requested_at, exited_at FROM execution_launches "
            "WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert launch["state"] == "started"
        assert launch["exit_requested_at"] is not None
        assert launch["exited_at"] is None
        assert kb.reconcile_execution_launch_exits(conn) == 0


def test_boolean_liveness_failure_cannot_certify_process_death(
    kanban_home, monkeypatch,
):
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn, "tri-state-observer")
        claimed = kb.claim_task(conn, worker, claimer="tri-state:observer")
        assert claimed is not None
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                os.getpid(),
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        kb.consume_execution_launch(
            conn,
            launch_id=claimed.execution_launch_id,
            nonce=claimed.execution_launch_nonce,
            task_id=worker,
            run_id=claimed.current_run_id,
            claim_lock=claimed.claim_lock,
            worker_pid=os.getpid(),
        )
        assert kb.finish_execution_launch(
            conn,
            launch_id=claimed.execution_launch_id,
            worker_pid=os.getpid(),
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        assert kb.reconcile_execution_launch_exits(conn) == 0
        assert conn.execute(
            "SELECT state FROM execution_launches WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()["state"] == "started"
        with pytest.raises(kb.ExecutionAdmissionError, match="cannot seal"):
            kb.seal_execution_admission(
                conn,
                authorization_id=AUTH,
                generation=GENERATION,
                policy_sha256=bound["policy_sha256"],
                reason="boolean liveness failure is unknown",
            )


def test_crash_detector_never_falls_back_to_legacy_for_revoking_launch(
    kanban_home, monkeypatch,
):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "crash-fallback-fence")
        claimed = kb.claim_task(
            conn,
            worker,
            claimer=f"{kb._claimer_id().split(':', 1)[0]}:crash-fallback",
        )
        assert claimed is not None
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                os.getpid(),
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        kb.consume_execution_launch(
            conn,
            launch_id=claimed.execution_launch_id,
            nonce=claimed.execution_launch_nonce,
            task_id=worker,
            run_id=claimed.current_run_id,
            claim_lock=claimed.claim_lock,
            worker_pid=os.getpid(),
        )

        def denied(_pid, _signal):
            raise PermissionError("test denies termination")

        assert kb.reclaim_task(
            conn,
            worker,
            reason="leave exact launch revoking",
            signal_fn=denied,
        ) is False
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
        monkeypatch.setattr(
            kb,
            "_observe_execution_process_identity",
            lambda pid, expected: {
                "state": "unknown",
                "pid": pid,
                "expected_start_time": expected,
                "observed_start_time": None,
                "reason": "adversarial status probe failure",
            },
        )

        assert kb.detect_crashed_workers(conn) == []
        task = kb.get_task(conn, worker)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == claimed.current_run_id
        assert task.claim_lock == claimed.claim_lock
        launch = conn.execute(
            "SELECT state FROM execution_launches WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert launch["state"] == "revoking"
        run = conn.execute(
            "SELECT ended_at, outcome FROM task_runs WHERE id = ?",
            (claimed.current_run_id,),
        ).fetchone()
        assert run["ended_at"] is None
        assert run["outcome"] is None


def test_process_observer_uses_registration_birth_marker_representation():
    expected = kb._worker_process_start_time(os.getpid())
    assert expected is not None
    observation = kb._observe_execution_process_identity(
        os.getpid(), expected,
    )
    assert observation["state"] == "alive"
    assert observation["observed_start_time"] == expected


@pytest.mark.parametrize("wrong_field", ["task", "run", "claim"])
def test_startup_authorization_rejects_wrong_claim_binding(
    kanban_home, wrong_field,
):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, f"wrong-{wrong_field}")
        local_lock = f"{kb._claimer_id().split(':', 1)[0]}:wrong-{wrong_field}"
        claimed = kb.claim_task(conn, worker, claimer=local_lock)
        assert claimed is not None
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                os.getpid(),
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        values = {
            "task_id": worker,
            "run_id": claimed.current_run_id,
            "claim_lock": claimed.claim_lock,
        }
        if wrong_field == "task":
            values["task_id"] = "t_wrong"
        elif wrong_field == "run":
            values["run_id"] += 1
        else:
            values["claim_lock"] += "-wrong"
        with pytest.raises(kb.ExecutionAdmissionError, match="startup proof"):
            kb.consume_execution_launch(
                conn,
                launch_id=claimed.execution_launch_id,
                nonce=claimed.execution_launch_nonce,
                worker_pid=os.getpid(),
                **values,
            )


def test_only_one_concurrent_startup_consumer_can_win(kanban_home):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "concurrent-consumer")
        local_lock = f"{kb._claimer_id().split(':', 1)[0]}:concurrent"
        claimed = kb.claim_task(conn, worker, claimer=local_lock)
        assert claimed is not None
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                os.getpid(),
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )

    barrier = threading.Barrier(2)

    def consume():
        with kb.connect() as other:
            barrier.wait(timeout=5)
            try:
                kb.consume_execution_launch(
                    other,
                    launch_id=claimed.execution_launch_id,
                    nonce=claimed.execution_launch_nonce,
                    task_id=worker,
                    run_id=claimed.current_run_id,
                    claim_lock=claimed.claim_lock,
                    worker_pid=os.getpid(),
                )
                return "started"
            except kb.ExecutionAdmissionError:
                return "refused"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: consume(), range(2)))
    assert sorted(outcomes) == ["refused", "started"]

    with kb.connect() as conn:
        assert kb.finish_execution_launch(
            conn, launch_id=claimed.execution_launch_id, worker_pid=os.getpid(),
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("body", "changed after claim"),
        ("model_override", "different-model"),
        ("workspace_path", "/tmp/different-workspace"),
    ],
)
def test_launch_revalidates_identity_after_claim(
    kanban_home, column, value,
):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, f"drift-{column}")
        claimed = kb.claim_task(conn, worker, claimer="identity:drift")
        assert claimed is not None
        conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, worker))
        conn.commit()
        with kb._execution_launch_fence(conn):
            with pytest.raises(kb.ExecutionAdmissionError, match="identity|workspace"):
                kb._mark_execution_launch_spawning(
                    conn, claimed, str(Path(claimed.workspace_path)),
                )


def test_real_child_cannot_reach_main_module_after_parent_eof(kanban_home, tmp_path):
    sentinel = tmp_path / "startup-reached"
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn)
        local_lock = f"{kb._claimer_id().split(':', 1)[0]}:child-test"
        claimed = kb.claim_task(conn, worker, claimer=local_lock)
        assert claimed is not None
        env = dict(os.environ)
        env.update({
            "HERMES_KANBAN_LAUNCH_REQUIRED": "1",
            "HERMES_KANBAN_DB": claimed.execution_launch_db_path,
            "HERMES_KANBAN_TASK": worker,
            "HERMES_KANBAN_RUN_ID": str(claimed.current_run_id),
            "HERMES_KANBAN_CLAIM_LOCK": claimed.claim_lock,
        })
        code = (
            "import hermes_cli.main; "
            f"open({str(sentinel)!r}, 'w', encoding='utf-8').write('reached')"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                proc.pid,
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        assert proc.stdin is not None
        proc.stdin.close()  # parent died before publishing the secret
        proc.wait(timeout=15)
        assert proc.returncode != 0
        assert not sentinel.exists()
        assert kb.reclaim_task(conn, worker, reason="child EOF cleanup")


def test_real_child_reaches_main_only_after_pid_bound_authorization(
    kanban_home, tmp_path,
):
    sentinel = tmp_path / "startup-authorized"
    env_dump = tmp_path / "startup-env.json"
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "authorized-child")
        local_lock = f"{kb._claimer_id().split(':', 1)[0]}:authorized-child"
        claimed = kb.claim_task(conn, worker, claimer=local_lock)
        assert claimed is not None
        env = dict(os.environ)
        env.update({
            "HERMES_KANBAN_LAUNCH_REQUIRED": "1",
            "HERMES_KANBAN_DB": claimed.execution_launch_db_path,
            "HERMES_KANBAN_TASK": worker,
            "HERMES_KANBAN_RUN_ID": str(claimed.current_run_id),
            "HERMES_KANBAN_CLAIM_LOCK": claimed.claim_lock,
        })
        assert claimed.execution_launch_nonce not in json.dumps(env)
        code = (
            "import json, os; import hermes_cli.main; "
            f"open({str(sentinel)!r}, 'w', encoding='utf-8').write('authorized'); "
            f"open({str(env_dump)!r}, 'w', encoding='utf-8').write(json.dumps(dict(os.environ)))"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                proc.pid,
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({
            "launch_id": claimed.execution_launch_id,
            "nonce": claimed.execution_launch_nonce,
        }).encode("utf-8") + b"\n")
        proc.stdin.close()
        proc.wait(timeout=20)
        assert proc.returncode == 0, proc.stderr.read().decode("utf-8", errors="replace")
        assert sentinel.read_text(encoding="utf-8") == "authorized"
        child_env = json.loads(env_dump.read_text(encoding="utf-8"))
        assert claimed.execution_launch_nonce not in json.dumps(child_env)
        assert "HERMES_KANBAN_LAUNCH_REQUIRED" not in child_env
        launch = conn.execute(
            "SELECT state, exit_requested_at FROM execution_launches "
            "WHERE task_id = ?", (worker,)
        ).fetchone()
        assert launch["state"] == "started"
        assert launch["exit_requested_at"] is not None
        assert kb.reconcile_execution_launch_exits(conn) == 1
        launch = conn.execute(
            "SELECT state FROM execution_launches WHERE task_id = ?", (worker,)
        ).fetchone()
        assert launch["state"] == "exited"


def test_shutdown_request_cannot_certify_a_still_running_child(
    kanban_home, tmp_path,
):
    sentinel = tmp_path / "alive-after-exit-request"
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn, "shutdown-order")
        claimed = kb.claim_task(
            conn,
            worker,
            claimer=f"{kb._claimer_id().split(':', 1)[0]}:shutdown-order",
        )
        assert claimed is not None
        env = dict(os.environ)
        env.update({
            "HERMES_KANBAN_LAUNCH_REQUIRED": "1",
            "HERMES_KANBAN_DB": claimed.execution_launch_db_path,
            "HERMES_KANBAN_TASK": worker,
            "HERMES_KANBAN_RUN_ID": str(claimed.current_run_id),
            "HERMES_KANBAN_CLAIM_LOCK": claimed.claim_lock,
        })
        code = (
            "import atexit, time\n"
            "def linger():\n"
            f" open({str(sentinel)!r}, 'w', encoding='utf-8').write('alive')\n"
            " time.sleep(2)\n"
            "atexit.register(linger)\n"
            "import hermes_cli.main\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                proc.pid,
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({
            "launch_id": claimed.execution_launch_id,
            "nonce": claimed.execution_launch_nonce,
        }).encode("utf-8") + b"\n")
        proc.stdin.close()
        deadline = time.time() + 15
        while not sentinel.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert sentinel.exists()
        assert proc.poll() is None
        launch = conn.execute(
            "SELECT state, exit_requested_at, exited_at FROM execution_launches "
            "WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert launch["state"] == "started"
        assert launch["exit_requested_at"] is not None
        assert launch["exited_at"] is None
        assert kb.reconcile_execution_launch_exits(conn) == 0
        with pytest.raises(kb.ExecutionAdmissionError, match="cannot seal"):
            kb.seal_execution_admission(
                conn,
                authorization_id=AUTH,
                generation=GENERATION,
                policy_sha256=bound["policy_sha256"],
                reason="still-running shutdown callback",
            )
        proc.wait(timeout=20)
        assert proc.returncode == 0, proc.stderr.read().decode(
            "utf-8", errors="replace",
        )
        assert kb.reconcile_execution_launch_exits(conn) == 1
        assert conn.execute(
            "SELECT state FROM execution_launches WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()["state"] == "exited"


def test_external_observer_finds_dead_worker_after_task_became_terminal(
    kanban_home,
):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "dead-after-terminal")
        claimed = kb.claim_task(
            conn,
            worker,
            claimer=f"{kb._claimer_id().split(':', 1)[0]}:dead-terminal",
        )
        assert claimed is not None
        env = dict(os.environ)
        env.update({
            "HERMES_KANBAN_LAUNCH_REQUIRED": "1",
            "HERMES_KANBAN_DB": claimed.execution_launch_db_path,
            "HERMES_KANBAN_TASK": worker,
            "HERMES_KANBAN_RUN_ID": str(claimed.current_run_id),
            "HERMES_KANBAN_CLAIM_LOCK": claimed.claim_lock,
        })
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os; import hermes_cli.main; os._exit(0)",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                proc.pid,
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({
            "launch_id": claimed.execution_launch_id,
            "nonce": claimed.execution_launch_nonce,
        }).encode("utf-8") + b"\n")
        proc.stdin.close()
        proc.wait(timeout=20)
        assert proc.returncode == 0, proc.stderr.read().decode(
            "utf-8", errors="replace",
        )
        launch = conn.execute(
            "SELECT state, exit_requested_at FROM execution_launches "
            "WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert launch["state"] == "started"
        assert launch["exit_requested_at"] is None
        # Simulate the worker having already made its business task terminal;
        # process reconciliation must not depend on that live task phase.
        conn.execute(
            "UPDATE tasks SET status = 'done', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
            (worker,),
        )
        conn.commit()
        assert kb.reconcile_execution_launch_exits(conn) == 1
        assert conn.execute(
            "SELECT state FROM execution_launches WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()["state"] == "exited"


def test_reloaded_task_cannot_fall_back_to_legacy_spawn(kanban_home):
    invoked = []
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn, "reload-capability")
        claimed = kb.claim_task(conn, worker, claimer="reload:claim")
        assert claimed is not None
        assert kb.reclaim_task(conn, worker, reason="reload probe")
        kb.seal_execution_admission(
            conn,
            authorization_id=AUTH,
            generation=GENERATION,
            policy_sha256=bound["policy_sha256"],
            reason="reload probe sealed",
        )
        reloaded = kb.get_task(conn, worker)
        assert reloaded is not None
        assert reloaded.execution_launch_id is None

        def spawn(*_args, **_kwargs):
            invoked.append(True)
            return None

        with pytest.raises(kb.ExecutionAdmissionError, match="missing.*capability"):
            kb._spawn_claimed_task(
                conn,
                reloaded,
                str(Path(reloaded.workspace_path)),
                board=None,
                spawn_fn=spawn,
            )
        assert invoked == []


def test_stripped_launch_marker_is_refused_before_main_import(
    kanban_home, tmp_path,
):
    sentinel = tmp_path / "marker-stripped-reached"
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "stripped-marker")
        env = dict(os.environ)
        env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
        env.update({
            "HERMES_KANBAN_DB": str(kb._connection_db_path(conn)),
            "HERMES_KANBAN_TASK": worker,
        })
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import hermes_cli.main; "
                f"open({str(sentinel)!r}, 'w', encoding='utf-8').write('bad')",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=20,
            check=False,
        )
        assert proc.returncode != 0
        assert not sentinel.exists()
        assert b"one-use launch capability" in proc.stderr


def test_worker_environment_allows_read_only_kanban_inspection(
    kanban_home,
):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "read-only-child-cli")
        env = dict(os.environ)
        env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
        env.update({
            "HERMES_KANBAN_DB": str(kb._connection_db_path(conn)),
            "HERMES_KANBAN_TASK": worker,
        })
        code = (
            "import sys; "
            f"sys.argv = ['hermes', 'kanban', 'show', {worker!r}, '--json']; "
            "import hermes_cli.main; print('inspection-import-ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=20,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr.decode(
            "utf-8", errors="replace",
        )
        assert b"inspection-import-ok" in proc.stdout


def test_real_read_only_show_preserves_legacy_board_bytes_and_events(
    kanban_home,
):
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "read-only-real-command")
        event_id = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? ORDER BY id LIMIT 1",
            (worker,),
        ).fetchone()["id"]
        conn.execute(
            "UPDATE task_events SET kind = 'ready' WHERE id = ?",
            (event_id,),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db_path = kb._connection_db_path(conn)

    before = db_path.read_bytes()
    env = dict(os.environ)
    env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
    env.update({
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_TASK": worker,
    })
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "show",
            worker,
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    payload = json.loads(proc.stdout)
    assert payload["task"]["id"] == worker
    assert any(event["kind"] == "ready" for event in payload["events"])
    assert db_path.read_bytes() == before

    raw = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        assert raw.execute(
            "SELECT kind FROM task_events WHERE id = ?", (event_id,)
        ).fetchone()[0] == "ready"
    finally:
        raw.close()


def test_read_only_board_connection_denies_sql_and_pragma_mutation(
    kanban_home,
):
    with kb.connect() as conn:
        worker = _task(conn, "read-only-authorizer")
        db_path = kb._connection_db_path(conn)

    with kb.connect_readonly_closing(db_path=db_path) as conn:
        assert kb.get_task(conn, worker).id == worker
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute(
                "UPDATE tasks SET title = 'mutated' WHERE id = ?",
                (worker,),
            )
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute("PRAGMA query_only=OFF")


def test_read_only_show_refuses_absent_board_without_creating_it(
    tmp_path,
):
    missing = tmp_path / "missing" / "kanban.db"
    task_id = "t_missingboard"
    env = dict(os.environ)
    env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
    env.update({
        "HERMES_KANBAN_DB": str(missing),
        "HERMES_KANBAN_TASK": task_id,
    })
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "show",
            task_id,
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=20,
        check=False,
    )
    assert proc.returncode != 0
    assert b"requires an existing board database" in proc.stderr
    assert not missing.exists()
    assert not missing.parent.exists()


def test_read_only_show_refuses_corrupt_board_without_changing_it(tmp_path):
    corrupt = tmp_path / "kanban.db"
    original = b"not a sqlite database\n"
    corrupt.write_bytes(original)
    task_id = "t_corruptboard"
    env = dict(os.environ)
    env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
    env.update({
        "HERMES_KANBAN_DB": str(corrupt),
        "HERMES_KANBAN_TASK": task_id,
    })
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "show",
            task_id,
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=20,
        check=False,
    )
    assert proc.returncode != 0
    assert b"read-only inspection refused" in proc.stderr
    assert corrupt.read_bytes() == original
    assert not list(tmp_path.glob("kanban.db.corrupt.*"))


def test_worker_environment_rejects_agent_override_before_kanban_show(
    kanban_home, tmp_path,
):
    sentinel = tmp_path / "argv-override-reached"
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "argv-override")
        env = dict(os.environ)
        env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
        env.update({
            "HERMES_KANBAN_DB": str(kb._connection_db_path(conn)),
            "HERMES_KANBAN_TASK": worker,
        })
        code = (
            "import sys; "
            f"sys.argv = ['hermes', '-z', 'prompt', 'kanban', 'show', "
            f"{worker!r}, '--json']; "
            "import hermes_cli.main; "
            f"open({str(sentinel)!r}, 'w', encoding='utf-8').write('bad')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=20,
            check=False,
        )
        assert proc.returncode != 0
        assert not sentinel.exists()
        assert b"one-use launch capability" in proc.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["-z", "prompt", "kanban", "show", "{task}", "--json"],
        ["--oneshot=prompt", "kanban", "show", "{task}", "--json"],
        [
            "--provider", "openai-codex", "--model", "gpt-5.6-sol",
            "-z", "prompt", "kanban", "show", "{task}", "--json",
        ],
    ],
)
def test_real_main_parser_agent_entry_cannot_masquerade_as_inspection(
    kanban_home, tmp_path, argv,
):
    sentinel = tmp_path / "real-parser-inference-reached"
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "real-parser-override")
        env = dict(os.environ)
        env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
        env.update({
            "HERMES_KANBAN_DB": str(kb._connection_db_path(conn)),
            "HERMES_KANBAN_TASK": worker,
        })
        rendered_argv = [worker if item == "{task}" else item for item in argv]
        code = (
            "import pathlib, runpy, sys, types; "
            "fake = types.ModuleType('hermes_cli.oneshot'); "
            f"fake.run_oneshot = lambda *a, **k: "
            f"(pathlib.Path({str(sentinel)!r}).write_text('bad'), 0)[1]; "
            "sys.modules['hermes_cli.oneshot'] = fake; "
            f"sys.argv = ['hermes', *{rendered_argv!r}]; "
            "runpy.run_module('hermes_cli.main', run_name='__main__')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=20,
            check=False,
        )
        assert proc.returncode != 0
        assert not sentinel.exists()
        assert b"one-use launch capability" in proc.stderr


def test_direct_cli_py_entry_cannot_bypass_worker_admission(
    kanban_home, tmp_path,
):
    sentinel = tmp_path / "direct-cli-entry-reached"
    repo_root = Path(__file__).resolve().parents[2]
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "direct-cli-override")
        env = dict(os.environ)
        env.pop("HERMES_KANBAN_LAUNCH_REQUIRED", None)
        env.update({
            "HERMES_KANBAN_DB": str(kb._connection_db_path(conn)),
            "HERMES_KANBAN_TASK": worker,
        })
        code = (
            "import pathlib, runpy, sys, types; "
            "fake = types.ModuleType('fire'); "
            f"fake.Fire = lambda *a, **k: "
            f"pathlib.Path({str(sentinel)!r}).write_text('bad'); "
            "sys.modules['fire'] = fake; "
            f"sys.argv = ['cli.py', 'kanban', 'show', {worker!r}]; "
            f"runpy.run_path({str(repo_root / 'cli.py')!r}, run_name='__main__')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=20,
            check=False,
        )
        assert proc.returncode != 0
        assert not sentinel.exists()
        assert b"one-use launch capability" in proc.stderr


def test_unknown_process_birth_closes_pipe_without_signalling(
    kanban_home, monkeypatch,
):
    class Writer:
        closed = False

        def close(self):
            self.closed = True

    class Process:
        def wait(self, timeout=None):
            raise TimeoutError(timeout)

        def poll(self):
            return None

    writer = Writer()
    signals = []
    monkeypatch.setattr(kb, "_worker_process_start_time", lambda _pid: None)
    monkeypatch.setattr(
        kb,
        "_terminate_reclaimed_worker",
        lambda *_args, **_kwargs: signals.append(True),
    )
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn, "unknown-birth")
        claimed = kb.claim_task(conn, worker, claimer="unknown:birth")
        assert claimed is not None

        def spawn(_task, _workspace, *, board=None, execution_launch=None):
            assert execution_launch is not None
            return kb.SpawnedWorker(
                pid=765432,
                startup_writer=writer,
                process=Process(),
            )

        with pytest.raises(
            kb.ExecutionAdmissionError,
            match="process start time is unavailable",
        ):
            kb._spawn_claimed_task(
                conn,
                claimed,
                str(Path(claimed.workspace_path)),
                board=None,
                spawn_fn=spawn,
            )
        assert writer.closed
        assert signals == []
        launch = conn.execute(
            "SELECT state, worker_pid, worker_start_time FROM execution_launches "
            "WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert launch["state"] == "spawning"
        assert launch["worker_pid"] == 765432
        assert launch["worker_start_time"] is None
        assert kb.get_task(conn, worker).worker_pid is None
        with pytest.raises(kb.ExecutionAdmissionError, match="cannot seal"):
            kb.seal_execution_admission(
                conn,
                authorization_id=AUTH,
                generation=GENERATION,
                policy_sha256=bound["policy_sha256"],
                reason="unknown process identity remains held",
            )


def test_recycled_pid_cleanup_uses_only_durable_birth_marker(
    kanban_home, monkeypatch,
):
    class Writer:
        def write(self, _payload):
            raise BrokenPipeError("probe")

        def flush(self):
            pass

        def close(self):
            pass

    class Process:
        def wait(self, timeout=None):
            raise TimeoutError(timeout)

        def poll(self):
            return None

    monkeypatch.setattr(kb, "_worker_process_start_time", lambda _pid: 111)
    observed_expected = []

    def observe(_pid, expected_start_time):
        observed_expected.append(expected_start_time)
        return {
            "state": "dead",
            "reason": "pid_reused",
            "observed_start_time": 222,
            "expected_start_time": expected_start_time,
        }

    monkeypatch.setattr(kb, "_observe_execution_process_identity", observe)
    monkeypatch.setattr(
        kb.os,
        "kill",
        lambda *_args: pytest.fail("recycled PID must not be signalled"),
    )
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "recycled-pid")
        claimed = kb.claim_task(
            conn,
            worker,
            claimer=f"{kb._claimer_id().split(':', 1)[0]}:recycled-pid",
        )
        assert claimed is not None

        def spawn(_task, _workspace, *, board=None, execution_launch=None):
            assert execution_launch is not None
            return kb.SpawnedWorker(
                pid=876543,
                startup_writer=Writer(),
                process=Process(),
            )

        with pytest.raises(BrokenPipeError, match="probe"):
            kb._spawn_claimed_task(
                conn,
                claimed,
                str(Path(claimed.workspace_path)),
                board=None,
                spawn_fn=spawn,
            )
        launch = conn.execute(
            "SELECT state, worker_pid, worker_start_time FROM execution_launches "
            "WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert launch["state"] == "failed"
        assert launch["worker_pid"] == 876543
        assert launch["worker_start_time"] == 111
        assert observed_expected == [111]


@pytest.mark.parametrize("terminal_action", ["complete", "block"])
def test_terminal_task_race_during_revoke_preserves_worker_result(
    kanban_home, monkeypatch, terminal_action,
):
    monkeypatch.setattr(kb, "_worker_process_start_time", lambda _pid: 111)
    terminalized = False

    def observe(_pid, expected_start_time):
        assert expected_start_time == 111
        return {
            "state": "dead" if terminalized else "alive",
            "reason": "probe",
            "observed_start_time": 111,
            "expected_start_time": expected_start_time,
        }

    monkeypatch.setattr(kb, "_observe_execution_process_identity", observe)
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn, f"revoke-{terminal_action}")
        claimed = kb.claim_task(
            conn,
            worker,
            claimer=f"{kb._claimer_id().split(':', 1)[0]}:revoke-race",
        )
        assert claimed is not None
        with kb._execution_launch_fence(conn):
            kb._mark_execution_launch_spawning(
                conn, claimed, str(Path(claimed.workspace_path)),
            )
            kb._set_worker_pid(
                conn,
                worker,
                654321,
                expected_run_id=claimed.current_run_id,
                expected_claim_lock=claimed.claim_lock,
                execution_launch_id=claimed.execution_launch_id,
            )
        kb.consume_execution_launch(
            conn,
            launch_id=claimed.execution_launch_id,
            nonce=claimed.execution_launch_nonce,
            task_id=worker,
            run_id=claimed.current_run_id,
            claim_lock=claimed.claim_lock,
            worker_pid=654321,
        )

        def finish_task(_pid, _signal):
            nonlocal terminalized
            with kb.connect() as other:
                if terminal_action == "complete":
                    assert kb.complete_task(
                        other,
                        worker,
                        result="preserved-result",
                        expected_run_id=claimed.current_run_id,
                        fire_lifecycle_hook=False,
                    )
                else:
                    assert kb.block_task(
                        other,
                        worker,
                        reason="preserved-block",
                        kind="needs_input",
                        expected_run_id=claimed.current_run_id,
                    )
            terminalized = True

        assert kb.reclaim_task(
            conn,
            worker,
            reason="concurrent terminal race",
            signal_fn=finish_task,
        ) is False
        task = kb.get_task(conn, worker)
        assert task is not None
        assert task.status == ("done" if terminal_action == "complete" else "blocked")
        if terminal_action == "complete":
            assert task.result == "preserved-result"
        launch = conn.execute(
            "SELECT state FROM execution_launches WHERE launch_id = ?",
            (claimed.execution_launch_id,),
        ).fetchone()
        assert launch["state"] == "exited"
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE id = ?",
            (claimed.current_run_id,),
        ).fetchone()
        assert run["outcome"] == (
            "completed" if terminal_action == "complete" else "blocked"
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?",
            (worker,),
        ).fetchone()[0] == 1
        sealed = kb.seal_execution_admission(
            conn,
            authorization_id=AUTH,
            generation=GENERATION,
            policy_sha256=bound["policy_sha256"],
            reason="terminal race safely reconciled",
        )
        assert sealed["state"] == "sealed"


def test_default_spawn_refuses_live_admission_without_launch_context(
    kanban_home, monkeypatch,
):
    popen_calls = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: popen_calls.append(True),
    )
    with kb.connect() as conn:
        worker, _bound = _activate_one_worker(conn, "direct-default-spawn")
        task = kb.get_task(conn, worker)
        assert task is not None
        with pytest.raises(kb.ExecutionAdmissionError, match="unguarded"):
            kb._default_spawn(task, str(Path(task.workspace_path)))
        assert popen_calls == []


def test_review_identity_binds_the_injected_review_skill(kanban_home):
    with kb.connect() as conn:
        worker, bound = _activate_one_worker(conn, "reviewer", lane="review")
        identity = bound["tasks"][0]["identity"]
        assert "sdlc-review" in identity["skills"]
        claimed = kb.claim_review_task(conn, worker, claimer="review:test")
        assert claimed is not None
        assert claimed.execution_launch_lane == "review"


def _cli(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    kanban_cli.build_parser(sub)
    return kanban_cli.kanban_command(parser.parse_args(["kanban", *argv]))


def test_cli_requires_orchestrator_and_round_trips_exact_policy(
    kanban_home, tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(
        kanban_cli,
        "_configured_kanban_orchestrator_profile",
        lambda: "coordinator",
    )
    monkeypatch.setenv("HERMES_PROFILE", "specialist")
    assert _cli([
        "admission", "begin", AUTH,
        "--generation", str(GENERATION),
        "--evidence-sha256", EVIDENCE,
        "--json",
    ]) == 1
    assert "only configured orchestrator" in capsys.readouterr().err

    monkeypatch.setenv("HERMES_PROFILE", "coordinator")
    assert _cli([
        "admission", "begin", AUTH,
        "--generation", str(GENERATION),
        "--evidence-sha256", EVIDENCE,
        "--json",
    ]) == 0
    began = json.loads(capsys.readouterr().out)
    assert began["state"] == "prepared"

    with kb.connect() as conn:
        worker = _task(conn, "cli-collector")
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps({"tasks": [{
        "task_id": worker,
        "execution_kind": "worker",
        "allowed_claim_status": "ready",
    }]}), encoding="utf-8")
    assert _cli([
        "admission", "bind", AUTH,
        "--generation", str(GENERATION),
        "--evidence-sha256", EVIDENCE,
        "--tasks-file", str(tasks_file),
        "--json",
    ]) == 0
    bound = json.loads(capsys.readouterr().out)
    assert bound["tasks"][0]["task_id"] == worker

    assert _cli([
        "admission", "activate", AUTH,
        "--generation", str(GENERATION),
        "--policy-sha256", bound["policy_sha256"],
        "--json",
    ]) == 0
    active = json.loads(capsys.readouterr().out)
    assert active["state"] == "active"

    assert _cli(["admission", "show", AUTH, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["policy_sha256"] == bound["policy_sha256"]

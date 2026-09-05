"""Fail-closed board execution-admission kernel tests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
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
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee="governed-worker",
        created_by="coordinator",
        workspace_kind="dir",
        workspace_path="/tmp/governed-workspace",
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

        def spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return None

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

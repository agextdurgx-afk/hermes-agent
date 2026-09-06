"""Durable cron failure incidents: signature dedup, lifecycle, ack, CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.incidents as incidents
import cron.jobs as cron_jobs
import cron.scheduler as sched


def _point_db(monkeypatch, tmp_path):
    """Point the incident store at a throwaway executions.db (same file shape
    the scheduler uses). ``cron.executions.EXECUTIONS_FILE`` stays None so the
    incident store falls back to its own override."""
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return incidents


def _job(**overrides):
    job = {
        "id": "incident-gating-test",
        "name": "incident gating test",
        "prompt": "hello",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
        "deliver": "local",
        "model": None,
        "provider": None,
        "provider_snapshot": "openrouter",
        "base_url": None,
    }
    job.update(overrides)
    return job


def _tick_failing(job, tmp_path, deliveries, error="boom unrelated"):
    """Run one run_one_job tick whose agent raises ``error`` (the failure
    path that composes the per-run failure ping). Mirrors the drift-alert-once
    harness so the incident gating is exercised through the real scheduler."""
    fake_db = MagicMock()

    def fake_deliver(jb, content, adapters=None, loop=None, **kwargs):
        deliveries.append(content)
        return None

    with cron_jobs.use_cron_store(tmp_path), \
         patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.get_shared_session_db", return_value=fake_db), \
         patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={
                   "api_key": "test-key",
                   "base_url": "https://example.invalid/v1",
                   "provider": "openrouter",
                   "api_mode": "chat_completions",
               }), \
         patch.object(sched, "_deliver_result", side_effect=fake_deliver), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = RuntimeError(error)
        mock_agent_cls.return_value = mock_agent
        sched.run_one_job(dict(job))
    return mock_agent_cls.called


# ── Store + dedup ──────────────────────────────────────────────────────────


def test_new_failure_creates_incident_and_is_new(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)

    inc_id, is_new = inc.upsert_incident("job-1", "Provider timeout: read timed out")

    assert is_new is True
    assert inc_id.startswith("job-1_")
    row = inc.get_incident(inc_id)
    assert row is not None
    assert row["job_id"] == "job-1"
    assert row["state"] == "detected"
    assert row["failure_type"] == "timeout"
    assert row["first_seen_at"] == row["last_seen_at"]
    assert inc.count_incidents() == 1


def test_same_signature_dedups_same_incident(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)

    id1, new1 = inc.upsert_incident("job-1", "Provider timeout: read timed out")
    id2, new2 = inc.upsert_incident("job-1", "PROVIDER TIMEOUT: read timed out   ")

    assert id1 == id2, "normalized (case/whitespace) signature must dedup"
    assert new1 is True
    assert new2 is False
    assert inc.count_incidents() == 1
    # Refresh updates last_seen but never resets an open state.
    assert inc.get_incident(id1)["state"] == "detected"


def test_error_change_mints_new_incident(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)

    id1, _ = inc.upsert_incident("job-1", "provider timeout")
    id2, new2 = inc.upsert_incident("job-1", "provider rate limit 429")

    assert id1 != id2
    assert new2 is True
    assert inc.count_incidents() == 2


def test_errors_with_same_long_prefix_keep_distinct_incidents(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    common = "structured runner envelope " + ("x" * 800)

    contention_id, contention_new = inc.upsert_incident(
        "job-1", f"{common} maintenance admission is busy with live process 42"
    )
    integrity_id, integrity_new = inc.upsert_incident(
        "job-1", f"{common} structural health found an integrity failure"
    )

    assert contention_new is True
    assert integrity_new is True
    assert contention_id != integrity_id
    assert inc.count_incidents() == 2


def test_complete_long_error_still_dedups_after_display_truncation(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    error = "same structured envelope " + ("x" * 600) + " exact terminal cause"

    first_id, first_new = inc.upsert_incident("job-1", error)
    second_id, second_new = inc.upsert_incident("job-1", error.upper())

    assert first_new is True
    assert second_new is False
    assert first_id == second_id
    assert len(inc.get_incident(first_id)["error"]) <= incidents.MAX_ERROR_CHARS


# ── Redaction / classification ─────────────────────────────────────────────


def test_redaction_applied_to_incident_error(monkeypatch, tmp_path):
    # agent.redact snapshots _REDACT_ENABLED from HERMES_REDACT_SECRETS at
    # module-import time. When another collected test module imports the
    # gateway/scheduler chain (e.g. test_codex_execution_paths.py), that
    # import happens at COLLECTION time — before the conftest env scrub —
    # so a developer shell exporting HERMES_REDACT_SECRETS=false freezes
    # redaction off and this test fails only in full-directory runs.
    # Pin the flag explicitly, matching the repo-wide pattern.
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)
    inc = _point_db(monkeypatch, tmp_path)
    secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"

    inc_id, _ = inc.upsert_incident("job-1", f"failed: {secret} boom")

    row = inc.get_incident(inc_id)
    assert secret not in row["error"]
    assert "boom" in row["error"]


def test_error_truncated_to_bounded_length(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    long_error = "x" * 2000

    inc_id, _ = inc.upsert_incident("job-1", long_error)

    assert len(inc.get_incident(inc_id)["error"]) <= 500


def test_failure_type_classification(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    cases = [
        ("delivery failed for telegram chat", "delivery"),
        ("Provider read timed out after 60s", "timeout"),
        ("authentication failed: invalid API key", "auth"),
        ("HTTP 429: rate limit exceeded", "rate_limit"),
        ("configuration validation blocked the run", "config"),
        ("script exited with code 1", "script"),
        ("agent crashed mid-conversation", "agent"),
        ("something completely unexpected happened", "unknown"),
    ]
    for error, expected in cases:
        assert inc._classify_failure_type(error) == expected, (error, expected)


# ── Lifecycle / ack ────────────────────────────────────────────────────────


def test_lifecycle_transitions(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "boom")

    assert inc.get_incident(inc_id)["state"] == "detected"
    assert inc.set_incident_state(inc_id, "alerted") is True
    assert inc.set_incident_state(inc_id, "closed") is True
    row = inc.get_incident(inc_id)
    assert row["state"] == "closed"
    assert row["acked_at"] and row["closed_at"]

    # Closed is terminal for that signature: no re-open, no re-transition.
    assert inc.set_incident_state(inc_id, "alerted") is False
    assert inc.set_incident_state(inc_id, "closed") is False
    assert inc.ack_incident(inc_id) is False
    assert inc.get_incident(inc_id)["state"] == "closed"

    # Invalid states are rejected, not raised.
    assert inc.set_incident_state(inc_id, "bogus") is False
    assert inc.list_incidents(state="bogus") == []
    assert inc.count_incidents(state="bogus") == 0


def test_acked_signature_stays_closed_on_refresh(monkeypatch, tmp_path):
    """Ack is per-signature: upserting the same error after ack must NOT
    resurrect the incident — a changed error is what mints a new one."""
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "same failure text")
    inc.ack_incident(inc_id)

    same_id, is_new = inc.upsert_incident("job-1", "SAME FAILURE TEXT")

    assert same_id == inc_id
    assert is_new is False
    assert inc.get_incident(inc_id)["state"] == "closed"


# ── Missing DB / lazy schema ───────────────────────────────────────────────


def test_missing_db_no_crash(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)

    inc_id, is_new = inc.upsert_incident("job-1", "boom")

    assert is_new is True
    assert (tmp_path / "cron" / "executions.db").is_file()
    assert inc.list_incidents() == [inc.get_incident(inc_id)]
    assert inc.count_incidents() == 1
    assert inc.get_incident("nope") is None


# ── Scheduler gating ───────────────────────────────────────────────────────


def test_unacked_failure_still_alerts(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    deliveries = []
    job = _job()
    with cron_jobs.use_cron_store(tmp_path):
        cron_jobs.save_jobs([job])
        _tick_failing(job, tmp_path, deliveries, error="unacked boom")
        _tick_failing(job, tmp_path, deliveries, error="unacked boom")

    assert len(deliveries) == 2, "unacked failures must keep alerting per run"
    rows = inc.list_incidents()
    assert len(rows) == 1
    assert rows[0]["state"] == "detected"


def test_ack_suppresses_alert_until_signature_changes(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    deliveries = []
    job = _job()
    with cron_jobs.use_cron_store(tmp_path):
        cron_jobs.save_jobs([job])
        # First failure: alert delivered, incident minted.
        _tick_failing(job, tmp_path, deliveries, error="boom signature A")
        assert len(deliveries) == 1
        rows = inc.list_incidents()
        assert len(rows) == 1 and rows[0]["state"] == "detected"
        inc_id = rows[0]["id"]

        # Acknowledge it.
        assert inc.ack_incident(inc_id) is True

        # Same signature: alert suppressed, incident stays closed.
        _tick_failing(job, tmp_path, deliveries, error="boom signature A")
        assert len(deliveries) == 1, "acked signature must not re-ping"
        assert inc.get_incident(inc_id)["state"] == "closed"

        # Changed signature: new incident, alert again.
        _tick_failing(job, tmp_path, deliveries, error="boom signature B")
        assert len(deliveries) == 2, "changed signature must re-alert"
        assert inc.count_incidents() == 2


def test_mark_incident_alerted_sets_state_never_resurrects(monkeypatch, tmp_path):
    """The post-delivery 'alerted' transition records that a ping went out,
    and is a no-op on a closed (acked) incident — it can never resurrect one."""
    inc = _point_db(monkeypatch, tmp_path)

    inc_id, _ = inc.upsert_incident("job-1", "boom")
    sched._mark_incident_alerted(inc_id)
    assert inc.get_incident(inc_id)["state"] == "alerted"

    inc.ack_incident(inc_id)
    sched._mark_incident_alerted(inc_id)
    assert inc.get_incident(inc_id)["state"] == "closed"

    # Best-effort: bad/missing ids never raise.
    sched._mark_incident_alerted(None)
    sched._mark_incident_alerted("nonexistent")


def test_best_effort_incident_store_failure_returns_false(monkeypatch, tmp_path):
    """An incident-store error must never break the cron delivery path."""
    _point_db(monkeypatch, tmp_path)
    with patch("cron.incidents.upsert_incident",
               side_effect=RuntimeError("db locked")):
        assert sched._upsert_incident_for_failure(_job(), "boom") == (False, None)


# ── CLI ────────────────────────────────────────────────────────────────────


def test_cli_list_and_ack(monkeypatch, tmp_path, capsys):
    from hermes_cli.cron import cron_incidents

    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "provider timeout boom")

    # List.
    list_args = argparse.Namespace(
        incident_action="list", state=None, incident_id=None
    )
    assert cron_incidents(list_args) == 0
    out = capsys.readouterr().out
    assert inc_id in out
    assert "job-1" in out

    # State filter.
    filter_args = argparse.Namespace(
        incident_action="list", state="closed", incident_id=None
    )
    assert cron_incidents(filter_args) == 0
    out = capsys.readouterr().out
    assert "No cron failure incidents recorded." in out

    # Ack.
    ack_args = argparse.Namespace(
        incident_action="ack", state=None, incident_id=inc_id
    )
    assert cron_incidents(ack_args) == 0
    assert inc.get_incident(inc_id)["state"] == "closed"
    out = capsys.readouterr().out
    assert "acknowledged" in out.lower()

    # Ack again: already closed, still a clean exit.
    assert cron_incidents(ack_args) == 0
    out = capsys.readouterr().out
    assert "already closed" in out.lower()

    # Ack with a missing id is a usage error.
    missing_args = argparse.Namespace(
        incident_action="ack", state=None, incident_id=None
    )
    assert cron_incidents(missing_args) == 1


def _cas_fixture(monkeypatch, tmp_path):
    import hashlib
    from cron import executions
    inc = _point_db(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", inc.EXECUTIONS_FILE)
    with executions._transaction():
        pass
    job = "causal-job-123"
    errors = ["Script exited with code 1\nstdout:\nrefusal " + "x" * 600,
              "Script exited with code 1\nstdout:\nhealth failure caused by refusal"]
    logs = []
    rows = []
    with executions._transaction() as conn:
        for index, error in enumerate(errors):
            log = tmp_path / f"{index}.md"
            log.write_text(f"# Cron Job\n\n**Job ID:** {job}\n\n{error}\n")
            content = log.read_bytes()
            logs.append({"path": str(log), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            row = dict(id=f"execution-{index}", job_id=job, source="builtin", process_id="owned-process",
                       pid=123, process_started_at=456, status="failed", handoff_pending=0, handoff_started_at=None,
                       claimed_at=f"2099-01-01T00:0{index}:00+00:00", started_at=f"2099-01-01T00:0{index}:01+00:00",
                       finished_at=f"2099-01-01T00:0{index}:02+00:00", error=error)
            conn.execute(f"INSERT INTO executions ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", list(row.values()))
            row["error_sha256"] = hashlib.sha256(row.pop("error").encode()).hexdigest()
            rows.append(row)
    incident_id, _ = inc.upsert_incident(job, errors[-1], output_file=logs[-1]["path"])
    entry = {"incident": inc.get_incident(incident_id), "execution": rows[-1], "log": logs[-1],
             "causal_executions": [{"execution": row, "log": log} for row, log in zip(rows, logs)]}
    return inc, {"schema_version": 1, "acknowledgement_id": "reviewed-operation", "job_id": job,
                 "incidents": [entry], "latest_execution": rows[-1]}


def test_conditional_ack_is_atomic_idempotent_and_rejects_different_request(monkeypatch, tmp_path):
    import copy
    import pytest
    inc, request = _cas_fixture(monkeypatch, tmp_path)
    receipt = inc.acknowledge_incidents_cas(request)
    assert receipt["incidents"][0]["state"] == "closed"
    # Simulates a lost response immediately after COMMIT.
    assert inc.acknowledge_incidents_cas(request) == receipt
    changed = copy.deepcopy(request)
    changed["incidents"][0]["incident"]["error"] += "unrelated error"
    with pytest.raises(ValueError, match="identity was reused"):
        inc.acknowledge_incidents_cas(changed)


def test_conditional_ack_refuses_changed_incident_causal_rows_or_logs(monkeypatch, tmp_path):
    import pytest
    from cron import executions
    inc, request = _cas_fixture(monkeypatch, tmp_path)
    original = request["incidents"][0]
    # A common error prefix cannot bind a different complete execution cause.
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET error=error || 'unrelated tail' WHERE id='execution-0'")
    with pytest.raises(ValueError, match="causal execution changed"):
        inc.acknowledge_incidents_cas(request)
    assert inc.get_incident(original["incident"]["id"])["state"] == "detected"


def test_conditional_ack_refuses_substituted_log_and_active_job(monkeypatch, tmp_path):
    import pytest
    from cron import executions
    inc, request = _cas_fixture(monkeypatch, tmp_path)
    log = Path(request["incidents"][0]["causal_executions"][0]["log"]["path"])
    content = log.read_bytes()
    log.write_bytes(content + b"substitution")
    with pytest.raises(ValueError, match="causal log changed"):
        inc.acknowledge_incidents_cas(request)
    log.write_bytes(content)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET status='running' WHERE id='execution-0'")
    with pytest.raises(ValueError, match="drained job"):
        inc.acknowledge_incidents_cas(request)


def test_conditional_ack_rolls_back_closure_if_receipt_write_crashes(monkeypatch, tmp_path):
    import sqlite3
    import pytest
    inc, request = _cas_fixture(monkeypatch, tmp_path)
    class CrashingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("INSERT INTO cron_incident_acknowledgements"):
                raise RuntimeError("crash before atomic receipt")
            return super().execute(sql, parameters)
    connect = inc._connect
    monkeypatch.setattr(inc, "_connect", lambda: sqlite3.connect(inc._db_path(), factory=CrashingConnection))
    with pytest.raises(RuntimeError, match="crash before atomic receipt"):
        inc.acknowledge_incidents_cas(request)
    monkeypatch.setattr(inc, "_connect", connect)
    assert inc.get_incident(request["incidents"][0]["incident"]["id"])["state"] == "detected"
    assert inc.acknowledge_incidents_cas(request)["incidents"][0]["state"] == "closed"


def test_conditional_ack_detects_concurrent_sqlite_change(monkeypatch, tmp_path):
    import sqlite3
    import pytest
    inc, request = _cas_fixture(monkeypatch, tmp_path)
    # A separate connection mutates the exact row after the read snapshot was
    # formed. BEGIN IMMEDIATE must recapture and compare, never generic-ack it.
    other = sqlite3.connect(inc._db_path())
    other.execute("UPDATE cron_incidents SET last_seen_at='2099-02-01' WHERE id=?", (request["incidents"][0]["incident"]["id"],))
    other.commit(); other.close()
    with pytest.raises(ValueError, match="incident content changed"):
        inc.acknowledge_incidents_cas(request)


def test_conditional_ack_two_real_processes_share_one_atomic_receipt(monkeypatch, tmp_path):
    import json
    import os
    import subprocess
    inc, request = _cas_fixture(monkeypatch, tmp_path)
    script = "import json,sys; from cron.incidents import acknowledge_incidents_cas; print(json.dumps(acknowledge_incidents_cas(json.load(sys.stdin)),sort_keys=True))"
    env = {**os.environ, "HERMES_HOME": str(tmp_path)}
    children = [subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for _ in range(2)]
    for child in children:
        child.stdin.write(json.dumps(request)); child.stdin.close(); child.stdin = None
    outputs = [child.communicate(timeout=15) for child in children]
    assert all(child.returncode == 0 for child in children), outputs
    assert json.loads(outputs[0][0]) == json.loads(outputs[1][0])


def test_conditional_ack_exact_pair_closes_both_incidents_and_allows_idle_ticks(monkeypatch, tmp_path):
    import copy
    from cron import executions
    inc, request = _cas_fixture(monkeypatch, tmp_path)
    request.pop("latest_execution")
    health = request["incidents"][0]
    refusal = copy.deepcopy(health)
    with executions._transaction() as conn:
        error = conn.execute("SELECT error FROM executions WHERE id='execution-0'").fetchone()[0]
    incident_id, _ = inc.upsert_incident(request["job_id"], error, output_file=refusal["causal_executions"][0]["log"]["path"])
    refusal.update(incident=inc.get_incident(incident_id), causal_role="refusal",
                   execution=refusal["causal_executions"][0]["execution"], log=refusal["causal_executions"][0]["log"])
    request["incidents"].append(refusal)
    receipt = inc.acknowledge_incidents_cas(request)
    assert len(receipt["incidents"]) == 2
    with executions._transaction() as conn:
        conn.execute("INSERT INTO executions (id,job_id,source,process_id,pid,process_started_at,status,claimed_at) VALUES ('idle-tick',?,'builtin','idle',321,654,'completed','2099-02-01')", (request["job_id"],))
    assert inc.acknowledge_incidents_cas(request) == receipt


def _baseline_fixture(monkeypatch, tmp_path):
    import hashlib,json
    from cron import recovery_baselines as recovery
    inc, previous = _cas_fixture(monkeypatch, tmp_path)
    snapshot = recovery.capture_recovery_snapshot(previous["job_id"])
    baseline = {"mode": "prospective_recovery_baseline_v1", "incident_disposition": "preserved_unresolved",
                "reviewed_tezoff_sha": "a"*40, "reviewed_hermes_sha": "b"*40, "cron_snapshot": snapshot}
    path = tmp_path / "baseline.json"; path.write_text(json.dumps(baseline))
    baseline_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    review = {"verdict":"pass", "reviewer_model":"gpt-6-astra", "schema_version":5, "review_version":5, "replacement_authorized":True, "controls":{f"A{i}":"pass" for i in range(1,13)}, "incident_disposition":"preserved_unresolved", "recovery_baseline_sha256":baseline_sha,
              "reviewed_tezoff_sha":"a"*40,"reviewed_hermes_sha":"b"*40}
    review_path = tmp_path / "review.json"; review_path.write_text(json.dumps(review))
    authority = {"baseline":{"path":str(path),"sha256":baseline_sha},
                 "review":{"path":str(review_path),"sha256":hashlib.sha256(review_path.read_bytes()).hexdigest()}}
    request = {"schema_version":1,"snapshot":snapshot,"authority":authority,
               "baseline_id": recovery._digest({"snapshot":snapshot,"authority":authority})}
    return inc,recovery,request


def test_recovery_baseline_preserves_unresolved_incidents_and_is_idempotent(monkeypatch,tmp_path):
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    before=inc.list_incidents()
    result=recovery.register_recovery_baseline(request)
    assert result["incident_disposition"] == "preserved_unresolved"
    assert recovery.register_recovery_baseline(request) == result
    assert inc.list_incidents() == before
    assert recovery.inspect_recovery_baselines() == [{"request":request,"receipt":result}]


def test_recovery_baseline_refuses_changed_failure_even_with_same_error_prefix(monkeypatch,tmp_path):
    import pytest
    from cron import executions
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET error=error || 'new failure' WHERE id='execution-0'")
    with pytest.raises(ValueError,match="inventory changed"):
        recovery.register_recovery_baseline(request)
    assert inc.list_incidents()[0]["state"] == "detected"


def test_recovery_baseline_later_failures_and_log_drift_are_never_grandfathered(monkeypatch,tmp_path):
    import pytest
    from cron import executions
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    recovery.register_recovery_baseline(request)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET error=error || 'new failure' WHERE id='execution-0'")
    with pytest.raises(ValueError,match="later execution changed"):
        recovery.inspect_recovery_baselines()


def test_recovery_baseline_atomic_registration_requires_drained_job_but_inspection_allows_running_work(monkeypatch,tmp_path):
    import pytest
    from cron import executions
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    with executions._transaction() as conn:
        conn.execute("INSERT INTO executions(id,job_id,source,process_id,pid,status,claimed_at) VALUES('new',?,'builtin','live',123,'running','2099-03-01')",(request["snapshot"]["job_id"],))
    with pytest.raises(ValueError,match="drained"):
        recovery.register_recovery_baseline(request)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET status='completed' WHERE id='new'")
    recovery.register_recovery_baseline(request)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET status='running' WHERE id='new'")
    assert len(recovery.inspect_recovery_baselines()) == 1


def test_recovery_baseline_rejects_review_and_log_substitution(monkeypatch,tmp_path):
    import pytest
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    path=Path(request["authority"]["review"]["path"])
    original=path.read_bytes(); path.write_bytes(original+b" ")
    with pytest.raises(ValueError,match="authority artifact changed"):
        recovery.register_recovery_baseline(request)
    path.write_bytes(original)
    log=Path(request["snapshot"]["logs"][0]["path"]); log.write_text("substituted")
    with pytest.raises(ValueError,match="log lacks exact job"):
        recovery.register_recovery_baseline(request)


def test_recovery_baseline_crash_rolls_back_without_changing_incident(monkeypatch,tmp_path):
    import sqlite3,pytest
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    before=inc.list_incidents()
    connect=inc._connect
    class Crash(sqlite3.Connection):
        def execute(self,sql,parameters=()):
            if sql.startswith("INSERT INTO cron_recovery_baselines"):
                super().execute(sql,parameters)
                raise RuntimeError("lost transaction")
            return super().execute(sql,parameters)
    monkeypatch.setattr(inc,"_connect",lambda:sqlite3.connect(inc._db_path(),factory=Crash))
    with pytest.raises(RuntimeError,match="lost transaction"):
        recovery.register_recovery_baseline(request)
    monkeypatch.setattr(inc,"_connect",connect)
    assert recovery.inspect_recovery_baselines() == []
    assert inc.list_incidents() == before
    recovery.register_recovery_baseline(request)


def test_recovery_baseline_two_real_processes_register_once_without_acknowledgement(monkeypatch,tmp_path):
    import json,os,subprocess
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    script="import json,sys; from cron.recovery_baselines import register_recovery_baseline; print(json.dumps(register_recovery_baseline(json.load(sys.stdin)),sort_keys=True))"
    children=[subprocess.Popen([sys.executable,"-c",script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,"HERMES_HOME":str(tmp_path)}) for _ in range(2)]
    for child in children:
        child.stdin.write(json.dumps(request));child.stdin.close();child.stdin=None
    output=[child.communicate(timeout=15) for child in children]
    assert all(child.returncode==0 for child in children),output
    assert json.loads(output[0][0])==json.loads(output[1][0])
    assert inc.list_incidents()[0]["state"]=="detected"


def test_recovery_baseline_unknown_outcome_never_becomes_operational_debt(monkeypatch,tmp_path):
    import pytest
    from cron import executions
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    recovery.register_recovery_baseline(request)
    with executions._transaction() as conn:
        conn.execute("INSERT INTO executions(id,job_id,source,process_id,pid,status,claimed_at) VALUES('unknown',?,'builtin','lost',123,'unknown','2099-03-01')",(request["snapshot"]["job_id"],))
    with pytest.raises(ValueError,match="unknown execution outcome"):
        recovery.inspect_recovery_baselines()


def test_recovery_baseline_full_review_controls_are_required(monkeypatch,tmp_path):
    import hashlib,json,pytest
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    path = Path(request["authority"]["review"]["path"])
    review = json.loads(path.read_text()); review["controls"]["A12"] = "fail"
    path.write_text(json.dumps(review))
    request["authority"]["review"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    request["baseline_id"] = recovery._digest({"snapshot":request["snapshot"],"authority":request["authority"]})
    with pytest.raises(ValueError,match="prospective authority"):
        recovery.register_recovery_baseline(request)
    assert inc.list_incidents()[0]["state"] == "detected"


def test_recovery_baseline_atomically_retains_every_execution_and_detects_missing_pins(monkeypatch,tmp_path):
    import pytest
    from cron import executions
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    recovery.register_recovery_baseline(request)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
    assert len(recovery.inspect_recovery_baselines()) == 1
    with executions._transaction() as conn:
        conn.execute("DELETE FROM cron_evidence_pins WHERE authority_id=?", (request["baseline_id"],))
    with pytest.raises(ValueError,match="retention evidence changed"):
        recovery.inspect_recovery_baselines()
    assert inc.list_incidents()[0]["state"] == "detected"


def test_conditional_ack_retains_its_causal_rows_through_normal_history_pruning(monkeypatch,tmp_path):
    from cron import executions
    inc,request = _cas_fixture(monkeypatch,tmp_path)
    receipt = inc.acknowledge_incidents_cas(request)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
    assert inc.acknowledge_incidents_cas(request) == receipt


def test_readonly_baseline_observation_allows_active_tick_but_registration_still_requires_idle(monkeypatch,tmp_path):
    import pytest
    from cron import executions
    inc,recovery,request = _baseline_fixture(monkeypatch,tmp_path)
    with executions._transaction() as conn:
        conn.execute("INSERT INTO executions(id,job_id,source,process_id,pid,status,claimed_at) VALUES('tick',?,'builtin','live',123,'running','2099-03-01')", (request['snapshot']['job_id'],))
    assert recovery.inspect_recovery_snapshot(request['snapshot']['job_id']) == request['snapshot']
    with pytest.raises(ValueError, match='drained'):
        recovery.capture_recovery_snapshot(request['snapshot']['job_id'])
    with pytest.raises(ValueError, match='drained'):
        recovery.register_recovery_baseline(request)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET status='unknown' WHERE id='tick'")
    with pytest.raises(ValueError, match='unknown execution outcome'):
        recovery.inspect_recovery_snapshot(request['snapshot']['job_id'])

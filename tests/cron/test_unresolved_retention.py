"""Unreviewed incidents retain evidence before a reviewer creates any pins."""
import sqlite3

import pytest

from cron import executions, incidents
from cron.evidence import retained_log_paths, retain_evidence
from cron.jobs import _prune_job_output


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "cron"
    directory = root / "output" / "alpha"
    directory.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", None)
    logs = [directory / f"2099-01-0{i}.md" for i in range(1, 5)]
    for log in logs: log.write_text("original complete output " + log.name)
    incident, _ = incidents.upsert_incident("alpha", "failure before review", output_file=str(logs[0]))
    with executions._transaction() as conn:
        for identity, job, status in [("a1", "alpha", "failed"), ("a2", "alpha", "failed"), ("success", "alpha", "completed"), ("other", "beta", "failed"), ("unknown", "beta", "unknown")]:
            conn.execute("INSERT INTO executions(id,job_id,source,process_id,pid,status,claimed_at,finished_at,error) VALUES(?,?,'builtin','dead',123,?,'2099-01-01','2099-01-01','complete error')", (identity, job, status))
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    return root, directory, logs, incident


def test_unreviewed_incident_log_survives_rotation_without_disabling_unrelated_cleanup(history):
    root, directory, logs, _ = history
    assert retained_log_paths(root / "executions.db") == {str(logs[0])}
    assert _prune_job_output(directory, keep=1) == 2
    assert logs[0].is_file() and logs[-1].is_file()
    assert not logs[1].exists() and not logs[2].exists()


def test_all_failed_attempts_for_unresolved_job_survive_but_success_and_unrelated_failure_do_not(history):
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
        assert {row[0] for row in conn.execute("SELECT id FROM executions")} == {"a1", "a2", "unknown"}


def test_closed_incident_releases_live_references_but_never_receipt_pins(history):
    root, directory, logs, incident = history
    with incidents._transaction() as conn:
        retain_evidence(conn, "reviewed-history", [
            {"kind": "execution", "identity": "a1", "sha256": "a" * 64},
            {"kind": "log", "identity": str(logs[0]), "sha256": "b" * 64},
        ])
        conn.execute("UPDATE cron_incidents SET state='closed' WHERE id=?", (incident,))
        executions._prune_unlocked(conn)
        assert {row[0] for row in conn.execute("SELECT id FROM executions")} == {"a1", "unknown"}
    assert retained_log_paths(root / "executions.db") == {str(logs[0])}
    assert _prune_job_output(directory, keep=1) == 2
    assert logs[0].is_file()


def test_closed_unpinned_incident_returns_to_normal_retention(history):
    _, directory, logs, incident = history
    with incidents._transaction() as conn:
        conn.execute("UPDATE cron_incidents SET state='closed' WHERE id=?", (incident,))
        executions._prune_unlocked(conn)
        assert [row[0] for row in conn.execute("SELECT id FROM executions")] == ["unknown"]
    assert _prune_job_output(directory, keep=1) == 3
    assert not logs[0].exists()


def test_unreadable_incident_inventory_prevents_deletion(history):
    _, directory, logs, _ = history
    with incidents._transaction() as conn:
        conn.execute("UPDATE cron_incidents SET job_id=''")
    assert _prune_job_output(directory, keep=1) == 0
    assert all(log.is_file() for log in logs)
    with executions._transaction() as conn:
        with pytest.raises(ValueError, match="no job identity"):
            executions._prune_unlocked(conn)
        assert conn.execute("SELECT count(*) FROM executions").fetchone()[0] == 5


def test_missing_log_is_not_reconstructed_and_its_failed_attempts_remain(history):
    root, _, logs, _ = history
    logs[0].unlink()
    assert retained_log_paths(root / "executions.db") == {str(logs[0])}
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
        assert {row[0] for row in conn.execute("SELECT id FROM executions")} == {"a1", "a2", "unknown"}
    assert not logs[0].exists()

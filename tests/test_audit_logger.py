"""
Tests for audit logger with hash chaining.
Skill reference: .claude/skills/secrets-security/SKILL.md
"""

import json
import os
import tempfile
import pytest

from src.utils.audit import AuditLogger


@pytest.fixture
def audit_path():
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w"
    ) as f:
        path = f.name
    # Remove so AuditLogger starts fresh
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestAuditLogger:
    def test_creates_log_file(self, audit_path):
        logger = AuditLogger(audit_path)
        logger.log("test_event", {"key": "value"})
        assert os.path.exists(audit_path)

    def test_log_entry_format(self, audit_path):
        logger = AuditLogger(audit_path)
        logger.log("order_submitted", {"symbol": "AAPL", "qty": 100})

        with open(audit_path) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "order_submitted"
        assert entry["data"]["symbol"] == "AAPL"
        assert "timestamp" in entry
        assert "entry_hash" in entry
        assert "prev_hash" in entry

    def test_hash_chain_integrity(self, audit_path):
        logger = AuditLogger(audit_path)
        logger.log("event_1", {"data": "first"})
        logger.log("event_2", {"data": "second"})
        logger.log("event_3", {"data": "third"})

        assert logger.verify_chain()

    def test_tampered_log_detected(self, audit_path):
        logger = AuditLogger(audit_path)
        logger.log("event_1", {"data": "first"})
        logger.log("event_2", {"data": "second"})

        # Tamper with the first entry
        with open(audit_path) as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["data"]["data"] = "tampered"
        lines[0] = json.dumps(entry) + "\n"

        with open(audit_path, "w") as f:
            f.writelines(lines)

        # Re-create logger to verify
        logger2 = AuditLogger(audit_path)
        assert not logger2.verify_chain()

    def test_genesis_hash_on_new_file(self, audit_path):
        logger = AuditLogger(audit_path)
        assert logger._last_hash == "genesis"

    def test_multiple_entries_sequential(self, audit_path):
        logger = AuditLogger(audit_path)
        for i in range(10):
            logger.log(f"event_{i}", {"index": i})

        with open(audit_path) as f:
            lines = f.readlines()
        assert len(lines) == 10
        assert logger.verify_chain()

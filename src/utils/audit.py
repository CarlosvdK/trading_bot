"""
Audit logging with hash chaining.
Skill reference: .claude/skills/secrets-security/SKILL.md
"""

import hashlib
import json
from datetime import datetime, timezone


class AuditLogger:
    """
    Append-only audit log with hash chaining.
    Each entry includes a hash of the previous entry, making tampering detectable.
    """

    def __init__(self, path: str):
        self.path = path
        self._last_hash = self._compute_file_hash()

    def log(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
            "prev_hash": self._last_hash,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["entry_hash"] = entry_hash

        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._last_hash = entry_hash

    def _compute_file_hash(self) -> str:
        try:
            with open(self.path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return "genesis"

    def verify_chain(self) -> bool:
        """Verify the hash chain integrity of the audit log."""
        try:
            with open(self.path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return True

        prev_hash = "genesis"
        for i, line in enumerate(lines):
            entry = json.loads(line.strip())
            stored_hash = entry.pop("entry_hash")
            if entry.get("prev_hash") != prev_hash and i > 0:
                return False
            entry_str = json.dumps(entry, sort_keys=True)
            computed_hash = hashlib.sha256(entry_str.encode()).hexdigest()
            if computed_hash != stored_hash:
                return False
            prev_hash = stored_hash

        return True

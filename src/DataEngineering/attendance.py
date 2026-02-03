"""
Log attendance with debouncing: same member not logged again within a cooldown.
"""
import json
import time
from pathlib import Path


DEFAULT_ATTENDANCE_PATH = Path(__file__).resolve().parent / "attendance.json"
DEFAULT_DEBOUNCE_SECONDS = 30


class AttendanceLogger:
    def __init__(self, attendance_path=None, debounce_seconds=None):
        self.attendance_path = Path(attendance_path or DEFAULT_ATTENDANCE_PATH)
        self.debounce_seconds = debounce_seconds if debounce_seconds is not None else DEFAULT_DEBOUNCE_SECONDS
        self._last_logged = {}  # member_id -> timestamp

    def mark_present(self, member_id):
        """
        If member_id was not logged in the last debounce_seconds, append a record
        and return True. Otherwise return False.
        """
        now = time.time()
        last = self._last_logged.get(member_id, 0)
        if now - last < self.debounce_seconds:
            return False
        self._last_logged[member_id] = now
        record = {"member_id": member_id, "timestamp": now}
        self._append(record)
        return True

    def _append(self, record):
        """Append one record to the attendance file (JSON array)."""
        self.attendance_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        if self.attendance_path.exists():
            with open(self.attendance_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        records.append(record)
        with open(self.attendance_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)


def get_attendance_logger(attendance_path=None, debounce_seconds=None):
    """Convenience: return an AttendanceLogger instance."""
    return AttendanceLogger(
        attendance_path=attendance_path,
        debounce_seconds=debounce_seconds,
    )

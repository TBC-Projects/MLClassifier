# attendance_logger.py
import csv
from datetime import datetime
from pathlib import Path

# Works when run as: python attendance_logger.py
CSV_FILE = Path(__file__).parent / "attendance.csv"

def log_attendance(name):
    """Log attendance with timestamp."""
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name])
    
    print(f"✅ Logged: {name}")
    print(f"📁 Saved to: {CSV_FILE.absolute()}")

if __name__ == "__main__":
    log_attendance("Alice")
    log_attendance("Bob")
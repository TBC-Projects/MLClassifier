"""
Load and save club member database (members.json).
"""
import json
from pathlib import Path


DEFAULT_MEMBERS_PATH = Path(__file__).resolve().parent / "members.json"


def load_members(members_path=None):
    """
    Load member list from JSON.
    Returns list of dicts with keys: member_id, name, folder.
    """
    path = Path(members_path or DEFAULT_MEMBERS_PATH)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_members(members, members_path=None):
    """Save member list to JSON."""
    path = Path(members_path or DEFAULT_MEMBERS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(members, f, indent=2)


def get_member_folders(members_path=None):
    """
    Return list of (member_id, name, folder_path) where folder_path is absolute.
    """
    members = load_members(members_path)
    base = Path(members_path or DEFAULT_MEMBERS_PATH).resolve().parent
    result = []
    for m in members:
        folder = base / m["folder"]
        result.append((m["member_id"], m["name"], folder))
    return result

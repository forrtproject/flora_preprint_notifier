from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Set


def _default_blacklist_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "email_blacklist.txt"


def blacklist_path() -> Path:
    raw = os.environ.get("EMAIL_BLACKLIST_PATH")
    if raw and raw.strip():
        return Path(raw.strip())
    return _default_blacklist_path()


@lru_cache(maxsize=1)
def load_blacklist() -> Dict[str, Set[str]]:
    data: Dict[str, Set[str]] = {"emails": set(), "locals": set(), "domains": set()}
    path = blacklist_path()
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            kind = None
            value = txt
            if ":" in txt:
                pref, rest = txt.split(":", 1)
                pref_l = pref.strip().lower()
                if pref_l in {"email", "local", "domain"}:
                    kind = pref_l
                    value = rest.strip()
            val = value.lower().strip()
            if not val:
                continue
            if kind == "email" or (kind is None and "@" in val):
                data["emails"].add(val)
            elif kind == "domain":
                data["domains"].add(val)
            else:
                data["locals"].add(val)
    return data


def clear_blacklist_cache() -> None:
    load_blacklist.cache_clear()


def is_blacklisted_email(email: str) -> bool:
    if not email or "@" not in email:
        return False
    txt = email.lower().strip()
    local, _, domain = txt.rpartition("@")
    if not local or not domain:
        return False
    bl = load_blacklist()
    if txt in bl["emails"]:
        return True
    if local in bl["locals"]:
        return True
    if domain in bl["domains"]:
        return True
    return False


def add_blacklisted_email(email: str) -> None:
    txt = (email or "").lower().strip()
    if not txt or "@" not in txt:
        raise ValueError("Expected a full email address")
    path = blacklist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"\nemail:{txt}\n")
    clear_blacklist_cache()

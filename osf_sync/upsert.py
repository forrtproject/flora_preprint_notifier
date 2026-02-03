from __future__ import annotations
from typing import Iterable, Dict

from .dynamo.preprints_repo import PreprintsRepo


def upsert_batch(objs: Iterable[Dict]) -> int:
    rows = list(objs)
    if not rows:
        return 0
    repo = PreprintsRepo()
    return repo.upsert_preprints(rows)

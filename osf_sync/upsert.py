from __future__ import annotations
from typing import Iterable, Dict

from .dynamo.preprints_repo import PreprintsRepo
from .preprint_filters import get_min_original_publication_date, is_preprint_before_min_date


def upsert_batch(objs: Iterable[Dict]) -> int:
    rows = list(objs)
    if not rows:
        return 0
    repo = PreprintsRepo()
    min_date = get_min_original_publication_date()
    filtered = []
    for obj in rows:
        if is_preprint_before_min_date(obj, min_date):
            osf_id = obj.get("id")
            if osf_id:
                repo.delete_preprint_and_related(osf_id)
            continue
        filtered.append(obj)
    if not filtered:
        return 0
    return repo.upsert_preprints(filtered)

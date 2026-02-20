import math
import unittest
from decimal import Decimal

from osf_sync.dynamo.api_cache_repo import _to_dynamo_value


class ApiCacheRepoConversionTests(unittest.TestCase):
    def test_converts_nested_floats_to_decimals(self) -> None:
        payload = {
            "score": 0.75,
            "meta": {"weights": [0.1, 0.2], "ok": True},
            "vals": (1.5, "x"),
        }
        out = _to_dynamo_value(payload)
        self.assertEqual(out["score"], Decimal("0.75"))
        self.assertEqual(out["meta"]["weights"], [Decimal("0.1"), Decimal("0.2")])
        self.assertEqual(out["vals"], [Decimal("1.5"), "x"])

    def test_non_finite_floats_are_dropped_to_none(self) -> None:
        payload = {"a": math.nan, "b": math.inf, "c": -math.inf}
        out = _to_dynamo_value(payload)
        self.assertIsNone(out["a"])
        self.assertIsNone(out["b"])
        self.assertIsNone(out["c"])


if __name__ == "__main__":
    unittest.main()

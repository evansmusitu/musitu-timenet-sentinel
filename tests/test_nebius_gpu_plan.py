import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from scripts.nebius_gpu_plan import estimate_cost, read_inventory


class PlannerTests(unittest.TestCase):
    def test_bounded_estimates(self):
        self.assertEqual(estimate_cost(45), Decimal("1.85"))
        self.assertEqual(estimate_cost(60), Decimal("2.30"))
        with self.assertRaises(ValueError):
            estimate_cost(45, rate=Decimal("9.50"))
        self.assertEqual(estimate_cost(15, rate=Decimal("9.50")), Decimal("2.88"))

    def test_invalid_limits(self):
        for minutes in (0, -1, 61):
            with self.assertRaises(ValueError):
                estimate_cost(minutes)
        with self.assertRaises(ValueError):
            estimate_cost(30, 2)

    def test_inventory_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name, obj in {
                "platforms.json": {"items": [{"metadata": {"name": "gpu-h100"}}, {"metadata": {"name": "cpu-only"}}]},
                "subnets.json": {"items": []},
                "instances.json": {"items": []},
            }.items():
                (path / name).write_text(json.dumps(obj), encoding="utf-8")
            result = read_inventory(path)
            self.assertEqual(result["gpu_platforms_advertised"], ["gpu-h100"])
            self.assertFalse(result["subnet_configured"])
            self.assertFalse(result["ready_to_launch"])
            self.assertFalse(result["billing_hard_cap_configured"])
            self.assertEqual(result["advertised_gpu_45m_indicative_cost_usd"], {})
            (path / "platforms.json").write_text(json.dumps({"items": [{"metadata": {"name": "gpu-b300-sxm"}}]}),encoding="utf-8")
            pricy = read_inventory(path)
            self.assertEqual(pricy["advertised_gpu_45m_indicative_cost_usd"], {"gpu-b300-sxm": "7.63"})
            self.assertFalse(pricy["advertised_gpu_45m_below_planning_threshold"])


if __name__ == "__main__":
    unittest.main()

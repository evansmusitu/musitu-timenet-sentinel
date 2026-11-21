import json
import unittest
from scripts.nebius_region_catalog import items_of, summarize, summarize_platforms

class RegionTests(unittest.TestCase):
    def test_projects(self):
        p={"items":[{"metadata":{"id":"project-one", "name":"demo"},"status":{"region":"eu-south1"}}]}
        self.assertEqual(summarize(p),[{"id":"project-one","name":"demo","region":"eu-south1"}])
    def test_gpu_platforms(self):
        p={"items":[{"metadata":{"name":"cpu-d3"}},{"metadata":{"name":"gpu-rtx6000-a"}},{"metadata":{"name":"gpu-b300-sxm"}}]}
        self.assertEqual(summarize_platforms(p),["gpu-b300-sxm","gpu-rtx6000-a"])
    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            items_of({"items": [None]})
        with self.assertRaises(ValueError):
            items_of({"items": "broken"})

if __name__=="__main__":
    unittest.main()

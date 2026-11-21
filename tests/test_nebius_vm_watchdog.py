import unittest
from datetime import datetime, timedelta, timezone
from scripts.nebius_vm_watchdog import qualified, expired_instances

PROJECT="project-abc"
NOW=datetime(2026,11,22,12,tzinfo=timezone.utc)

def sample(*,age=60, name="musitu-timenet-gpu-experiment",
           state="RUNNING", labels=None, project=PROJECT, platform="gpu-rtx6000-a"):
    return {"metadata":{"id":"instance-xyz", "name":name,"parent_id":project,
            "created_at":(NOW-timedelta(minutes=age)).isoformat(),
            "labels":labels if labels is not None else {"managed-by":"musitu-timenet-sentinel",
                 "purpose":"opentime-hackathon","owner":"evansmusitu"}},
        "status":{"state":state},"spec":{"resources":{"platform":platform}}}

class WatchdogTests(unittest.TestCase):
    def test_expired_owned_vm(self):
        self.assertTrue(qualified(sample(),NOW,PROJECT))
    def test_no_expiry_yet(self):
        self.assertFalse(qualified(sample(age=20),NOW,PROJECT))
    def test_wrong_owner_or_name(self):
        self.assertFalse(qualified(sample(project="project-other"),NOW,PROJECT))
        self.assertFalse(qualified(sample(name="another-vm"),NOW,PROJECT))
        self.assertFalse(qualified(sample(labels={"owner":"evansmusitu"}),NOW,PROJECT))
    def test_wrong_state_or_platform(self):
        self.assertFalse(qualified(sample(state="STOPPED"),NOW,PROJECT))
        self.assertFalse(qualified(sample(platform="cpu-d3"),NOW,PROJECT))
        self.assertFalse(qualified(sample(state="ERROR"),NOW,PROJECT))
    def test_fail_closed_missing_metadata(self):
        self.assertFalse(qualified({"metadata":{"id":"instance-x"}},NOW,PROJECT))
        self.assertEqual(expired_instances({"items":[]},NOW,PROJECT),[])
        self.assertEqual(len(expired_instances({"items":[sample(age=60),sample(age=1)]},NOW,PROJECT)),1)

if __name__=="__main__":
    unittest.main()

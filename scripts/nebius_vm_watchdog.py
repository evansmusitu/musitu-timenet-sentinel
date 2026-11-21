#!/usr/bin/env python3
"""Fail-closed watchdog for MUSITU TimeNet test VMs.

Default is AUDIT. The only permitted cloud mutation in STOP mode is a Nebius
instance stop for an expired, strictly tagged, named, project-scoped VM.
Never creates/deletes a VM, disk or other resource.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TARGET_REGIONS = frozenset({"eu-south1", "uk-south2", "us-central1"})
NAME_PREFIX = "musitu-timenet-gpu-"
EXPECTED_LABELS = {
    "managed-by": "musitu-timenet-sentinel",
    "purpose": "opentime-hackathon",
    "owner": "evansmusitu",
}
EXPIRY_MINUTES = 45
MAX_STOP_ACTIONS = 1


def _parse_time(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _items(document):
    items = document if isinstance(document, list) else document.get("items", []) if isinstance(document, dict) else None
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise ValueError("Invalid instance listing")
    return items


def qualified(item, now, expected_project, min_age=EXPIRY_MINUTES):
    """Return True for an old MUSITU-owned running instance, else False."""
    meta = item.get("metadata") or {}
    labels = meta.get("labels") or {}
    spec = item.get("spec") or {}
    status = item.get("status") or {}
    iid = str(meta.get("id") or "")
    name = str(meta.get("name") or "")
    parent = str(meta.get("parent_id") or "")
    created = _parse_time(meta.get("created_at") or item.get("created_at"))
    state = str(status.get("state") or "").upper()
    if not re.fullmatch(r"instance-[a-z0-9-]+", iid):
        return False
    if parent != expected_project or not expected_project.startswith("project-"):
        return False
    if not name.startswith(NAME_PREFIX) or len(name) > 100:
        return False
    if not isinstance(labels, dict) or any(labels.get(k) != v for k,v in EXPECTED_LABELS.items()):
        return False
    if state not in {"RUNNING", "STARTING"}:
        return False
    if not created or created > now:
        return False
    age = (now - created).total_seconds()/60
    if age < min_age:
        return False
    # Fail closed for any request with a missing or unexpected platform.
    platform = str((spec.get("resources") or {}).get("platform") or "")
    if not platform.startswith("gpu-"):
        return False
    return True


def expired_instances(document, now, expected_project):
    return [x for x in _items(document) if qualified(x, now, expected_project)]


def run_checked(args, profile, *, expect_json=True):
    cmd = ["nebius"] + args + ["--profile", profile]
    if expect_json:
        cmd += ["--format", "json"]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=75)
    if p.returncode:
        raise RuntimeError("Nebius request failed; no further mutation permitted")
    return json.loads(p.stdout) if expect_json else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--projects",type=Path,required=True)
    parser.add_argument("--profile",default="musitu-watchdog")
    parser.add_argument("--mode",choices=("audit","stop-expired"),default="audit")
    args=parser.parse_args()
    if args.mode == "stop-expired" and __import__("os").environ.get("MUSITU_WATCHDOG_STOP_ALLOWED") != "true":
        raise SystemExit("Stopping disabled: explicit repository safeguard is not enabled")

    from scripts.nebius_region_catalog import summarize
    projects=summarize(json.loads(args.projects.read_text(encoding="utf-8")))
    now=datetime.now(timezone.utc)
    total=0
    eligible=[]
    for project in projects:
        pid=project["id"]
        if project["region"] not in TARGET_REGIONS or not pid.startswith("project-"):
            continue
        try:
            report=run_checked(["compute","instance","list","--parent-id",pid,"--all"],args.profile)
        except RuntimeError:
            print("Instance inventory unavailable for region",project["region"],"(no action)")
            continue
        total += len(_items(report))
        for instance in expired_instances(report,now,pid):
            eligible.append((project["region"],pid,instance["metadata"]["id"]))

    print("Checked RTX regions:",",".join(sorted(TARGET_REGIONS)))
    print("Instances inspected:",total,"; expired, tagged MUSITU GPU instances:",len(eligible))
    print("Watchdog mode:",args.mode)
    if args.mode == "audit":
        print("No VM modifications performed.")
        return
    if len(eligible)>MAX_STOP_ACTIONS:
        raise SystemExit("More than one eligible instance. Refusing bulk stopping.")
    for region,pid,iid in eligible:
        # Double-check live identity, ownership and age directly before stopping.
        item=run_checked(["compute","instance","get","--id",iid],args.profile)
        if not qualified(item,datetime.now(timezone.utc),pid):
            raise SystemExit("Instance failed final identity verification; stopping aborted")
        print("Stopping expired MUSITU GPU in region",region,"via Nebius API...")
        run_checked(["compute","instance","stop","--id",iid],args.profile,expect_json=False)
        verification=run_checked(["compute","instance","get","--id",iid],args.profile)
        status=str((verification.get("status") or {}).get("state") or "").upper()
        if status not in {"STOPPED", "STOPPING"}:
            raise SystemExit("Stop was requested but state remains unverified; investigate immediately")
        print("Stop response verified: state",status)
    print("Watchdog completed. Stopping a VM does not remove separately billed disks.")


if __name__=="__main__":
    main()

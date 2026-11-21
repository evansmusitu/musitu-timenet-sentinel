# MUSITU TimeNet: GPU preflight — 2026-11-22

## Verified live results
The following GitHub Actions run completed successfully, with **no cloud write operations**:

https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44958783291

- Nebius IAM service-account credentials: **PASS** using existing encrypted Actions secrets.
- Canonical project parent: `project-e04p7wefbc00y3c7de0pcn`. Note the existing `NEBIUS_PROJECT_ID` secret contains the console's short ID without the `project-` prefix; the preflight normalizes and checks it against the exact expected project. The smoke test had not validated project-scoped calls.
- Project-scoped compute platforms: **2**. Advertised GPU platform: **`gpu-b300-sxm`**.
- Configured VPC subnet exists: **YES**.
- Existing VMs: **0**.
- RTX PRO 6000 96 GB GPU: **NOT advertised in the existing eu-west2 project**. Do not use its lower hourly rate to estimate running in this project.
- B300 GPU quota, real-time available capacity, usable preset, GPU image, actual per-account pricing and total balance/expiry: **NOT yet verified from the authenticated API**.
- No new VM, disk, public IP, or cloud resource was created.

## Cost plan — not a billing guarantee
Published Nebius pricing lists RTX PRO 6000 at approximately $1.80/GPU-hour and B300 between $7.85/GPU-hour in some tables and $9.50/GPU-hour effective October 1, 2026 in another view. To avoid underbudgeting, the cost planner uses **$9.50/GPU-hour** for the advertised B300 platform until actual account pricing is confirmed.

At $9.50/GPU-hour, the B300 45-minute estimate is $7.13 for GPU time, plus $0.50 illustrative headroom = **$7.63**. This exceeds the **$3.00** per-run planning threshold.

A short 15-minute B300 investigation estimates $2.88 including the same illustrative headroom, but capacity could still be zero and a 15-minute window is unlikely to complete a full 5B checkpoint adaptation. *Do not provision simply to use the credit.*

The previously shown RTX PRO 6000 45-minute estimate of $1.85 is a **reference only**, not a rate applicable to this project.

Storage, disk retention, spot variability, taxes, minimum billable increments, and other services can increase actual costs. Neither a GitHub job timeout nor a calculated estimate provides a genuine Nebius billing hard cap.

## Prerequisites to launch any paid work
1. Verify Nebius promotional balance source, remaining credit, expiry and whether billing can continue after credits are exhausted. Previous web-console view showed $100 but source and expiration were not verified.
2. Verify actual available GPU quota and presets in the target region/project. The prior UI inspection reported B300 zero quota; the current CLI preflight establishes **platform advertisement only**.
3. Prefer a cheaper GPU if a separate tenant project in an appropriate region genuinely offers `gpu-rtx6000-a` 1-GPU 96 GB configuration, and confirm project-scoped service-account permissions, network, image and boot disk there. No new project has yet been created.
4. Do not let a guest OS shutdown stand in for a Nebius API shutdown. Nebius warns a recovery policy can restart guest-shutdown instances. Choose `FAIL` recovery policy, enforce a short runtime, and validate a separate, independently triggered stop/delete watchdog and retained-disk cleanup **before** enabling provisioning.
5. Do not rely solely on GitHub Actions `timeout-minutes` or `finally` for terminating a remote VM; cancellation and runner outages can skip cleanup. A separate cloud-side lifecycle control or external independent watchdog is required.
6. Save training artifacts outside the ephemeral GPU VM and verify license/compatibility of `opentime/tslm-nov26`. The existing `sentinel/opentime_train.py` is **untested on GPU**, not a completed hackathon submission result.
7. Require explicit approval for the actual GPU hourly price and exact total budget before starting any billable resource.

## Current run gate
**FAIL CLOSED — no Nebius GPU deployment or paid workload is enabled.**

Code:
- [Preflight workflow](../.github/workflows/nebius-gpu-preflight.yml)
- [Cost planner](../scripts/nebius_gpu_plan.py)
- [Unit tests](../tests/test_nebius_gpu_plan.py)

References:
- https://nebius.com/prices
- https://docs.nebius.com/cli/reference/compute/instance/create
- https://docs.nebius.com/cli/reference/compute/instance/stop
- https://docs.nebius.com/cli/reference/compute/instance/delete

# MUSITU TimeNet — RTX GPU readiness and watchdog status (22 November 2026)

## Validated, without purchasing GPU time

**GitHub Actions Nebius service-account authentication** has been confirmed against a read-only Nebius IAM request.

The [nine-project discovery](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44960550371) found nine accessible Nebius projects, including the lower-cost GPU candidates below. The [single-GPU preset inspection](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44960983182) confirmed:

| Region | Nebius platform | GPU VRAM | Single-GPU preset | Project subnets |
| --- | --- | --- | --- | --- |
| eu-south1 | gpu-rtx6000-a | 96 GiB | 1gpu-24vcpu-218gb | 1 |
| uk-south2 | gpu-rtx6000-a | 96 GiB | 1gpu-24vcpu-218gb | 1 |
| us-central1 | gpu-rtx6000 | 96 GiB | 1gpu-24vcpu-218gb | 1 |

**Limitations:** Each region returned a quota record called `compute.instance.gpu.rtx6000`, but the read-only CLI response exposed no explicit positive `spec.limit` or active status. This is **not** proof of quota, allocation, a valid GPU reservation, or capacity. The VM inventory was independently read by the watchdog and found zero instances. More explicit quota evidence and a project-specific account price are required before provisioning.

Nebius's public **on-demand** rate for an RTX PRO 6000 is approximately **$1.80 per GPU-hour** (see [Nebius pricing](https://nebius.com/prices)); spot prices can change over time and should not be treated as a budget guarantee. A 45-minute GPU-only run at this on-demand reference would be $1.35, with our planning headroom bringing the illustrative figure to $1.85; actual costs may include storage, network and taxes.

## Independent watchdog

- [Watchdog workflow](../.github/workflows/nebius-vm-watchdog.yml) is scheduled **hourly while no billable GPU is running** and can be manually invoked.
- [Watchdog test suite](../tests/test_nebius_vm_watchdog.py) checks identity matching, instance age, state, project and GPU platform.
- It checks only three listed RTX regions, and only VMs with the expected `musitu-timenet-gpu-` prefix and exact `managed-by`, `purpose`, and `owner` labels.
- [Latest validated audit run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44960911612) found the 3 target projects, read all 3 instance inventories, and found **zero** owned expired instances.
- **Current watchdog mode: AUDIT ONLY.** Stop actions are blocked unless GitHub repository variable `NEBIUS_WATCHDOG_ENABLE_STOP=true` is explicitly set and a matching VM exceeds 45 minutes.
- Before any paid launch, tighten scheduled polling to every 5 minutes, verify stop privileges with a nonproduction controlled VM, add a separate independent cloud-side lifecycle control or emergency shutdown procedure, and verify the real VM's tags and timestamps.
- GitHub scheduling can be delayed or interrupted. A GitHub timeout or cron schedule is **not** a guaranteed Nebius spending cap or standalone cleanup guarantee.
- This watchdog can stop only a running GPU VM; it does **not** delete disks or prevent storage charges. It does not initiate a VM or an experiment.

Official Nebius commands include [instance stop](https://docs.nebius.com/cli/reference/compute/instance/stop) and [instance get](https://docs.nebius.com/cli/reference/compute/instance/get).

## Unclosed gates

1. Get confirmed positive RTX quota and live capacity in one eligible project, preferably `eu-south1` or `uk-south2`.
2. Verify remaining $100 credit origin/expiration and that consumption cannot exceed the intended cash-free budget.
3. Validate the exact GPU image, dependencies, disk/mount configuration and connectivity to the checkpoint/download source.
4. Check the `opentime/tslm-nov26` license and actual GPU compatibility. The training runner has **not** yet been executed on Nebius.
5. Prove reliable stop, failure handling, artifacts and separately triggered emergency recovery.
6. Require a specific approved maximum GPU spend and start window before a billable launch.

**Current launch gate: NOT APPROVED; zero GPU VMs created.**

# Nebius compute authentication decision — 2026-11-21

## Verified

- GitHub Actions issues an OIDC identity token in this repository's **main-branch** manually triggered workflow. A read-only preflight **passed**: https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44936558257
- The GitHub `sub` is customized rather than the default `repo:OWNER/REPO:ref:refs/heads/main`; a future Nebius trust configuration must bind the actual validated `sub`, not assume the default.
- The existing Nebius service account is `musitu-timenet-github`, project `default-project-eu-west2`. Its IAM Authorized keys have not been configured. The two GitHub Actions secrets for project ID and service-account ID were previously confirmed present; the private-key and authorized-key-ID secrets were absent.
- A Nebius console read-only check showed a **$100.00** balance and **$0.00** consumption for the stated period; the voucher source and expiration date were not established. No GPU or other billable resource was created.
- Nebius and NVIDIA specify **96 GB** of GDDR7 VRAM for the NVIDIA RTX PRO 6000 Blackwell. The previously described 24 GB value was incorrect (24 refers to vCPUs for a common single-GPU instance).
- Public on-demand pricing for RTX PRO 6000 is approximately **$1.80 per GPU-hour** (verify real-time console pricing, region, disk/network charges and spot variability prior to use). Public material lists availability in regions such as `eu-south1`, but the project is `eu-west2`: direct in-project GPU availability remains unverified.

## Nebius workload identity federation caveat

The official [Nebius FederatedCredentialsService protocol](https://github.com/nebius/api/blob/main/nebius/iam/v1/federated_credentials_service.proto) identifies inbound OIDC Federated Credentials as **PUBLIC PREVIEW**. It explicitly says custom external credentials providers are for early adopters, with local JWK set storage also available. However, the inspected tenant console showed no inbound GitHub OIDC trust-setting UI. Its tenant-level `Federations` section is **SAML user SSO**, not service-account OIDC.

Thus GitHub OIDC preflight passing **does NOT establish Nebius authentication**. Do not use OIDC in a privileged workflow before completing and verifying Nebius-side subject/issuer authorization and repository-branch guardrails.

## Authentication choices

1. **Preferred if tenant supports it:** inbound GitHub Actions OIDC using an exact issuer `https://token.actions.githubusercontent.com`, exact validated custom subject, controlled audience, and a narrowly scoped service account trust. Request preview enablement/support confirmation as necessary, and validate a read-only Nebius call before considering compute.
2. **Documented fallback:** add an RSA authorized public key to the Nebius service account, put its matching private key directly into an encrypted GitHub Actions Secret, and add the authorized key ID. The bootstrap script is owner-guarded and is an **explicit manual task**; it must not automatically run on folder open. Never commit, print, or upload the private key to chat.
3. **Direct console:** the owner account successfully signed in via Google SSO in TinyFish browser and can inspect Nebius GUI or Echo without a GitHub IAM credential. This is not unattended GitHub control; browser sessions and Cloudflare challenges may interrupt.

## Cost and safety

Do **not** provision GPU/VMs while the credit source, region availability, automatic cleanup, and potential overage controls remain unverified. Keep the existing public CPU/temporal benchmark and the unrun experimental OpenTime-5B GPU runner clearly distinguished. A successful GitHub identity preflight is not evidence of having fine-tuned or evaluated the 5B checkpoint.

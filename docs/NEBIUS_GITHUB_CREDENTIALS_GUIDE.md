# MUSITU TimeNet — GitHub Actions ↔ Nebius credentials

## Reason for this approach
GitHub Actions will automate time-limited Nebius GPU work. GitHub authentication is not itself cheaper than using the Nebius console; savings come from careful runtimes, explicit start/stop, and avoiding idle GPUs. GitHub Actions and Codespaces can also have usage charges, depending on account limits.

## Verified account IDs
- Repository: `evansmusitu/musitu-timenet-sentinel`.
- Nebius project: `e04p7wefbc00y3c7de0pcn`.
- Project-scoped service account: `serviceaccount-e04afk1xvsf8pwvd58`.
- Nebius billing showed a $100 credit balance, but its origin and expiration were not verified.
- Earlier GitHub Actions secrets audit: `NEBIUS_PROJECT_ID` and `NEBIUS_SERVICE_ACCOUNT_ID` exist. **Do not assume** `NEBIUS_PRIVATE_KEY` and `NEBIUS_PUBLIC_KEY_ID` exist: they were missing at last verified check.

## One GitHub Codespaces command (owner only)
Open the existing [Codespace improved disco](https://github.com/codespaces/improved-disco-gxq9gw45vqvp29gwr) in the authenticated owner account, open the terminal and run:

```bash
git pull --ff-only && bash scripts/bootstrap-nebius-status.sh
```

This performs an owner-login check, generates a private RSA key **outside** the public repo, attempts to encrypt/store it using `gh secret set`, and commits **only** the public PEM at `nebius/musitu-timenet-github-public.pem`. If `gh secret set` is unauthorized, the script stops without committing the public key; it writes nonsecret diagnostic flags to `.setup-status/nebius-bootstrap.txt`. If Codespaces GitHub authorization lacks secrets write permission, it must be explicitly granted by the owner. Do not send or print the private key to anyone.

## Register the public key in Nebius
1. Download the PUBLIC key from `nebius/musitu-timenet-github-public.pem`.
2. Open [Nebius Cloud console](https://console.nebius.com/) → Administration → IAM → Service accounts → `musitu-timenet-github` → Authorized keys → Upload authorized key.
3. Upload the PEM public file. You may set an expiry date.
4. Copy the new authorized key ID (prefix `publickey-`) and create `NEBIUS_PUBLIC_KEY_ID` as an encrypted GitHub Actions secret. **Never upload or commit `private.pem`.**

## Verify without buying compute
From GitHub Actions, trigger **Nebius service-account auth smoke test (read only)** on main. That workflow is gated to repo `evansmusitu/musitu-timenet-sentinel` and actor `evansmusitu`, uses no GPU create actions, and checks a Nebius read-only IAM operation. It will fail safely if any required secret is missing.

## Spending gate
Do not enable any paid GPU workflow until Nebius API auth works, the actual GPU shape is available in the chosen project/region, its pricing and credit terms are verified, and a stop/cleanup plan with a hard runtime limit is in place. Nebius RTX PRO 6000 Blackwell has 96 GB VRAM; actual pricing/availability must be checked at launch.

Official Nebius documentation:
- https://docs.nebius.com/iam/service-accounts/authorized-keys
- https://docs.nebius.com/cli/configure
- https://docs.nebius.com/cli/install
Official GitHub:
- https://cli.github.com/manual/gh_secret_set

# Nebius IAM → GitHub Actions: secure bootstrap

## Current status
- The setup script is committed at `scripts/bootstrap-nebius-github.sh`.
- No private key is committed. The repository ignores `*.pem` except the intended public key and ignores `*.key`.
- The `NEBIUS_PRIVATE_KEY` secret has **not** yet been confirmed stored.
- The Nebius IAM authorized-key upload and public key ID registration have **not** been confirmed.
- No Nebius GPU resources are created by these steps.

## One command in a GitHub Codespace

Open [New Codespace for this repository](https://github.com/codespaces/new?repo=1465909261), choose `main`, and launch the terminal. Run:

```bash
bash scripts/bootstrap-nebius-github.sh
```

The script uses OpenSSL to create a private RSA key **outside** the Git repository, uploads it to encrypted GitHub Actions Secrets through the signed-in `gh` CLI, and commits **only the public key** at `nebius/musitu-timenet-github-public.pem`. It does not print key material. Do not run the command in GitHub Actions: the default `GITHUB_TOKEN` may not have repository-secrets write permission. It is intended for an authenticated GitHub Codespace. If GitHub requests authorization, authorize the minimally necessary repository Secrets access and retry. Avoid copying the private key into chat, workflows, commits, issues, or public Actions logs.

## Finish Nebius IAM (requires Nebius sign-in)

1. Download the `nebius/musitu-timenet-github-public.pem` file from this repository.
2. In Nebius console, open IAM → service account `musitu-timenet-github` → Authorized keys → Upload. Upload only the public PEM.
3. Copy the generated `publickey-...` **ID** (identifier, not a secret).
4. Add repository Secrets at [GitHub Actions Secrets](https://github.com/evansmusitu/musitu-timenet-sentinel/settings/secrets/actions): `NEBIUS_PUBLIC_KEY_ID`, `NEBIUS_SERVICE_ACCOUNT_ID`, `NEBIUS_PROJECT_ID`. The private key should already have been saved by the script as `NEBIUS_PRIVATE_KEY`.
5. Run the manual `Verify Nebius service-account key (no cloud access)` action, which validates that the stored private key corresponds to the public PEM without logging the secret.
6. Separately verify Nebius authenticated CLI / API access. The manual key check only proves key-pair consistency and does not prove the Nebius authorization is active.

## Spend and security rules

This is a PUBLIC GitHub repository. Before provisioning any compute, restrict GitHub workflows to a protected environment and verify the service account's **project-scoped** IAM permissions. Keep spending within available Nebius voucher credits; enable billing alerts and hard-stop/cleanup for all GPU instances. No automatically triggered workflow should create paid resources. Revoke the corresponding Nebius authorized key and delete GitHub Secrets when no longer needed.

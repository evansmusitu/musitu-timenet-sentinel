#!/usr/bin/env bash
# Bootstrap Nebius service-account authorization without printing private keys.
# Execute inside an authenticated GitHub Codespace for this repository.
set -Eeuo pipefail
set +x
umask 077

repo="evansmusitu/musitu-timenet-sentinel"
case "$PWD" in
  */musitu-timenet-sentinel) ;;
  *) echo "ERROR: open a Codespace at the musitu-timenet-sentinel repository." >&2; exit 2 ;;
esac
for command in openssl gh git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: missing $command." >&2; exit 2
  fi
done
if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI is not authenticated. Sign in to GitHub Codespaces." >&2
  exit 2
fi

# Persist locally outside the repo; never commit this private key.
secure_dir="${HOME}/.local/share/musitu-nebius"
mkdir -p "$secure_dir"
chmod 700 "$secure_dir"
private="${secure_dir}/private.pem"
public="${secure_dir}/public.pem"
if [[ ! -f "$private" ]]; then
  echo "Generating 4096-bit RSA service-account keypair securely..."
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 -out "$private" >/dev/null 2>&1
fi
chmod 600 "$private"
openssl pkey -in "$private" -pubout -out "$public" 2>/dev/null
chmod 644 "$public"
openssl pkey -pubin -in "$public" -noout >/dev/null 2>&1

# Transmit directly to GitHub's encrypted Actions Secrets API.
# Never print or store private-key material in a GitHub tracked path or log.
echo "Saving encrypted NEBIUS_PRIVATE_KEY repository secret..."
if ! gh secret set NEBIUS_PRIVATE_KEY --repo "$repo" < "$private"; then
  echo "ERROR: GitHub CLI lacks Actions Secrets write permission. Key remains private in Codespace." >&2
  exit 3
fi

mkdir -p nebius
cp "$public" nebius/musitu-timenet-github-public.pem
git add -- nebius/musitu-timenet-github-public.pem
if ! git diff --cached --quiet; then
  git -c user.name="MUSITU Sentinel Setup" -c user.email="actions@users.noreply.github.com" \
    commit -m "Publish only Nebius service account public key"
  git push origin HEAD:main
fi

echo "SUCCESS: private key saved to Actions Secrets; only public key committed."
echo "PUBLIC KEY URL: https://github.com/$repo/blob/main/nebius/musitu-timenet-github-public.pem"
echo "NEXT: download public PEM, upload it under Nebius service account > Authorized keys."
echo "Then configure NEBIUS_SERVICE_ACCOUNT_ID, NEBIUS_PUBLIC_KEY_ID, NEBIUS_PROJECT_ID via GitHub Actions secrets."

#!/usr/bin/env bash
# Status-only wrapper. Never logs or reads the value of a private key.
set -u
set +x
umask 077
mkdir -p .setup-status

secrets_api="unknown"
if command -v gh >/dev/null 2>&1; then
  if gh api --silent "repos/evansmusitu/musitu-timenet-sentinel/actions/secrets/public-key" >/dev/null 2>&1; then
    secrets_api="accessible"
  else
    secrets_api="unavailable"
  fi
else
  secrets_api="gh-missing"
fi

if bash scripts/bootstrap-nebius-github.sh >/dev/null 2>&1; then
  result=success
  code=0
else
  code=$?
  result=failed
fi
public_file="missing"
if [[ -f nebius/musitu-timenet-github-public.pem ]]; then
  public_file="present"
fi
private_file="missing"
if [[ -f "${HOME}/.local/share/musitu-nebius/private.pem" ]]; then
  private_file="present"
fi
printf 'result=%s\nexit_code=%s\nsecret_api=%s\nprivate_key_file=%s\npublic_key_file=%s\n' \
  "$result" "$code" "$secrets_api" "$private_file" "$public_file" > .setup-status/nebius-bootstrap.txt
echo "Nebius setup result: $result (exit code $code)"
echo "Diagnostic: .setup-status/nebius-bootstrap.txt (status flags only, no credentials)"
exit "$code"

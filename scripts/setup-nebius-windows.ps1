#requires -Version 5.1
<#
MUSITU Sentinel Nebius GitHub Actions credential bootstrap (Windows).
Only the public key is uploaded to Nebius; private key is stored locally
and transmitted directly to encrypted GitHub Actions Secrets over stdin.
No GPU resources are created.
#>
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repo = 'evansmusitu/musitu-timenet-sentinel'
$expectedLogin = 'evansmusitu'
$saId = 'serviceaccount-e04afk1xvsf8pwvd58'
$projectId = 'e04p7wefbc00y3c7de0pcn'
$keyDir = Join-Path $env:LOCALAPPDATA 'MUSITU\NebiusTimeNet'
$privatePem = Join-Path $keyDir 'private.pem'
$publicPem = Join-Path $keyDir 'public.pem'

function Get-CommandLocation([string]$name, [string[]]$fallbacks) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($f in $fallbacks) {
        if ($f -and (Test-Path -LiteralPath $f)) { return $f }
    }
    return $null
}

function Install-Tool([string]$packageId) {
    $winget = Get-CommandLocation 'winget.exe' @()
    if (-not $winget) {
        throw "Install $packageId using Windows Package Manager (winget), then run this same script again."
    }
    Write-Host "Installing $packageId through Windows Package Manager..."
    & $winget install --id $packageId --exact --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Installation failed for $packageId (code $LASTEXITCODE)." }
}

# PowerShell 7+ exposes the built-in RSA PEM export routines. No OpenSSL needed.
if ($PSVersionTable.PSVersion.Major -lt 7) {
    $pwsh = Get-CommandLocation 'pwsh.exe' @(
        (Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe')
    )
    if (-not $pwsh) {
        Install-Tool 'Microsoft.PowerShell'
        $pwsh = Get-CommandLocation 'pwsh.exe' @(
            (Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe')
        )
    }
    if (-not $pwsh) { throw 'PowerShell 7 was installed but not found. Open PowerShell 7 and rerun the command.' }
    Write-Host 'Continuing in PowerShell 7...'
    & $pwsh -NoLogo -NoProfile -File $PSCommandPath
    exit $LASTEXITCODE
}

$gh = Get-CommandLocation 'gh.exe' @(
    (Join-Path $env:ProgramFiles 'GitHub CLI\gh.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'GitHub CLI\gh.exe')
)
if (-not $gh) {
    Install-Tool 'GitHub.cli'
    $gh = Get-CommandLocation 'gh.exe' @(
        (Join-Path $env:ProgramFiles 'GitHub CLI\gh.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'GitHub CLI\gh.exe')
    )
}
if (-not $gh) { throw 'GitHub CLI was installed but not found. Reopen PowerShell 7 and rerun this script.' }

Write-Host 'Checking GitHub authentication...'
& $gh auth status 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'GitHub will show a browser-based login; complete it with your evansmusitu account.'
    & $gh auth login --hostname github.com --git-protocol https --web --scopes repo
    if ($LASTEXITCODE -ne 0) { throw 'GitHub authentication was not completed.' }
}
$actualLogin = (& $gh api user --jq '.login' 2>$null | Select-Object -First 1)
if ($LASTEXITCODE -ne 0 -or $actualLogin.Trim() -ne $expectedLogin) {
    throw "Authenticated GitHub user must be $expectedLogin. Switch accounts with 'gh auth login'."
}
Write-Host "GitHub owner confirmed: $expectedLogin"

New-Item -ItemType Directory -Force -Path $keyDir | Out-Null
# Restrict local key folder access to the current Windows user.
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls.exe $keyDir '/inheritance:r' '/grant:r' ("${identity}:(OI)(CI)F") 1>$null
if ($LASTEXITCODE -ne 0) { throw 'Unable to restrict permissions on the local key directory.' }

function To-Pem([string]$kind, [byte[]]$bytes) {
    $b64 = [Convert]::ToBase64String($bytes, [Base64FormattingOptions]::InsertLineBreaks)
    return "-----BEGIN $kind-----`n$b64`n-----END $kind-----`n"
}
if ((Test-Path -LiteralPath $privatePem) -xor (Test-Path -LiteralPath $publicPem)) {
    throw 'Only one file from the keypair exists. Stop: avoid accidentally generating mismatched keys.'
}
if (-not (Test-Path -LiteralPath $privatePem)) {
    Write-Host 'Generating a 4096-bit RSA key locally (never printed or committed)...'
    $rsa = [System.Security.Cryptography.RSA]::Create(4096)
    try {
        $privateText = To-Pem 'PRIVATE KEY' ($rsa.ExportPkcs8PrivateKey())
        $publicText  = To-Pem 'PUBLIC KEY' ($rsa.ExportSubjectPublicKeyInfo())
        $utf8 = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText($privatePem, $privateText, $utf8)
        [System.IO.File]::WriteAllText($publicPem, $publicText, $utf8)
        $privateText = $null
        $publicText = $null
    } finally { $rsa.Dispose() }
} else {
    Write-Host 'Reusing your existing local RSA keypair (no rotation).'
}
# Never pass the private key via a process argument or log; GH reads it from stdin.
Write-Host 'Uploading encrypted NEBIUS_PRIVATE_KEY to GitHub Actions Secrets...'
Get-Content -LiteralPath $privatePem -Raw -Encoding UTF8 |
    & $gh secret set NEBIUS_PRIVATE_KEY --repo $repo --app actions
if ($LASTEXITCODE -ne 0) {
    throw 'GitHub Secrets upload failed. Private key remains only in your local protected folder.'
}
& $gh secret set NEBIUS_SERVICE_ACCOUNT_ID --repo $repo --app actions --body $saId
if ($LASTEXITCODE -ne 0) { throw 'Could not save service account ID.' }
& $gh secret set NEBIUS_PROJECT_ID --repo $repo --app actions --body $projectId
if ($LASTEXITCODE -ne 0) { throw 'Could not save Nebius project ID.' }

Write-Host ''
Write-Host 'The GitHub private-key secret is saved. Now register its PUBLIC key in Nebius.'
Write-Host "Public key file: $publicPem"
Write-Host 'Nebius: Administration > IAM > Service accounts > musitu-timenet-github > Authorized keys > Upload.'
Start-Process explorer.exe -ArgumentList ('/select,"' + $publicPem + '"')
Start-Process 'https://console.nebius.com/'
Write-Host 'Upload public.pem (NOT private.pem); copy the new authorized key ID.'
$keyId = Read-Host 'Paste only the new Nebius authorized-key ID (publickey-...) here'
$keyId = $keyId.Trim()
if ($keyId -notmatch '^publickey-[A-Za-z0-9-]+$') {
    Write-Host 'Skipping key ID: it was not a publickey- identifier. Existing private-key secret stays saved.'
    Write-Host 'Rerun this script when the public key is registered; it reuses the existing keypair.'
    exit 2
}
& $gh secret set NEBIUS_PUBLIC_KEY_ID --repo $repo --app actions --body $keyId
if ($LASTEXITCODE -ne 0) { throw 'Failed to save the authorized-key ID.' }

Write-Host ''
Write-Host 'Checking presence of four GitHub Actions secrets (names only)...'
$names = @(& $gh secret list --repo $repo --app actions --json name --jq '.[].name')
if ($LASTEXITCODE -ne 0) { throw 'Unable to list GitHub secret names.' }
foreach ($name in @('NEBIUS_PRIVATE_KEY','NEBIUS_PUBLIC_KEY_ID','NEBIUS_SERVICE_ACCOUNT_ID','NEBIUS_PROJECT_ID')) {
    if ($names -notcontains $name) { throw "Missing required secret: $name" }
    Write-Host "$name : PRESENT"
}
Write-Host ''
Write-Host 'All four encrypted GitHub secrets are configured.'
Write-Host 'Running the manual, read-only Nebius auth smoke test (no GPUs or spend)...'
& $gh workflow run 'nebius-auth-smoke.yml' --repo $repo --ref main
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Secret setup succeeded but GitHub workflow dispatch failed; use the link below to start the test.'
}
Write-Host "Check the result: https://github.com/$repo/actions/workflows/nebius-auth-smoke.yml"
Write-Host 'Do not delete the protected local key files until authentication is verified and a recovery plan exists.'

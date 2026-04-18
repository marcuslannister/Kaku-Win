$env:TERM_PROGRAM = "Kaku"

$kakuUserProfile = Join-Path $env:APPDATA "kaku\powershell\profile.ps1"
if (Test-Path $kakuUserProfile) {
    . $kakuUserProfile
}

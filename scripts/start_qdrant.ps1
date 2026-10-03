$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dockerUserBin = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin'
$dockerSystemBin = 'C:\Program Files\Docker\Docker\resources\bin'
foreach ($dockerBin in @($dockerUserBin, $dockerSystemBin)) {
    if (Test-Path -LiteralPath (Join-Path $dockerBin 'docker.exe')) {
        $env:PATH = "$dockerBin;$env:PATH"
        break
    }
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop is not installed. Install it and start its WSL 2 engine first.'
}
Push-Location $projectRoot
try {
    docker compose up -d qdrant
    if ($LASTEXITCODE -ne 0) { throw 'Qdrant startup failed. Check that Docker Desktop is running.' }
} finally {
    Pop-Location
}

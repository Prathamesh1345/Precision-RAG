param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:HF_HUB_OFFLINE = '1'
$env:PRAG_API_URL = 'http://127.0.0.1:8000'
$env:PRAG_DENSE_DEVICE = 'cpu'
$env:PRAG_RERANK_DEVICE = if ($Cpu) { 'cpu' } else { 'cuda' }
$env:PRAG_READ_ONLY = '1'
$apiPython = if ($Cpu) { Join-Path $projectRoot '.venv\Scripts\python.exe' } else { Join-Path $projectRoot '.venv-gpu\Scripts\python.exe' }
$uiPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'tmp') -Force | Out-Null
foreach ($port in 8000,8501) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use; the demo may already be running."
    }
}
$apiProcess = Start-Process -FilePath $apiPython -ArgumentList '-m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput 'tmp\api.stdout.log' -RedirectStandardError 'tmp\api.stderr.log' -PassThru
$uiProcess = Start-Process -FilePath $uiPython -ArgumentList '-m streamlit run "Work Done till now/ui/streamlit_app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput 'tmp\ui.stdout.log' -RedirectStandardError 'tmp\ui.stderr.log' -PassThru
@{api_launcher_pid=$apiProcess.Id; ui_launcher_pid=$uiProcess.Id; dense_device=$env:PRAG_DENSE_DEVICE; rerank_device=$env:PRAG_RERANK_DEVICE} | ConvertTo-Json | Set-Content -LiteralPath 'tmp\demo_processes.json'
Write-Output 'Demo starting: http://127.0.0.1:8501 | API: http://127.0.0.1:8000/docs'

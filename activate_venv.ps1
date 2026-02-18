# activate_venv.ps1
Write-Host "Meeting Tools - Activate Environment" -ForegroundColor Cyan
Write-Host "------------------------------------" -ForegroundColor Cyan

$VenvScript = ".\pymeetings\Scripts\Activate.ps1"

if (Test-Path $VenvScript) {
    Write-Host "Activating 'pymeetings' environment..." -ForegroundColor Green
    . $VenvScript
}
else {
    Write-Host "Could not find activation script at $VenvScript" -ForegroundColor Yellow
    Write-Host "Run 'python install_libs.py' to create it." -ForegroundColor Yellow
}

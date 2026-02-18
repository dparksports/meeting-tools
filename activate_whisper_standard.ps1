<# 
  Whisper Standard — Quick Start
  Activates the whisper-standard virtual environment and shows usage instructions.
  Run:  .\activate_whisper_standard.ps1
#>

# Resolve venv path: ~/venvs/whisper-standard
$venvDir = Join-Path $HOME "venvs\whisper-standard"
$venvPath = Join-Path $venvDir "Scripts\Activate.ps1"

if (-not (Test-Path $venvPath)) {
    Write-Host ""
    Write-Host "  [!] Virtual environment not found at: $venvDir" -ForegroundColor Yellow
    Write-Host ""
    $answer = Read-Host "      Create it now? (Y/n)"
    if ($answer -and $answer -notin @('Y', 'y', 'yes', '')) {
        Write-Host "  Aborted." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "  Creating virtual environment ..." -ForegroundColor Cyan
    python -m venv $venvDir
    if (-not (Test-Path $venvPath)) {
        Write-Host "  [!] Failed to create venv. Make sure Python is on PATH." -ForegroundColor Red
        exit 1
    }

    # Activate, then upgrade pip and install deps
    & $venvPath
    Write-Host "  Upgrading pip ..." -ForegroundColor Cyan
    pip install --upgrade pip --quiet

    Write-Host "  Installing PyTorch (CUDA 12.8) ..." -ForegroundColor Cyan
    pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet

    Write-Host "  Installing openai-whisper ..." -ForegroundColor Cyan
    pip install openai-whisper --quiet

    Write-Host ""
    Write-Host "  Done! All dependencies installed." -ForegroundColor Green
    Write-Host ""
}
else {
    & $venvPath
}

# Check for ffmpeg
$ffmpegFound = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegFound) {
    Write-Host ""
    Write-Host "  [WARNING] ffmpeg not found on PATH." -ForegroundColor Yellow
    Write-Host "  Standard Whisper requires ffmpeg for audio decoding." -ForegroundColor Yellow
    Write-Host "  Install: https://ffmpeg.org/download.html  or  winget install ffmpeg" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host "    Whisper Standard Transcribe - Ready" -ForegroundColor Cyan
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  USAGE:" -ForegroundColor Green
Write-Host '    python transcribe_standard.py "D:/Videos/meetings"'
Write-Host '    python transcribe_standard.py "D:/Videos/meetings" --model medium.en'
Write-Host '    python transcribe_standard.py "D:/Videos/meetings" --model large'
Write-Host ""
Write-Host "  AVAILABLE MODELS:" -ForegroundColor Green
Write-Host "    tiny.en, tiny, base.en, base, small.en, small,"
Write-Host "    medium.en, medium, large, turbo"
Write-Host ""
Write-Host "  OPTIONS:" -ForegroundColor Green
Write-Host "    --model NAME       Whisper model to use (default: medium.en)"
Write-Host "    --device cpu       Force CPU mode"
Write-Host "    --beam-size 3      Adjust beam size (default: 5)"
Write-Host ""
Write-Host "  OUTPUT:" -ForegroundColor Green
Write-Host "    Transcripts saved to: FOLDER.whisper-MODEL/FILE.whisper-MODEL.txt"
Write-Host "    Re-run to resume - already-done files are skipped."
Write-Host ""
Write-Host ('  Venv: ' + $venvDir) -ForegroundColor DarkGray
Write-Host ""

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$ActivatePath = Join-Path $VenvPath "Scripts\Activate.ps1"

Set-Location $ProjectRoot

Write-Host "Classroom Intelligence Demo - audio setup"
Write-Host "Project: $ProjectRoot"

if (Test-Path $ActivatePath) {
    Write-Host "Activating local .venv..."
    . $ActivatePath
} else {
    Write-Host "Local .venv not found. Using current Python environment."
}

python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-audio.txt

Write-Host "Verifying Whisper imports and CUDA preference..."
python -c "import torch; print('torch', torch.__version__); print('torch CUDA', torch.version.cuda); print('cuda available', torch.cuda.is_available()); import whisper; print('openai-whisper ready'); import faster_whisper; print('faster-whisper ready')"

Write-Host "Audio setup complete. The app prefers openai-whisper on CUDA when torch.cuda.is_available() is true."

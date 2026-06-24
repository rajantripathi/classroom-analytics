$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$ActivatePath = Join-Path $VenvPath "Scripts\Activate.ps1"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

Set-Location $ProjectRoot

Write-Host "Classroom Intelligence Demo - GPU setup"
Write-Host "Project: $ProjectRoot"

if (-not (Test-Path $PythonPath)) {
    Write-Host "Local .venv not found. Creating .venv..."
    python -m venv .venv
}

if (Test-Path $ActivatePath) {
    Write-Host "Activating local .venv..."
    . $ActivatePath
} else {
    throw "Could not find .venv activation script at $ActivatePath"
}

Write-Host "Using Python:"
python --version

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Removing CPU-only PyTorch packages if present..."
python -m pip uninstall -y torch torchvision torchaudio

Write-Host "Installing CUDA-enabled PyTorch from official PyTorch cu128 wheel index..."
Write-Host "If this fails, check the official PyTorch selector for the current RTX 5090/CUDA wheel."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing dashboard/runtime dependencies..."
python -m pip install -r requirements.txt

Write-Host "Verifying CUDA from PyTorch..."
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "GPU setup complete. For this shell, activate with: .\.venv\Scripts\Activate.ps1"

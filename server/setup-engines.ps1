# Creates one virtual environment per TTS engine under <models_dir>\venvs\<engine>.
# Usage:  .\setup-engines.ps1                 # all engines
#         .\setup-engines.ps1 omnivoice fish_s2
# Engines pin different torch/transformers versions, so each gets its own environment.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Names)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\scripts\common.ps1"

$uv = Get-Uv
$models = Get-ModelsDir
if (-not $Names) { $Names = @($Engines.Keys) }
New-Item -ItemType Directory -Force (Join-Path $models "venvs"), (Join-Path $models "src") | Out-Null

foreach ($name in $Names) {
    $e = $Engines[$name]
    if (-not $e) { throw "Unknown engine '$name'. Known: $($Engines.Keys -join ', ')" }
    $venv = Join-Path $models "venvs\$name"
    $py = Join-Path $venv "Scripts\python.exe"
    Write-Host "`n=== $name -> $venv" -ForegroundColor Cyan

    if ($name -eq "breeze") {
        # Breeze TTS 2: BreezeBlue's inference code (cloned) with its pinned packages; torch 2.9.1 for CUDA 12.8.
        $src = Join-Path $models "src\$($e.Dir)"
        if (-not (Test-Path $src)) { git clone $e.Git $src }
        git -C $src fetch --quiet origin $e.Commit 2>$null
        git -C $src checkout --quiet $e.Commit
        if (-not (Test-Path $py)) { & $uv venv --python $e.Python $venv }
        & $uv pip install --python $py torch==2.9.1+cu128 torchaudio==2.9.1+cu128 `
            --extra-index-url $TorchIndex --index-strategy unsafe-best-match
        & $uv pip install --python $py qwen-tts==0.1.1 transformers==4.57.3 "huggingface-hub>=0.34" "numpy>=2.0" `
            "soundfile>=0.13" "triton-windows>=3.5,<3.6" --extra-index-url $TorchIndex --index-strategy unsafe-best-match
    } elseif ($e.Git) {
        $src = Join-Path $models "src\$($e.Dir)"
        if (-not (Test-Path $src)) { git clone $e.Git $src }
        git -C $src fetch --quiet origin $e.Commit 2>$null
        git -C $src checkout --quiet $e.Commit
        # Both projects ship uv lockfiles that pull CUDA 12.8 torch on Windows.
        $env:UV_PROJECT_ENVIRONMENT = $venv
        Push-Location $src
        try {
            if ($name -eq "fish_s2") { & $uv sync --python $e.Python --extra cu128 }
            else { & $uv sync --python $e.Python }
        } finally { Pop-Location; Remove-Item Env:UV_PROJECT_ENVIRONMENT }
        # torch.compile on Windows (used by both) needs Triton.
        & $uv pip install --python $py "triton-windows<3.5"
    } else {
        if (-not (Test-Path $py)) { & $uv venv --python $e.Python $venv }
        & $uv pip install --python $py torch==2.8.0+cu128 torchaudio==2.8.0+cu128 `
            --extra-index-url $TorchIndex --index-strategy unsafe-best-match
        $pkg = @{ omnivoice = "omnivoice"; qwen3tts = "qwen-tts"; voxcpm = "voxcpm" }[$name]
        & $uv pip install --python $py $pkg soundfile numpy huggingface_hub `
            --extra-index-url $TorchIndex --index-strategy unsafe-best-match
        # VoxCPM2 runs about 4.5x faster with torch.compile, which needs Triton on Windows.
        if ($name -eq "voxcpm") { & $uv pip install --python $py "triton-windows<3.5" }
    }
    & $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
}
Write-Host "`nDone. Next: .\download-models.ps1" -ForegroundColor Green

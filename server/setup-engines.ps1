# Creates one virtual environment per TTS engine under <models_dir>\venvs\<engine>.
# Usage:  .\setup-engines.ps1                 # VoxCPM2 (speech) and Qwen3-TTS (voice design)
#         .\setup-engines.ps1 -All            # every engine (OmniVoice, Fish S2, IndexTTS, Breeze too)
#         .\setup-engines.ps1 omnivoice fish_s2
#         .\setup-engines.ps1 voxcpm -Latest  # newest packages instead of engine-locks\voxcpm.txt (and rewrite it)
# Engines pin different torch/transformers versions, so each gets its own environment. OmniVoice, Qwen3-TTS,
# VoxCPM2 and Breeze install the exact versions in engine-locks\<engine>.txt (a working set; newer releases
# can break the adapters); Fish S2 and IndexTTS use the lockfiles of their pinned upstream commits.
param([switch]$Latest, [switch]$All, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Names)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\scripts\common.ps1"

$uv = Get-Uv
$models = Get-ModelsDir
if (-not $Names) { $Names = if ($All) { @($Engines.Keys) } else { $DefaultEngines } }
New-Item -ItemType Directory -Force (Join-Path $models "venvs"), (Join-Path $models "src") | Out-Null

foreach ($name in $Names) {
    $e = $Engines[$name]
    if (-not $e) { throw "Unknown engine '$name'. Known: $($Engines.Keys -join ', ')" }
    $venv = Join-Path $models "venvs\$name"
    $py = Join-Path $venv "Scripts\python.exe"
    $lock = Join-Path $PSScriptRoot "engine-locks\$name.txt"
    $locked = (Test-Path $lock) -and -not $Latest
    Write-Host "`n=== $name -> $venv" -ForegroundColor Cyan

    if ($name -eq "breeze") {
        # Breeze TTS 2: BreezeBlue's inference code (cloned) with its pinned packages; torch 2.9.1 for CUDA 12.8.
        $src = Join-Path $models "src\$($e.Dir)"
        if (-not (Test-Path $src)) { git clone $e.Git $src }
        git -C $src fetch --quiet origin $e.Commit 2>$null
        git -C $src checkout --quiet $e.Commit
        if (-not (Test-Path $py)) { & $uv venv --python $e.Python $venv }
        if ($locked) {
            & $uv pip install --python $py -r $lock --extra-index-url $TorchIndex --index-strategy unsafe-best-match
        } else {
            & $uv pip install --python $py torch==2.9.1+cu128 torchaudio==2.9.1+cu128 `
                --extra-index-url $TorchIndex --index-strategy unsafe-best-match
            & $uv pip install --python $py qwen-tts==0.1.1 transformers==4.57.3 "huggingface-hub>=0.34" "numpy>=2.0" `
                "soundfile>=0.13" "triton-windows>=3.5,<3.6" --extra-index-url $TorchIndex --index-strategy unsafe-best-match
        }
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
        if ($locked) {
            & $uv pip install --python $py -r $lock --extra-index-url $TorchIndex --index-strategy unsafe-best-match
        } else {
            & $uv pip install --python $py torch==2.8.0+cu128 torchaudio==2.8.0+cu128 `
                --extra-index-url $TorchIndex --index-strategy unsafe-best-match
            $pkg = @{ omnivoice = "omnivoice"; qwen3tts = "qwen-tts"; voxcpm = "voxcpm openai-whisper" }[$name] -split " "
            & $uv pip install --python $py $pkg soundfile numpy huggingface_hub `
                --extra-index-url $TorchIndex --index-strategy unsafe-best-match
            # VoxCPM2 runs about 4.5x faster with torch.compile, which needs Triton on Windows.
            if ($name -eq "voxcpm") { & $uv pip install --python $py "triton-windows<3.5" }
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Installing $name failed (exit code $LASTEXITCODE)." }
    if ($Latest -and (Test-Path $lock)) {
        $head = Get-Content $lock | Where-Object { $_ -like "#*" }
        [IO.File]::WriteAllLines($lock, @($head) + @(& $uv pip freeze --python $py))
        Write-Host "Wrote $lock" -ForegroundColor Yellow
    }
    & $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
}
Write-Host "`nDone. Next: .\download-models.ps1" -ForegroundColor Green

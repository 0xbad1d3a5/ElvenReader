# Downloads llama.cpp (Windows CPU + CUDA builds) and the dramatization LLM into <models_dir>.
# Usage:  .\download-llm.ps1                    # Qwen3.5-35B-A3B Q4_K_M (22 GB)
#         .\download-llm.ps1 -Alternative       # also Qwen3-30B-A3B-Instruct-2507 Q4_K_M (18.6 GB) and gpt-oss-20b MXFP4 (12 GB)
param([switch]$Alternative, [string]$LlamaBuild = "b11870")
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\scripts\common.ps1"

$models = Get-ModelsDir
$ll = Join-Path $models "llama.cpp"
New-Item -ItemType Directory -Force $ll, (Join-Path $models "llm") | Out-Null
$base = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaBuild"
$assets = @(
    @("llama-$LlamaBuild-bin-win-cpu-x64.zip", "cpu"),
    @("llama-$LlamaBuild-bin-win-cuda-12.4-x64.zip", "cuda"),
    @("cudart-llama-bin-win-cuda-12.4-x64.zip", "cuda")
)
foreach ($a in $assets) {
    $marker = if ($a[0] -like "cudart*") { "cudart64_12.dll" } else { "llama-server.exe" }
    if (Test-Path (Join-Path $ll "$($a[1])\$marker")) { continue }
    $zip = Join-Path $ll $a[0]
    Write-Host "Downloading $($a[0])..."
    curl.exe -sSL -o $zip "$base/$($a[0])"
    Expand-Archive -Force $zip (Join-Path $ll $a[1])
    Remove-Item $zip
}

# Any engine environment has huggingface_hub.
$py = Get-ChildItem (Join-Path $models "venvs\*\Scripts\python.exe") | Select-Object -First 1
if (-not $py) { throw "Run setup-engines.ps1 first." }
$files = @(@("unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-Q4_K_M.gguf"))
if ($Alternative) {
    $files += , @("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
    $files += , @("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf")
}
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
foreach ($f in $files) {
    & $py.FullName -c "from huggingface_hub import hf_hub_download as d; print(d('$($f[0])', '$($f[1])', local_dir=r'$models\llm'))"
}
Write-Host "Done. The server starts llama.cpp itself when a book is analysed." -ForegroundColor Green

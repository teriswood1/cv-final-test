$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

python train_aspp_fixed_aug.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python test_aspp_fixed_aug.py
exit $LASTEXITCODE

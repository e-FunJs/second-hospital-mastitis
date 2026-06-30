$ErrorActionPreference = "Stop"

$LocalProject = "C:\Users\aaaa\Documents\Codex\2026-06-23\plm\work\nonpuerperal-mastitis-rag"
$LocalParent = Split-Path $LocalProject -Parent
$ProjectName = Split-Path $LocalProject -Leaf
$RemoteHost = "amax@10.102.102.5"
$RemoteBase = "/home/amax/E-FUN/Secondo_Ospedale"
$RemoteProject = "$RemoteBase/nonpuerperal-mastitis-rag"
$Archive = Join-Path $env:TEMP "nonpuerperal-mastitis-rag.tar.gz"
$RemoteArchive = "/tmp/nonpuerperal-mastitis-rag.tar.gz"

if (Test-Path $Archive) {
  Remove-Item $Archive -Force
}

Push-Location $LocalParent
try {
  tar -czf $Archive $ProjectName
}
finally {
  Pop-Location
}

ssh $RemoteHost "mkdir -p '$RemoteBase'"
scp $Archive "${RemoteHost}:$RemoteArchive"
ssh $RemoteHost "tar -xzf '$RemoteArchive' -C '$RemoteBase'"

ssh $RemoteHost "cd '$RemoteProject' && bash scripts/bootstrap_server.sh"

Write-Host "Uploaded and initialized: $RemoteProject"

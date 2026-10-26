# Generate an expert team using the huggingface-download conda environment.
# Usage:
#   .\scripts\run_expert_team.ps1 "How can we make ML models more sample-efficient?" -n 3 -d 8
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ResearchContext,
    [int]$NumExperts = 3,
    [int]$Diversity = 5,
    [string]$Model = "gpt-4.1",
    [string]$Save
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Args = @(
    "-m", "botmaker.expert_team",
    $ResearchContext,
    "-n", $NumExperts,
    "-d", $Diversity,
    "--model", $Model
)
if ($Save) { $Args += @("--save", $Save) }

Push-Location (Join-Path $Root "src")
try {
    conda run -n huggingface-download python @Args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

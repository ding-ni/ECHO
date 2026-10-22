param(
    [ValidateSet('气象核验', '边界复核', '水位对照', '气象标准化')]
    [string]$步骤 = '气象核验'
)

$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskProject '本地环境\python.exe'

function Invoke-EchoStep {
    & $taskPython -m echo @args
    if ($LASTEXITCODE -ne 0) { throw "核验失败，退出码：$LASTEXITCODE" }
}

Push-Location $taskProject
try {
    switch ($步骤) {
        '气象核验' {
            Invoke-EchoStep weather-audit --manifest '本地数据/审计/候选站点.csv' --output '本地数据/审计/气象二十站质量核验.json' --limit 20
        }
        '边界复核' {
            Invoke-EchoStep boundary-audit --manifest '本地数据/审计/候选站点.csv' --shapefile '本地数据/原始/shapefiles/shapefiles/CAMELSH_shapefile.shp' --source-crs EPSG:4326 --crs-evidence 'https://www.nature.com/articles/s41597-025-05612-6 Methods: WGS84' --output '本地数据/审计/候选站点_边界复核.csv'
        }
        '水位对照' {
            Invoke-EchoStep station-evidence --manifest '本地数据/审计/候选站点_边界复核.csv' --output '本地数据/审计/USGS支持期水位核验.json' --snapshots '本地参考资料/sources/usgs_support_H' --limit 20
        }
        '气象标准化' {
            Invoke-EchoStep weather-prepare --manifest '本地数据/审计/USGS支持期水位核验.candidates.csv' --mapping '01_数据准备/气象字段核验_v1.json' --output '本地数据/标准化气象/二十站预备_步末对齐'
        }
    }
}
finally { Pop-Location }

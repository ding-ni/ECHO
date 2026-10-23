param(
    [string]$GraphOutput = '本地数据/模型视图/真实分支河网_v1',
    [string]$CatalogueOutput = '本地数据/审计/正式空间资料目录_v1',
    [switch]$DownloadDEM
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $python = Join-Path $projectRoot '本地环境/python.exe'
    $index = '本地数据/模型视图/扩展水位训练_v2/index.json'
    if (-not (Test-Path (Join-Path $GraphOutput 'index.json'))) {
        & $python -m echo.geo.network_inputs --h-index $index `
            --shapefile '本地数据/原始/HydroBASINS/hybas_na_lev12_v1c/hybas_na_lev12_v1c.shp' `
            --catchment-shapefile '本地数据/原始/shapefiles/shapefiles/CAMELSH_shapefile.shp' `
            --output $GraphOutput
        if ($LASTEXITCODE -ne 0) { throw '真实分支河网构建失败' }
    }
    if (-not (Test-Path (Join-Path $CatalogueOutput 'index.json'))) {
        & $python -m echo.data.spatial_catalog --index $index `
            --graph-index (Join-Path $GraphOutput 'index.json') --output $CatalogueOutput
        if ($LASTEXITCODE -ne 0) { throw '官方空间资料目录查询失败' }
    }
    if ($DownloadDEM) {
        & $python -m echo.data.dem_assets --catalogue-root $CatalogueOutput `
            --output '本地数据/原始/3DEP_本地八站_v1' `
            --station-ids 01081500 02365470 02401000 08084800 11465200 01017000 01122500 02371500
        if ($LASTEXITCODE -ne 0) { throw 'DEM下载器执行失败' }
    }
}
finally { Pop-Location }

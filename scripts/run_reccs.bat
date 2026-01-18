@echo off
setlocal EnableDelayedExpansion

set "EDGE=%~1"
set "CLUST=%~2"
set "OUT=%~3"
set "NUM=%~4"
if "%NUM%"=="" set "NUM=1"

set "EDGE=%EDGE:\=/%"
set "CLUST=%CLUST:\=/%"
set "OUT=%OUT:\=/%"

docker compose --profile reccs run --rm reccs bash docker/reccs/run.sh "%EDGE%" "%CLUST%" "%OUT%" %NUM%

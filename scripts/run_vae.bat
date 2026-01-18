@echo off
setlocal EnableDelayedExpansion

set "INPUT_GRAPH=%~1"
set "OUTPUT_DIR=%~2"
set "FINAL_DIR=%~3"
set "SAMPLES=%~4"

if "%SAMPLES%"=="" set "SAMPLES=5"

set "INPUT_GRAPH=%INPUT_GRAPH:\=/%"
set "OUTPUT_DIR=%OUTPUT_DIR:\=/%"
set "FINAL_DIR=%FINAL_DIR:\=/%"

echo --- Running VAE Generator ---
uv run --locked python -m src.VAE.main --input_graph "%INPUT_GRAPH%" --output_dir "%OUTPUT_DIR%" --final_dir "%FINAL_DIR%" --samples %SAMPLES%

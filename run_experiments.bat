@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

set START_TIME=%TIME%
echo Start Time: %START_TIME%


REM ----- UV: reproducible env from uv.lock -----
uv lock --check
uv sync --locked
REM Runs data loader in the locked env
uv run --locked python -m src.data.load_data

REM ----- Generate Run ID -----
for /f "tokens=*" %%i in ('uv run -q python -c "import datetime; print(datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S'))" 2^>nul') do set RUN_ID=%%i
set RESULTS_DIR=results\%RUN_ID%

echo ==========================================================
echo Starting Run: %RUN_ID%
echo Results Directory: %RESULTS_DIR%
echo ==========================================================

mkdir "%RESULTS_DIR%\RECCS"
mkdir "%RESULTS_DIR%\EC-SBM"
mkdir "%RESULTS_DIR%\VAE"
mkdir "%RESULTS_DIR%\final_dataset\VAE"

REM ----- Docker: build images -----
docker compose --profile reccs build reccs
docker compose --profile ecsbm build ecsbm

if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=30
if not "%1"=="" set NUM_SAMPLES=%1

echo Using %NUM_SAMPLES% samples per experiment.

echo ==========================================================
echo Running Karate Experiments
echo ==========================================================
call scripts\run_reccs.bat data/karate.tsv data/karate.clustering.tsv "%RESULTS_DIR%/RECCS/karate" %NUM_SAMPLES%
call scripts\run_ecsbm.bat data/karate.tsv data/karate.clustering.tsv "%RESULTS_DIR%/EC-SBM/karate" %NUM_SAMPLES%
call scripts\run_vae.bat data/karate.tsv "%RESULTS_DIR%/VAE/karate" "%RESULTS_DIR%/final_dataset/VAE" %NUM_SAMPLES%

echo ==========================================================
echo Running Polbooks Experiments
echo ==========================================================
call scripts\run_reccs.bat data/polbooks.tsv data/polbooks.clustering.tsv "%RESULTS_DIR%/RECCS/polbooks" %NUM_SAMPLES%
call scripts\run_ecsbm.bat data/polbooks.tsv data/polbooks.clustering.tsv "%RESULTS_DIR%/EC-SBM/polbooks" %NUM_SAMPLES%
call scripts\run_vae.bat data/polbooks.tsv "%RESULTS_DIR%/VAE/polbooks" "%RESULTS_DIR%/final_dataset/VAE" %NUM_SAMPLES%

echo ==========================================================
echo Running Football Experiments
echo ==========================================================
call scripts\run_reccs.bat data/football.tsv data/football.clustering.tsv "%RESULTS_DIR%/RECCS/football" %NUM_SAMPLES%
call scripts\run_ecsbm.bat data/football.tsv data/football.clustering.tsv "%RESULTS_DIR%/EC-SBM/football" %NUM_SAMPLES%
call scripts\run_vae.bat data/football.tsv "%RESULTS_DIR%/VAE/football" "%RESULTS_DIR%/final_dataset/VAE" %NUM_SAMPLES%

echo ==========================================================
echo Standardizing Outputs
echo ==========================================================
echo Standardizing outputs...
uv run python -m src.data.standardize_outputs --run_dir %RESULTS_DIR%

echo ==========================================================
echo Running Analysis & Visualization
echo ==========================================================

mkdir "%RESULTS_DIR%\analysis"
mkdir "%RESULTS_DIR%\plots"

echo [1/5] Basic Metrics...
uv run python -m src.analysis.basic_metrics --run_dir %RESULTS_DIR%

echo [2/5] Validation Utility Metrics...
uv run python -m src.analysis.validation_utility --run_dir %RESULTS_DIR%

echo [3/5] Fidelity Metrics...
uv run python -m src.analysis.fidelity_metrics --run_dir %RESULTS_DIR%

echo [4/5] Robustness Analysis...
uv run python -m src.analysis.robustness_analysis --run_dir %RESULTS_DIR% --noise_levels 0.1 0.2 --repetitions 3

echo [5/6] Quality Index...
uv run python -m src.analysis.quality_index --run_dir %RESULTS_DIR%

echo [6/6] Generating Plots...
uv run python -m src.analysis.generate_plots --run_dir %RESULTS_DIR%

echo [REMOVE] Generating Final Report...
uv run python -m src.utils.result_reporter --run_dir %RESULTS_DIR%

echo ==========================================================
echo Experiment Run %RUN_ID% Completed!
echo Results: %RESULTS_DIR%
echo Analysis: %RESULTS_DIR%\analysis
echo Plots: %RESULTS_DIR%\plots
echo Quality Index: %RESULTS_DIR%\analysis\quality_index.csv
echo ==========================================================
set END_TIME=%TIME%
echo End Time: %END_TIME%

REM Calculate duration using PowerShell
powershell -Command "$s=[TimeSpan]::Parse('%START_TIME%'); $e=[TimeSpan]::Parse('%END_TIME%'); if ($e -lt $s) { $e = $e.Add([TimeSpan]::FromDays(1)) }; $d=$e-$s; Write-Host 'Duration: ' $d.ToString()"

echo ==========================================================
pause

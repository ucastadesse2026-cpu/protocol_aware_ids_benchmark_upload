@echo off
REM Runs the five experiment scripts and make_tables.py with one config (default: configs\default.yaml).
REM Usage (Anaconda Prompt, repository root, env activated):   run_all.bat [configs\smoke.yaml]
REM NOTE: written for Windows cmd but not executed in the authoring sandbox (Linux).
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=configs\default.yaml
if not exist logs mkdir logs
for %%S in (run_baseline run_standard_attacks run_ccsds_attacks run_indomain run_defenses run_federated) do (
  echo === %%S with %CFG% ===
  python experiments\%%S.py --config %CFG% > logs\%%S.txt 2>&1
  if errorlevel 1 (
    echo FAILED: %%S - see logs\%%S.txt
    exit /b 1
  )
)
python make_tables.py --config %CFG% > logs\make_tables.txt 2>&1
if errorlevel 1 (
  echo FAILED: make_tables - see logs\make_tables.txt
  exit /b 1
)
echo Done. Tables and figures are in the results folder named by the config output.dir.
endlocal

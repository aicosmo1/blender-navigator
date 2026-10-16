@echo off
setlocal

if "%~1"=="" (
    echo Usage: run.bat "your Blender goal in quotes" [--json]
    echo Example: run.bat "bevel the edges of a cube"
    echo Example: run.bat "make cycles render background transparent"
    echo Example: run.bat "add cloth simulation to a plane" --json
    exit /b 1
)

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" blender_navigator.py %*

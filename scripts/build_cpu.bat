@echo off
REM Run from repo root: scripts\build_cpu.bat
set ROOT=%~dp0..
cd /d %ROOT%

echo Building matrix_cpu.exe ...
cl src\cpu\matrix_mult_cpu.c /Fe:matrix_cpu.exe /O2
if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    exit /b 1
)
del matrix_mult_cpu.obj 2>nul
echo matrix_cpu.exe built successfully.

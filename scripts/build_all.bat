@echo off
REM Run from repo root: scripts\build_all.bat
REM Requires: x64 Native Tools Command Prompt for VS, CUDA in PATH.
set ROOT=%~dp0..
cd /d %ROOT%

echo Building CPU matrix multiplication...
cl src\cpu\matrix_mult_cpu.c /Fe:matrix_cpu.exe /O2
if %ERRORLEVEL% NEQ 0 goto :fail
del matrix_mult_cpu.obj 2>nul

echo Building convolution (C)...
cl src\cpu\convolution_cpu.c /Fe:convolution.exe /O2
if %ERRORLEVEL% NEQ 0 goto :fail
del convolution_cpu.obj 2>nul

echo Building naive CUDA matrix...
nvcc src\cuda\matrix_mult_naive.cu -o matrix_gpu.exe
if %ERRORLEVEL% NEQ 0 goto :fail

echo Building tiled CUDA matrix...
nvcc src\cuda\matrix_mult_tiled.cu -o matrix_gpu_tiled.exe
if %ERRORLEVEL% NEQ 0 goto :fail

echo Building cuBLAS matrix...
nvcc src\cuda\matrix_mult_cublas.cu -o matrix_cublas.exe -lcublas
if %ERRORLEVEL% NEQ 0 goto :fail

echo Building CUDA convolution...
nvcc src\cuda\convolution_gpu.cu -o convolution_gpu.exe
if %ERRORLEVEL% NEQ 0 goto :fail

echo Building shared library (matrix_lib.dll) for Python...
nvcc -Xcompiler /LD -shared src\lib\matrix_lib.cu -o matrix_lib.dll
if %ERRORLEVEL% NEQ 0 goto :fail

echo All builds succeeded.
goto :eof

:fail
echo Build failed.
exit /b 1

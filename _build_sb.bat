@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cmake --build build --target LichtFeld-Studio --config Release -j > build_out.log 2>&1
echo BUILD_EXIT=%errorlevel%

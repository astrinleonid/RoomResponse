@echo off
cd sdl_audio_core
call ..\. venv\Scripts\activate.bat
python setup.py build_ext --inplace
if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    python -c "import sdl_audio_core; print('Import successful!'); print('Version:', sdl_audio_core.__version__)"
) else (
    echo Build failed with error code %ERRORLEVEL%
)
pause

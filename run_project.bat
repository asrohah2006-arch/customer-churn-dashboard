@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY_CMD="

REM --- 1) Is a Python 3.11 already on PATH?
call :find311 python
if defined PY_CMD goto :found

REM --- 2) py launcher / Python install manager with 3.11 already installed?
call :find311 py -3.11
if defined PY_CMD goto :found

REM --- 3) Python install manager: install Python 3.11
where py >nul 2>&1
if not errorlevel 1 (
    echo Python 3.11 not found. Installing it with the Python install manager...
    py install 3.11
    call :find311 py -3.11
    if defined PY_CMD goto :found
)

REM --- 4) pymanager command (same install manager, unambiguous alias)
where pymanager >nul 2>&1
if not errorlevel 1 (
    echo Trying pymanager to install Python 3.11...
    pymanager install 3.11
    call :find311 py -3.11
    if defined PY_CMD goto :found
)

REM --- 5) python3.11 alias (available when the install manager PATH entry is set)
call :find311 python3.11
if defined PY_CMD goto :found

REM --- 6) winget fallback: classic python.org 3.11 installer
where winget >nul 2>&1
if not errorlevel 1 (
    echo Trying winget to install Python 3.11...
    winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements --disable-interactivity
    call :find311 "%LocalAppData%\Programs\Python\Python311\python.exe"
    if defined PY_CMD goto :found
    call :find311 py -3.11
    if defined PY_CMD goto :found
)

echo.
echo [ERROR] Could not find or install Python 3.11 automatically.
echo.
echo This project needs Python 3.11 because TensorFlow, its deep-learning
echo library, does not support the newest Python versions yet.
echo.
echo Fix: install the Python install manager from https://www.python.org/downloads/
echo (or from the Microsoft Store), then double-click run_project.bat again.
echo You do NOT need Visual Studio or any C++ compiler.
echo.
pause
exit /b 1

:find311
REM Test whether the given command runs Python 3.11.x. Sets PY_CMD on success.
%* --version > "%TEMP%\churn_pyver.txt" 2>&1
if errorlevel 1 exit /b 0
findstr /c:"Python 3.11." "%TEMP%\churn_pyver.txt" >nul 2>&1
if errorlevel 1 exit /b 0
set "PY_CMD=%*"
exit /b 0

:found
echo Using Python command: %PY_CMD%
%PY_CMD% --version

REM --- Recreate .venv if an older attempt built it with the wrong Python
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version 2>&1 | findstr /c:"Python 3.11." >nul
    if errorlevel 1 (
        echo Removing .venv created with a different Python version...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment with Python 3.11...
    %PY_CMD% -m venv .venv
    if errorlevel 1 goto :venv_fail
)

call ".venv\Scripts\activate.bat"

echo.
echo Upgrading pip, setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :pip_fail

echo.
echo Installing project packages (pre-built wheels only, no compiler needed)...
python -m pip install --only-binary :all: -r requirements.txt
if errorlevel 1 goto :req_fail

echo.
echo Generating the EDA report...
python src\eda.py
if errorlevel 1 goto :run_fail

echo.
echo Training the three models, this takes a few minutes...
python src\train.py --epochs 80
if errorlevel 1 goto :run_fail

echo.
echo Starting the dashboard. Open the address shown below,
echo usually http://localhost:8501 - press Ctrl+C here to stop it.
echo.
streamlit run app.py

echo.
pause
exit /b 0

:venv_fail
echo [ERROR] Could not create the virtual environment. Delete the .venv folder and run this file again.
goto :end_fail

:pip_fail
echo [ERROR] Could not upgrade pip. Check your internet connection and run this file again.
goto :end_fail

:req_fail
echo [ERROR] Package installation failed. This project needs Python 3.11 and an internet connection.
goto :end_fail

:run_fail
echo [ERROR] A project step failed. Read the message above, fix the problem, then run this file again.

:end_fail
echo.
pause
exit /b 1

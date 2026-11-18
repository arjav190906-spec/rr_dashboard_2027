@echo off
setlocal
cd /d "%~dp0rr_scout"
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher not found. Install Python 3.10+ and enable the Python Launcher.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto failed
pip install -r requirements.txt
if errorlevel 1 goto failed
python run_all.py
if errorlevel 1 goto failed
echo.
echo Pipeline finished. Opening its Streamlit scouting app...
streamlit run app.py
goto end
:failed
echo Pipeline could not finish. See the error above.
pause
:end

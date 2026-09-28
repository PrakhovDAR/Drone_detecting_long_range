@echo off
chcp 65001 >nul
echo Starting Data Pipeline...

if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found at .venv. Proceeding with global python.
)

echo.
echo === [1/5] Running 00_extract.py ===
python src\data_prep\00_extract.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo === [2/5] Running 01_filter.py ===
python src\data_prep\01_filter.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo === [3/5] Running 02_remap.py ===
python src\data_prep\02_remap.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo === [4/5] Running 03_hard_negatives.py ===
python src\data_prep\03_hard_negatives.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo === [5/5] Running 04_balance_and_split.py ===
python src\data_prep\04_balance_and_split.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo Pipeline completed successfully!
goto :EOF

:error
echo.
echo Pipeline failed with error code %ERRORLEVEL%.
exit /b %ERRORLEVEL%

@echo off
setlocal

cd /d "C:\Users\C772391\OneDrive - Anheuser-Busch InBev\Desktop\vilc-analysis"

if not exist logs mkdir logs

for /f "delims=" %%M in ('powershell -NoProfile -Command "(Get-Date).ToString(\"MMMM\")"') do set REPORT_MONTH=%%M
for /f "delims=" %%Y in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy\")"') do set REPORT_YEAR=%%Y

echo ================================================== >> logs\task_scheduler.log
echo [%date% %time%] Task started for %REPORT_MONTH% %REPORT_YEAR% >> logs\task_scheduler.log

".venv\Scripts\python.exe" generate_monthly_report_copy.py "%REPORT_MONTH%" "%REPORT_YEAR%" >> logs\task_scheduler.log 2>&1

echo [%date% %time%] Task ended with exit code %errorlevel% >> logs\task_scheduler.log
exit /b %errorlevel%
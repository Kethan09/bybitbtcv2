@echo off
cd /d %~dp0
call .venv\Scripts\activate
python live_bot.py
pause

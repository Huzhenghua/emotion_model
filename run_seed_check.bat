@echo off
REM 钉子3 多种子复现一键续跑（断点安全：已完成的种子自动跳过）
cd /d "%~dp0.."
python eval/seed_check.py
pause

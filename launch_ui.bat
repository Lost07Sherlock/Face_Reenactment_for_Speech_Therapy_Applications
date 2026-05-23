@echo off
title Face Reenactment + Voice Transformation
echo ==============================================
echo   Launching Integrated Streamlit Studio
echo ==============================================

call "%UserProfile%\Miniconda3\Scripts\activate.bat"
call conda activate main_env

cd LivePortrait
call streamlit run gui_final.py

pause

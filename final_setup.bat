@echo off
setlocal EnableDelayedExpansion

echo ===================================================
echo   MASTER SETUP  (Speech Therapy - LivePortrait Studio)
echo   - clones LivePortrait + Seed-VC
echo   - injects gui_final.py + evaluation_pipeline.py
echo   - copies requirements.txt into the LivePortrait repo
echo   - sets up main_env / seed-vc / whisper conda envs
echo   - downloads all pretrained weights
echo ===================================================
echo.

REM ============================================================
REM Pre-flight 1: Git for Windows
REM ============================================================
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Git not found. Attempting silent install...
    where winget >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        winget install --id Git.Git -e --source winget --silent --accept-source-agreements --accept-package-agreements
    ) else (
        powershell -Command "Invoke-WebRequest 'https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/Git-2.45.2-64-bit.exe' -OutFile 'git-installer.exe'"
        start /wait "" git-installer.exe /VERYSILENT /NORESTART /SUPPRESSMSGBOXES /NOCANCEL
        del git-installer.exe
    )
    set "PATH=%PATH%;C:\Program Files\Git\cmd;C:\Program Files\Git\bin"
    where git >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Git installation failed. Install manually from https://git-scm.com and rerun.
        pause
        exit /b 1
    )
)
echo [OK] Git available:
git --version

REM ============================================================
REM Pre-flight 2: Miniconda
REM ============================================================
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Conda not found. Installing Miniconda silently...
    powershell -Command "Invoke-WebRequest https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile miniconda.exe"
    start /wait "" miniconda.exe /S /D=%UserProfile%\Miniconda3
    del miniconda.exe
    set "PATH=%PATH%;%UserProfile%\Miniconda3;%UserProfile%\Miniconda3\Scripts;%UserProfile%\Miniconda3\Library\bin"
    where conda >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Conda installation failed. Install Miniconda manually and rerun.
        pause
        exit /b 1
    )
)
call "%UserProfile%\Miniconda3\Scripts\activate.bat"
echo [OK] Conda available:
conda --version


echo.
echo ===================================================
echo   [Step 0] Pre-flight: required template files
echo ===================================================
REM Verify the three template files exist in the main folder
set "MISSING=0"
if not exist "gui_final.py"           ( echo [MISSING] gui_final.py           & set "MISSING=1" )
if not exist "evaluation_pipeline.py" ( echo [MISSING] evaluation_pipeline.py & set "MISSING=1" )
if not exist "requirements.txt"       ( echo [MISSING] requirements.txt       & set "MISSING=1" )
if "!MISSING!"=="1" (
    echo.
    echo [ERROR] One or more template files missing from the main folder.
    echo         Place them next to this bat and rerun.
    pause
    exit /b 1
)
echo [OK] All three template files present.


echo.
echo ===================================================
echo   [Step 1] Cloning repositories
echo ===================================================

if not exist "LivePortrait" (
    echo Cloning LivePortrait...
    git clone https://github.com/KlingAIResearch/LivePortrait.git LivePortrait
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] LivePortrait clone failed.
        echo         If the URL is wrong, edit this bat to point at the right fork
        echo         (e.g. https://github.com/KwaiVGI/LivePortrait.git).
        pause
        exit /b 1
    )
) else (
    echo [INFO] LivePortrait folder already exists - skipping clone.
)

if not exist "Voice-Transformation" (
    echo Cloning seed-vc into Voice-Transformation...
    git clone https://github.com/Plachtaa/seed-vc.git Voice-Transformation
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] seed-vc clone failed. Check your network.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Voice-Transformation folder already exists - skipping clone.
)


echo.
echo ===================================================
echo   [Step 2] Injecting template files into LivePortrait
echo ===================================================

REM ---- gui_final.py at repo root ----
copy /Y "gui_final.py" "LivePortrait\gui_final.py" >nul
echo [OK] gui_final.py        -^> LivePortrait\

REM ---- evaluation_pipeline.py inside src/ ----
if not exist "LivePortrait\src" mkdir "LivePortrait\src"
copy /Y "evaluation_pipeline.py" "LivePortrait\src\evaluation_pipeline.py" >nul
echo [OK] evaluation_pipeline -^> LivePortrait\src\

REM ---- requirements.txt overwrite ----
copy /Y "requirements.txt" "LivePortrait\requirements.txt" >nul
echo [OK] requirements.txt    -^> LivePortrait\  (overwrote repo default)

REM ---- Working folders inside LivePortrait ----
if not exist "LivePortrait\temp_uploads"   mkdir "LivePortrait\temp_uploads"
if not exist "LivePortrait\temp_patients"  mkdir "LivePortrait\temp_patients"
if not exist "LivePortrait\animations"     mkdir "LivePortrait\animations"
if not exist "LivePortrait\outputs"        mkdir "LivePortrait\outputs"
echo [OK] Working folders created (temp_uploads, temp_patients, animations, outputs)


echo.
echo ===================================================
echo   [Part 1/4] LivePortrait env (main_env)
echo ===================================================
cd LivePortrait

call conda remove -y -n main_env --all >nul 2>&1
call conda create -y -n main_env python=3.10
call conda activate main_env

REM -------------------------------------------------------------------
REM Static ffmpeg (NOT conda-forge ffmpeg)
REM We avoid `conda install -c conda-forge ffmpeg` because recent builds
REM pull in gdk-pixbuf / glib / gettext, which causes the
REM "libintl_bind_textdomain_codeset" DLL error on fresh Windows installs.
REM
REM imageio-ffmpeg ships a statically-linked ffmpeg.exe with no external
REM DLL dependencies. It's also in requirements.txt, so it gets installed
REM there too - this just makes the static binary callable as `ffmpeg`
REM from PATH for subprocess.run([...]) calls in gui_final.py.
REM -------------------------------------------------------------------
python -m pip install --upgrade pip setuptools wheel
pip install imageio-ffmpeg==0.5.1
for /f "delims=" %%i in ('python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"') do set "FFMPEG_EXE=%%i"
echo Static ffmpeg at: !FFMPEG_EXE!
copy /Y "!FFMPEG_EXE!" "%CONDA_PREFIX%\Scripts\ffmpeg.exe" >nul
ffmpeg -version 2>nul | findstr /B "ffmpeg version"
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] ffmpeg not callable after copy. Aborting.
    pause
    exit /b 1
)

REM ---- Pin numpy first ----
pip install numpy==1.26.4

REM ---- PyTorch (CUDA 12.1) ----
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

REM ---- Repo requirements (CWD is LivePortrait\ at this point) ----
pip install --no-cache-dir -r requirements.txt

REM ---- Webcam recording deps (NOT in requirements.txt) ----
REM streamlit-webrtc + aiortc + av: video/audio capture from browser
REM mediapipe: face landmark detection for the alignment overlay (FaceGuideProcessor)
pip install streamlit-webrtc==0.47.7 aiortc==1.9.0
pip install av==11.0.0 --only-binary=:all:
pip install mediapipe==0.10.14

REM ---- Evaluation extras (NOT in requirements.txt) ----
REM librosa + soundfile: audio loading + RMS guardrail + acoustic dialogue score
REM huggingface-hub: weight downloads (LivePortrait + ContentVec)
pip install librosa==0.10.2 soundfile==0.12.1
pip install huggingface-hub>=0.28.1

REM ---- Lock numpy again (some deps may have bumped it) ----
pip install --no-deps numpy==1.26.4

REM ---- Sanity check: env should not contain gdk-pixbuf/poppler ----
echo.
echo Verifying env is clean of problem DLLs...
call conda list | findstr /I "gdk-pixbuf poppler" >nul
if !ERRORLEVEL! EQU 0 (
    echo [WARN] gdk-pixbuf or poppler detected. Removing...
    call conda remove -y --force gdk-pixbuf poppler 2>nul
) else (
    echo [OK]   No gdk-pixbuf/poppler in main_env.
)

REM ---- Download LivePortrait pretrained weights from HuggingFace ----
if not exist "pretrained_weights\liveportrait\base_models\appearance_feature_extractor.pth" (
    echo.
    echo Downloading LivePortrait pretrained weights from HuggingFace...
    echo This is ~500 MB and may take several minutes.
    echo try: > _dl_weights.py
    echo     from huggingface_hub import snapshot_download >> _dl_weights.py
    echo     snapshot_download( >> _dl_weights.py
    echo         repo_id='KwaiVGI/LivePortrait', >> _dl_weights.py
    echo         local_dir='pretrained_weights', >> _dl_weights.py
    echo         ignore_patterns=['*.git*', 'README.md', 'docs/*'], >> _dl_weights.py
    echo     ) >> _dl_weights.py
    echo     print('[OK] LivePortrait weights downloaded.') >> _dl_weights.py
    echo except Exception as e: >> _dl_weights.py
    echo     print('[ERROR] LivePortrait weight download failed:', e) >> _dl_weights.py
    python _dl_weights.py
    del _dl_weights.py
) else (
    echo [OK] LivePortrait weights already present.
)

REM ---- Pre-warm: silero-vad + ContentVec ----
echo.
echo Pre-downloading silero-vad and ContentVec-768...
echo try: > _warmup.py
echo     import torch >> _warmup.py
echo     from transformers import HubertModel >> _warmup.py
echo     print('[Warmup] silero-vad from torch.hub...') >> _warmup.py
echo     torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True) >> _warmup.py
echo     print('[Warmup] ContentVec-768 (lengyue233/content-vec-best)...') >> _warmup.py
echo     HubertModel.from_pretrained('lengyue233/content-vec-best') >> _warmup.py
echo     print('[Warmup] Done.') >> _warmup.py
echo except Exception as e: >> _warmup.py
echo     print('[Warmup] FAILED (non-fatal, will download on first run):', e) >> _warmup.py
python _warmup.py
del _warmup.py

cd ..


echo.
echo ===================================================
echo   [Part 2/4] Seed-VC env (seed-vc)
echo ===================================================
cd Voice-Transformation

call conda remove -y -n seed-vc --all >nul 2>&1
call conda create -y -n seed-vc python=3.10
call conda activate seed-vc

REM ---- Static ffmpeg (same fix as Part 1) ----
python -m pip install --upgrade pip setuptools wheel
pip install imageio-ffmpeg
for /f "delims=" %%i in ('python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"') do set "FFMPEG_EXE=%%i"
copy /Y "!FFMPEG_EXE!" "%CONDA_PREFIX%\Scripts\ffmpeg.exe" >nul

pip install numpy==1.26.4

echo Installing PyTorch (CUDA 12.4)...
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install accelerate scipy==1.13.1 librosa==0.10.2 huggingface-hub>=0.28.1 munch==4.0.0 einops==0.8.0 descript-audio-codec==1.0.0 pydub==0.25.1 resemblyzer jiwer==3.0.3 transformers==4.46.3 FreeSimpleGUI==5.1.1 soundfile==0.12.1 sounddevice==0.5.0 modelscope==1.18.1 funasr==1.1.5 hydra-core==1.3.2 pyyaml python-dotenv

pip install onnxruntime-gpu==1.18.1

pip install gradio==5.23.0

pip install --no-deps numpy==1.26.4

REM ---- Sanity check ----
call conda list | findstr /I "gdk-pixbuf poppler" >nul
if !ERRORLEVEL! EQU 0 (
    echo [WARN] gdk-pixbuf or poppler in seed-vc. Removing...
    call conda remove -y --force gdk-pixbuf poppler 2>nul
) else (
    echo [OK]   No gdk-pixbuf/poppler in seed-vc.
)

REM ---- Predefined voices folder ----
if not exist "predefined_voices" (
    mkdir predefined_voices
    echo Place these 6 reference voice WAV files here:                        > predefined_voices\README.txt
    echo   male_high.wav   male_medium.wav   male_low.wav                    >> predefined_voices\README.txt
    echo   female_high.wav female_medium.wav female_low.wav                  >> predefined_voices\README.txt
    echo.                                                                     >> predefined_voices\README.txt
    echo These are used by Seed-VC when the user selects "Predefined Voice    >> predefined_voices\README.txt
    echo Samples" in the Streamlit UI.  Each should be 5-10 seconds of clear  >> predefined_voices\README.txt
    echo speech, mono, 16 kHz or higher.                                      >> predefined_voices\README.txt
)


echo.
echo ===================================================
echo   [Part 3/4] Downloading Seed-VC V2 models
echo ===================================================

echo import sys > predownload.py
echo from argparse import Namespace >> predownload.py
echo import app >> predownload.py
echo args = Namespace(compile=False, enable_v1=True, enable_v2=True) >> predownload.py
echo try: >> predownload.py
echo     app.load_v2_models(args) >> predownload.py
echo     print('Seed-VC V2 weights downloaded successfully') >> predownload.py
echo except Exception as e: >> predownload.py
echo     print('Seed-VC weight download failed:', e) >> predownload.py
python predownload.py
del predownload.py

cd ..


echo.
echo ===================================================
echo   [Part 4/4] Whisper env (whisper)
echo ===================================================

call conda remove -y -n whisper --all >nul 2>&1
call conda create -y -n whisper python=3.10
call conda activate whisper

REM ---- Static ffmpeg ----
python -m pip install --upgrade pip setuptools wheel
pip install imageio-ffmpeg
for /f "delims=" %%i in ('python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"') do set "FFMPEG_EXE=%%i"
copy /Y "!FFMPEG_EXE!" "%CONDA_PREFIX%\Scripts\ffmpeg.exe" >nul

pip install numpy==1.26.4

REM ---- PyTorch (matches main_env CUDA 12.1 for driver consistency) ----
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install openai-whisper

pip install --no-deps numpy==1.26.4

REM ---- Sanity check ----
call conda list | findstr /I "gdk-pixbuf poppler" >nul
if !ERRORLEVEL! EQU 0 (
    echo [WARN] gdk-pixbuf or poppler in whisper. Removing...
    call conda remove -y --force gdk-pixbuf poppler 2>nul
) else (
    echo [OK]   No gdk-pixbuf/poppler in whisper.
)

REM ---- Pre-download Whisper "base" model ----
echo Pre-downloading Whisper base model...
echo try: > _warmup_whisper.py
echo     import whisper >> _warmup_whisper.py
echo     whisper.load_model('base') >> _warmup_whisper.py
echo     print('[Warmup] Whisper base cached.') >> _warmup_whisper.py
echo except Exception as e: >> _warmup_whisper.py
echo     print('[Warmup] Whisper download failed (non-fatal):', e) >> _warmup_whisper.py
python _warmup_whisper.py
del _warmup_whisper.py


echo.
echo ===================================================
echo   SETUP COMPLETE
echo ===================================================
echo.
echo   Main folder layout (after setup):
echo     final_setup.bat
echo     gui_final.py            (template kept in main folder)
echo     evaluation_pipeline.py  (template kept in main folder)
echo     requirements.txt        (template kept in main folder)
echo     LivePortrait\           (cloned + customised)
echo       gui_final.py            ^<- copied from template
echo       requirements.txt        ^<- copied from template
echo       src\evaluation_pipeline.py  ^<- copied from template
echo       pretrained_weights\     ^<- downloaded from HuggingFace
echo       temp_uploads\, temp_patients\, animations\, outputs\
echo     Voice-Transformation\   (cloned seed-vc)
echo       predefined_voices\
echo.
echo   Conda envs created:
echo     - main_env    LivePortrait + Streamlit UI + evaluation
echo     - seed-vc     Voice transformation (subprocess from UI)
echo     - whisper     Transcript dialogue score (subprocess from UI)
echo.
echo   ffmpeg.exe in each env: static binary from imageio-ffmpeg
echo   (no glib / gdk-pixbuf / libintl dependencies)
echo.
echo   Models downloaded:
echo     - LivePortrait weights  (LivePortrait\pretrained_weights\)
echo     - silero-vad            (~/.cache/torch/hub/)
echo     - ContentVec-768        (~/.cache/huggingface/)
echo     - Seed-VC V2 models     (seed-vc env caches)
echo     - Whisper base          (~/.cache/whisper/)
echo.
echo   MANUAL STEP - place 6 reference voice WAVs in:
echo     Voice-Transformation\predefined_voices\
echo     Filenames:
echo       male_high.wav   male_medium.wav   male_low.wav
echo       female_high.wav female_medium.wav female_low.wav
echo.
echo   To start the app, ensure launch_gui.bat contains:
echo     call conda activate main_env
echo     cd LivePortrait
echo     streamlit run gui_final.py
echo ===================================================
pause
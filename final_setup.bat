@echo off
setlocal EnableDelayedExpansion

echo ===================================================
echo   MASTER SETUP
echo   Face Reenactment + Voice Transform + Whisper
echo ===================================================

REM ============================================================
REM   MINICONDA CHECK
REM ============================================================
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Installing Miniconda...
    powershell -Command "Invoke-WebRequest https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile miniconda.exe"
    start /wait "" miniconda.exe /S /D=%UserProfile%\Miniconda3
    del miniconda.exe
)

call "%UserProfile%\Miniconda3\Scripts\activate.bat"


echo.
echo ===================================================
echo   [Part 1/4] LivePortrait env (main_env)
echo ===================================================
cd LivePortrait_Face_Reenactment

REM ---- Clean recreate env (safe) ----
call conda remove -y -n main_env --all >nul 2>&1
call conda create -y -n main_env python=3.10
call conda activate main_env

REM ---- System deps ----
REM Only ffmpeg is genuinely needed via conda (used by subprocess calls
REM for audio extraction, A/V merging, browser-safe re-encoding).
REM Previously installed glib/libiconv/vc/zlib/poppler caused a known
REM DLL conflict (gdk_pixbuf <-> libintl) on fresh Windows installs.
call conda install -y -c conda-forge ffmpeg

REM ---- Upgrade pip tools ----
python -m pip install --upgrade pip setuptools wheel

REM ---- CRITICAL: Pin numpy first ----
pip install numpy==1.26.4

REM ---- PyTorch (CUDA 12.1) ----
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

REM ---- Repo requirements ----
pip install --no-cache-dir -r requirements.txt

REM ---- Stable CV + ML stack ----
pip install opencv-contrib-python==4.9.0.80 mediapipe==0.10.14
pip install onnxruntime-gpu==1.18.1

REM ---- FIX: Install AV (prebuilt only, no compile) ----
pip install av==11.0.0 --only-binary=:all:

REM ---- UI deps ----
pip install gradio==4.44.1 streamlit==1.35.0 streamlit-webrtc==0.47.7 aiortc==1.9.0 streamlit-cropper==0.2.2

REM ---- Evaluation pipeline deps ----
REM   librosa         - audio loading for VAD + acoustic dialogue score
REM   fastdtw         - DTW alignment (articulation AND acoustic dialogue)
REM   scipy           - cosine distance metric used by FastDTW
REM   transformers    - loads ContentVec via HubertModel
REM   huggingface-hub - HF download backend for ContentVec
REM   soundfile       - audio I/O backend used by librosa
echo Installing evaluation-pipeline dependencies...
pip install librosa==0.10.2 fastdtw==0.3.4 scipy==1.13.1
pip install transformers==4.46.3 huggingface-hub>=0.28.1
pip install soundfile==0.12.1

REM ---- Lock numpy again (some deps may have bumped it) ----
pip install --no-deps numpy==1.26.4

REM ---- Working folders ----
if not exist "temp_uploads"   mkdir temp_uploads
if not exist "temp_patients"  mkdir temp_patients
if not exist "animations"     mkdir animations
if not exist "outputs"        mkdir outputs

REM ---- LivePortrait weights (insightface) ----
if not exist "pretrained_weights\insightface" (
    pip install gdown
    gdown --folder https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib -O weights_temp
    mkdir pretrained_weights
    xcopy weights_temp pretrained_weights /E /I /Y
    rmdir /S /Q weights_temp
)

REM ---- Pre-warm: silero-vad + ContentVec ----
echo Pre-downloading silero-vad and ContentVec-768...
echo try: > _warmup.py
echo     import torch >> _warmup.py
echo     from transformers import HubertModel >> _warmup.py
echo     print('[Warmup] silero-vad from torch.hub...') >> _warmup.py
echo     torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True) >> _warmup.py
echo     print('[Warmup] ContentVec-768 (lengyue233/content-vec-best) from HuggingFace...') >> _warmup.py
echo     HubertModel.from_pretrained('lengyue233/content-vec-best') >> _warmup.py
echo     print('[Warmup] Done.') >> _warmup.py
echo except Exception as e: >> _warmup.py
echo     print('[Warmup] FAILED (non-fatal, models will download on first run):', e) >> _warmup.py
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

REM Minimal conda installs (same fix as Part 1)
call conda install -y -c conda-forge ffmpeg

python -m pip install --upgrade pip setuptools wheel

pip install numpy==1.26.4

echo Installing PyTorch (CUDA 12.4)...
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install accelerate scipy==1.13.1 librosa==0.10.2 huggingface-hub>=0.28.1 munch==4.0.0 einops==0.8.0 descript-audio-codec==1.0.0 pydub==0.25.1 resemblyzer jiwer==3.0.3 transformers==4.46.3 FreeSimpleGUI==5.1.1 soundfile==0.12.1 sounddevice==0.5.0 modelscope==1.18.1 funasr==1.1.5 hydra-core==1.3.2 pyyaml python-dotenv

pip install onnxruntime-gpu==1.18.1

pip install gradio==5.23.0

pip install --no-deps numpy==1.26.4

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
echo   Transcript-based dialogue accuracy score
echo ===================================================

call conda remove -y -n whisper --all >nul 2>&1
call conda create -y -n whisper python=3.10
call conda activate whisper

call conda install -y -c conda-forge ffmpeg

python -m pip install --upgrade pip setuptools wheel

pip install numpy==1.26.4

pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install openai-whisper

pip install --no-deps numpy==1.26.4

echo Pre-downloading Whisper "base" model...
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
echo   Conda environments created:
echo     - main_env   LivePortrait + Streamlit UI + evaluation
echo     - seed-vc      Voice transformation (subprocess from UI)
echo     - whisper      Transcript dialogue score (subprocess from UI)
echo.
echo   Models pre-downloaded:
echo     - silero-vad           (main_env, ~2 MB)
echo     - ContentVec-768       (main_env, ~376 MB)
echo     - Seed-VC V2 models    (seed-vc)
echo     - Whisper base         (whisper, ~142 MB)
echo     - LivePortrait weights (main_env, pretrained_weights/)
echo.
echo   MANUAL STEP - place 6 reference voice WAVs in:
echo     Voice-Transformation\predefined_voices\
echo     Filenames:
echo       male_high.wav   male_medium.wav   male_low.wav
echo       female_high.wav female_medium.wav female_low.wav
echo     See predefined_voices\README.txt for details.
echo.
echo   To start the app:  launch_gui.bat
echo ===================================================
pause
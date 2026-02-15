# PDF OCR + TTS (NuMarkdown + VibeVoice)

This app lets you pick a PDF, OCR it with **NVIDIA-Nemotron-Parse-v1.1**, then unloads the OCR model, loads **VibeVoice**, and reads the OCR output aloud.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\henry\Desktop\OCR
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
# Install CUDA-enabled PyTorch (RTX 5090 requires CUDA 12.8+ builds)
.\.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If you are prompted for Hugging Face access, accept the model terms in your browser and run:

```powershell
.\.venv\Scripts\python -m pip install huggingface-hub
.\.venv\Scripts\huggingface-cli login
```

## Run

```powershell
.\.venv\Scripts\python app.py
```

## Smoke test (no model downloads)

```powershell
.\.venv\Scripts\python app.py --smoke
```

## End-to-end test (with models)

```powershell
.\.venv\Scripts\python test_pipeline.py --pdf test.pdf --pages 1 --dpi 96 --ocr-max-new-tokens 256 --tts-max-new-tokens 128 --tts-max-chars 300 --tts-model bezzam/VibeVoice-1.5B
```

## Notes

- The default OCR model id is `nvidia/NVIDIA-Nemotron-Parse-v1.1`.
- The default TTS model id is `microsoft/VibeVoice-1.5B-hf`. If it fails to load (for example, gated access), the app falls back to `bezzam/VibeVoice-1.5B`.
- You can override model IDs via the UI fields or environment variables:
  - `OCR_MODEL_ID`
  - `TTS_MODEL_ID`
  - `TTS_MODEL_ID_FALLBACK`
- Outputs are written to `outputs/` as Markdown text and WAV audio.

## Performance

These models are large. Expect significant GPU VRAM usage and first-run downloads. Close other GPU-heavy apps if you run into OOM.

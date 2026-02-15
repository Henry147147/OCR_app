# PDF OCR + TTS (GLM-OCR + Qwen3-TTS)

This desktop app lets you:

- OCR a PDF using **GLM-OCR** (default) or **NVIDIA-Nemotron-Parse**
- Generate speech from the OCR text using **Qwen3-TTS CustomVoice**
- Play/stop the generated WAV inside the app (Windows only)

## Setup (Windows PowerShell)

```powershell
cd C:\Users\henry\Desktop\OCR
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip

# Install CUDA-enabled PyTorch (example)
.\.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python -m pip install -r requirements.txt
```

If you are prompted for Hugging Face access, accept the model terms in your browser and run:

```powershell
.\.venv\Scripts\python -m pip install huggingface-hub
.\.venv\Scripts\huggingface-cli login
```

Optional (speed): install FlashAttention 2 if your environment supports it.

## Run

```powershell
.\.venv\Scripts\python app.py
```

## Smoke Test (No Model Downloads)

```powershell
.\.venv\Scripts\python app.py --smoke
```

## End-to-End Test (With Models)

```powershell
.\.venv\Scripts\python test_pipeline.py --pdf test.pdf --pages 1 --dpi 96 --ocr-engine glm --ocr-max-new-tokens 256 --tts-max-new-tokens 256 --tts-max-chars 300
```

## Notes

- Default OCR engine: `GLM-OCR` (model id: `zai-org/GLM-OCR`, prompt: `Text Recognition:`).
- Alternative OCR engine: `Nemotron-Parse` (model id: `nvidia/NVIDIA-Nemotron-Parse-v1.1`).
- Default TTS model id: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (speaker: `Ryan`).
- Outputs are written to `outputs/` as Markdown text and WAV audio.
- In-app audio playback uses `winsound` and is supported on Windows only.

### Environment Overrides

You can override defaults via environment variables:

- OCR:
  - `OCR_ENGINE` = `GLM-OCR` or `Nemotron-Parse`
  - `OCR_MODEL_ID_GLM`
  - `OCR_MODEL_ID_NEMOTRON`
  - `GLM_PROMPT`
- TTS:
  - `TTS_MODEL_ID`
  - `TTS_SPEAKER`
  - `TTS_LANGUAGE`

## Performance

These models are large. Expect significant GPU VRAM usage and first-run downloads. Close other GPU-heavy apps if you run into OOM.


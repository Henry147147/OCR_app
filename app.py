import gc
import os
import re
import shutil
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import fitz  # PyMuPDF
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig

from postprocessing import (
    convert_mmd_to_plain_text_ours,
    extract_classes_bboxes,
    postprocess_text,
    remove_nemotron_formatting,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OCR_ENGINE_GLM = "GLM-OCR"
OCR_ENGINE_NEMOTRON = "Nemotron-Parse"

DEFAULT_OCR_ENGINE = os.getenv("OCR_ENGINE", OCR_ENGINE_GLM)
DEFAULT_OCR_MODEL_ID_GLM = os.getenv("OCR_MODEL_ID_GLM", "zai-org/GLM-OCR")
DEFAULT_OCR_MODEL_ID_NEMOTRON = os.getenv("OCR_MODEL_ID_NEMOTRON", "nvidia/NVIDIA-Nemotron-Parse-v1.1")

DEFAULT_GLM_PROMPT = os.getenv("GLM_PROMPT", "Text Recognition:")
DEFAULT_GLM_MAX_NEW_TOKENS = int(os.getenv("GLM_OCR_MAX_NEW_TOKENS", "8192"))
DEFAULT_NEMOTRON_MAX_NEW_TOKENS = int(os.getenv("NEMOTRON_OCR_MAX_NEW_TOKENS", "2048"))

DEFAULT_TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_TTS_SPEAKER = os.getenv("TTS_SPEAKER", "Ryan")
DEFAULT_TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "Auto")
DEFAULT_TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "1200"))
DEFAULT_TTS_SILENCE_SECONDS = float(os.getenv("TTS_SILENCE_SECONDS", "0.25"))
DEFAULT_TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "2048"))
DEFAULT_TTS_TOP_P = float(os.getenv("TTS_TOP_P", "0.95"))
DEFAULT_TTS_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.7"))

NEMOTRON_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"


@dataclass
class AppState:
    pdf_path: str | None = None
    ocr_text: str = ""
    ocr_text_path: str | None = None
    tts_text: str = ""
    audio_path: str | None = None
    audio_sr: int | None = None


def detect_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def has_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


_RADIO_MODULES_READY = False
_MBART_PATCHED = False


def ensure_c_radio_modules():
    global _RADIO_MODULES_READY
    if _RADIO_MODULES_READY:
        return

    from transformers.dynamic_module_utils import _sanitize_module_name, create_dynamic_module
    from transformers.utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME

    repo_id = "nvidia/C-RADIOv2-H"
    snapshot_path = Path(snapshot_download(repo_id, allow_patterns=["*.py"]))
    commit_hash = snapshot_path.name
    module_path = Path(TRANSFORMERS_DYNAMIC_MODULE_NAME) / _sanitize_module_name(repo_id) / commit_hash
    create_dynamic_module(module_path)
    target_dir = Path(HF_MODULES_CACHE) / module_path

    for py_file in snapshot_path.glob("*.py"):
        dest = target_dir / py_file.name
        if not dest.exists():
            shutil.copy(py_file, dest)

    _RADIO_MODULES_READY = True


def ensure_mbart_layer_head_mask():
    global _MBART_PATCHED
    if _MBART_PATCHED:
        return

    try:
        from transformers.models.mbart.modeling_mbart import MBartDecoderLayer
    except Exception:
        return

    if "layer_head_mask" in MBartDecoderLayer.forward.__code__.co_varnames:
        _MBART_PATCHED = True
        return

    original_forward = MBartDecoderLayer.forward

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=True,
        cache_position=None,
        **kwargs,
    ):
        outputs = original_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        if isinstance(outputs, tuple):
            outputs = outputs + (past_key_values,)
        return outputs

    MBartDecoderLayer.forward = forward
    _MBART_PATCHED = True


def unload_model(obj):
    # Usage pattern: `obj = unload_model(obj)` to actually drop references in the caller.
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def parse_page_spec(spec: str, page_count: int) -> list[int]:
    spec = (spec or "").strip().lower()
    if not spec or spec in {"all", "*"}:
        return list(range(page_count))

    indices: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str) if start_str else 1
            end = int(end_str) if end_str else page_count
            for i in range(start, end + 1):
                if 1 <= i <= page_count:
                    indices.add(i - 1)
        else:
            i = int(part)
            if 1 <= i <= page_count:
                indices.add(i - 1)

    if not indices:
        raise ValueError("No valid pages selected.")
    return sorted(indices)


def render_pdf_to_images(pdf_path: str, page_indices: list[int], dpi: int = 200):
    doc = fitz.open(pdf_path)
    try:
        for idx in page_indices:
            page = doc.load_page(idx)
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                img = img.convert("RGB")
            yield idx + 1, img
    finally:
        doc.close()


def strip_think_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def load_glm_ocr_model(model_id: str):
    dtype = detect_dtype()
    device_map = "auto" if torch.cuda.is_available() else None

    model_cls = None
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore

        model_cls = AutoModelForImageTextToText
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            model_cls = AutoModelForVision2Seq
        except Exception:
            model_cls = AutoModel

    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def ocr_image_glm(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "page.png"
        image.save(img_path, format="PNG")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(img_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        inputs.pop("token_type_ids", None)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        in_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[0][in_len:]
        text = processor.decode(out_ids, skip_special_tokens=False)
        return strip_think_answer(text)


def load_nemotron_ocr_model(model_id: str):
    dtype = detect_dtype()
    ensure_c_radio_modules()
    ensure_mbart_layer_head_mask()

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.use_cache = False
    return model, processor, generation_config


def ocr_image_nemotron(model, processor, generation_config, image: Image.Image, max_new_tokens: int) -> str:
    inputs = processor(
        images=[image],
        text=NEMOTRON_TASK_PROMPT,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    generated_text = remove_nemotron_formatting(generated_text)
    classes, _bboxes, texts = extract_classes_bboxes(generated_text)
    if not texts:
        return generated_text.strip()

    processed_texts = []
    for cls, text in zip(classes, texts):
        processed_texts.append(
            postprocess_text(
                text,
                cls=cls,
                table_format="latex",
                text_format="markdown",
                blank_text_in_figures=False,
            )
        )
    return "\n".join(t.strip() for t in processed_texts if t and t.strip())


def clean_text_for_tts(text: str) -> str:
    text = convert_mmd_to_plain_text_ours(text)
    # Keep paragraph breaks for readability in the UI; chunking will still
    # normalize whitespace before synthesis.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_language(text: str) -> str:
    # Best-effort heuristic when user selects "Auto". Only used if the model
    # doesn't accept "Auto" as a language label.
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    if re.search(r"[\u0400-\u04ff]", text):
        return "Russian"
    return "English"


def load_qwen_tts_model(model_id: str):
    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "qwen-tts is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    dtype = detect_dtype()
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    kwargs: dict = {"device_map": device_map, "dtype": dtype}
    if torch.cuda.is_available() and has_flash_attention() and dtype in (torch.float16, torch.bfloat16):
        kwargs["attn_implementation"] = "flash_attention_2"

    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def generate_qwen_tts_audio(
    model,
    text: str,
    speaker: str,
    language: str,
    instruct: str,
    max_chars: int,
    silence_seconds: float,
    gen_kwargs: dict,
    clean_text: bool = True,
):
    if clean_text:
        text = clean_text_for_tts(text)
    else:
        text = re.sub(r"\s+", " ", text).strip()

    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("No text to synthesize.")

    supported_speakers: list[str] | None
    supported_languages: list[str] | None
    try:
        supported_speakers = list(model.get_supported_speakers())
    except Exception:
        supported_speakers = None
    try:
        supported_languages = list(model.get_supported_languages())
    except Exception:
        supported_languages = None

    if supported_speakers and speaker not in supported_speakers:
        match = next((s for s in supported_speakers if s.lower() == speaker.lower()), None)
        if match:
            speaker = match
        else:
            preview = ", ".join(supported_speakers[:20])
            suffix = "..." if len(supported_speakers) > 20 else ""
            raise ValueError(f"Unknown speaker '{speaker}'. Supported: {preview}{suffix}")

    # Expand speaker/language/instruct to match batch usage.
    speakers = [speaker] * len(chunks)
    instructs = [instruct] * len(chunks)

    langs: list[str]
    if (language or "").strip().lower() == "auto":
        inferred = [_infer_language(c) for c in chunks]
        if supported_languages:
            supported_set = {l for l in supported_languages}
            fixed: list[str] = []
            for lang in inferred:
                if lang in supported_set:
                    fixed.append(lang)
                    continue
                match = next((l for l in supported_languages if l.lower() == lang.lower()), None)
                if match:
                    fixed.append(match)
                    continue
                if "English" in supported_set:
                    fixed.append("English")
                else:
                    fixed.append(supported_languages[0])
            langs = fixed
        else:
            langs = inferred
    else:
        if supported_languages and language not in supported_languages:
            match = next((l for l in supported_languages if l.lower() == language.lower()), None)
            if match:
                language = match
            else:
                preview = ", ".join(supported_languages[:20])
                suffix = "..." if len(supported_languages) > 20 else ""
                raise ValueError(f"Unknown language '{language}'. Supported: {preview}{suffix}")
        langs = [language] * len(chunks)

    with torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=chunks,
            speaker=speakers,
            language=langs,
            instruct=instructs,
            **gen_kwargs,
        )

    audio_segments: list[np.ndarray] = []
    for wav in wavs:
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio_segments.append(arr)

    if not audio_segments:
        raise RuntimeError("TTS produced no audio.")

    if len(audio_segments) == 1:
        audio = audio_segments[0]
    else:
        silence = np.zeros(int(float(sr) * float(silence_seconds)), dtype=np.float32)
        stitched: list[np.ndarray] = []
        for idx, seg in enumerate(audio_segments):
            if idx > 0:
                stitched.append(silence)
            stitched.append(seg)
        audio = np.concatenate(stitched, axis=0)

    if not np.isfinite(audio).all():
        raise RuntimeError("TTS audio contains NaN/Inf values.")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio * (0.99 / peak)

    return audio.astype(np.float32, copy=False), int(sr)


def write_wav_pcm16(path: Path, audio: np.ndarray, sr: int):
    # winsound is picky; always write PCM_16 WAV.
    sf.write(str(path), audio, sr, subtype="PCM_16")


class OCRTTSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = AppState()

        self.ocr_thread: threading.Thread | None = None
        self.tts_thread: threading.Thread | None = None

        self.status_var = tk.StringVar(value="Idle")
        self.ocr_saved_var = tk.StringVar(value="")
        self.tts_saved_var = tk.StringVar(value="")

        self.root.title("PDF OCR + TTS (GLM-OCR + Qwen3-TTS)")
        self.root.geometry("1080x760")

        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        status_bar = ttk.Frame(main)
        status_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        status_bar.columnconfigure(1, weight=1)
        ttk.Label(status_bar, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Label(status_bar, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        self.ocr_tab = ttk.Frame(self.notebook, padding=10)
        self.tts_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.ocr_tab, text="OCR")
        self.notebook.add(self.tts_tab, text="TTS")

        self._build_ocr_tab()
        self._build_tts_tab()

    # -------------------------
    # Shared UI helpers
    # -------------------------
    def set_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    # -------------------------
    # OCR tab
    # -------------------------
    def _build_ocr_tab(self):
        tab = self.ocr_tab
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)

        cfg = ttk.Labelframe(tab, text="OCR Settings", padding=10)
        cfg.grid(row=0, column=0, columnspan=2, sticky="ew")
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="PDF File").grid(row=0, column=0, sticky="w")
        self.pdf_entry = ttk.Entry(cfg)
        self.pdf_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(cfg, text="Browse", command=self.browse_pdf).grid(row=0, column=2, sticky="ew")

        ttk.Label(cfg, text="Pages (e.g. 1-3,5 or all)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.pages_entry = ttk.Entry(cfg, width=24)
        self.pages_entry.insert(0, "all")
        self.pages_entry.grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(cfg, text="Render DPI").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.dpi_entry = ttk.Entry(cfg, width=10)
        self.dpi_entry.insert(0, "200")
        self.dpi_entry.grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(cfg, text="OCR Engine").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.ocr_engine_var = tk.StringVar(value=DEFAULT_OCR_ENGINE if DEFAULT_OCR_ENGINE in {OCR_ENGINE_GLM, OCR_ENGINE_NEMOTRON} else OCR_ENGINE_GLM)
        self.ocr_engine_combo = ttk.Combobox(
            cfg,
            textvariable=self.ocr_engine_var,
            values=[OCR_ENGINE_GLM, OCR_ENGINE_NEMOTRON],
            state="readonly",
            width=24,
        )
        self.ocr_engine_combo.grid(row=3, column=1, sticky="w", pady=(6, 0))
        self.ocr_engine_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_ocr_engine_change())

        ttk.Label(cfg, text="OCR Model Id").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.ocr_model_entry = ttk.Entry(cfg)
        self.ocr_model_entry.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(6, 0))

        ttk.Label(cfg, text="Max New Tokens").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.ocr_max_tokens_entry = ttk.Entry(cfg, width=10)
        self.ocr_max_tokens_entry.grid(row=5, column=1, sticky="w", pady=(6, 0))

        ttk.Label(cfg, text="GLM Prompt").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.glm_prompt_mode_var = tk.StringVar(value="Text Recognition")
        self.glm_prompt_mode_combo = ttk.Combobox(
            cfg,
            textvariable=self.glm_prompt_mode_var,
            values=["Text Recognition", "Formula Recognition", "Table Recognition", "Custom"],
            state="readonly",
            width=24,
        )
        self.glm_prompt_mode_combo.grid(row=6, column=1, sticky="w", pady=(6, 0))
        self.glm_prompt_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_glm_prompt_mode_change())

        self.glm_custom_prompt_entry = ttk.Entry(cfg)
        self.glm_custom_prompt_entry.grid(row=6, column=2, sticky="ew", pady=(6, 0))
        self.glm_custom_prompt_entry.insert(0, DEFAULT_GLM_PROMPT)

        btns = ttk.Frame(cfg)
        btns.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.run_ocr_button = ttk.Button(btns, text="Run OCR", command=self.run_ocr)
        self.run_ocr_button.grid(row=0, column=0, sticky="w")

        ttk.Label(tab, textvariable=self.ocr_saved_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 6))

        out_frame = ttk.Labelframe(tab, text="OCR Output", padding=8)
        out_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)

        self.ocr_output = tk.Text(out_frame, wrap="word")
        self.ocr_output.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(out_frame, command=self.ocr_output.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.ocr_output.configure(yscrollcommand=scroll.set)

        # Initialize defaults based on engine.
        self._on_ocr_engine_change()

    def _on_ocr_engine_change(self):
        engine = self.ocr_engine_var.get().strip()
        if engine == OCR_ENGINE_NEMOTRON:
            self.ocr_model_entry.delete(0, tk.END)
            self.ocr_model_entry.insert(0, DEFAULT_OCR_MODEL_ID_NEMOTRON)
            self.ocr_max_tokens_entry.delete(0, tk.END)
            self.ocr_max_tokens_entry.insert(0, str(DEFAULT_NEMOTRON_MAX_NEW_TOKENS))
            self.glm_prompt_mode_combo.state(["disabled"])
            self.glm_custom_prompt_entry.state(["disabled"])
        else:
            self.ocr_engine_var.set(OCR_ENGINE_GLM)
            self.ocr_model_entry.delete(0, tk.END)
            self.ocr_model_entry.insert(0, DEFAULT_OCR_MODEL_ID_GLM)
            self.ocr_max_tokens_entry.delete(0, tk.END)
            self.ocr_max_tokens_entry.insert(0, str(DEFAULT_GLM_MAX_NEW_TOKENS))
            self.glm_prompt_mode_combo.state(["!disabled"])
            self._on_glm_prompt_mode_change()

    def _on_glm_prompt_mode_change(self):
        mode = self.glm_prompt_mode_var.get().strip()
        if mode == "Custom":
            self.glm_custom_prompt_entry.state(["!disabled"])
        else:
            self.glm_custom_prompt_entry.state(["disabled"])

    def _get_glm_prompt(self) -> str:
        mode = self.glm_prompt_mode_var.get().strip()
        mapping = {
            "Text Recognition": "Text Recognition:",
            "Formula Recognition": "Formula Recognition:",
            "Table Recognition": "Table Recognition:",
        }
        if mode == "Custom":
            custom = self.glm_custom_prompt_entry.get().strip()
            return custom or DEFAULT_GLM_PROMPT
        return mapping.get(mode, DEFAULT_GLM_PROMPT)

    def browse_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if path:
            self.pdf_entry.delete(0, tk.END)
            self.pdf_entry.insert(0, path)
            self.state.pdf_path = path

    def _append_ocr_output(self, text: str):
        self.ocr_output.insert(tk.END, text)
        self.ocr_output.see(tk.END)
        self.root.update_idletasks()

    def run_ocr(self):
        if self.ocr_thread and self.ocr_thread.is_alive():
            messagebox.showinfo("Busy", "OCR is already running.")
            return

        pdf_path = self.pdf_entry.get().strip()
        if not pdf_path:
            messagebox.showerror("Missing PDF", "Please choose a PDF file.")
            return
        if not Path(pdf_path).exists():
            messagebox.showerror("Missing PDF", "The selected PDF does not exist.")
            return

        try:
            page_spec = self.pages_entry.get().strip()
            dpi = int(self.dpi_entry.get().strip() or "200")
            engine = self.ocr_engine_var.get().strip() or OCR_ENGINE_GLM
            model_id = self.ocr_model_entry.get().strip() or (
                DEFAULT_OCR_MODEL_ID_GLM if engine == OCR_ENGINE_GLM else DEFAULT_OCR_MODEL_ID_NEMOTRON
            )
            max_new_tokens = int(self.ocr_max_tokens_entry.get().strip() or "0")
            glm_prompt = self._get_glm_prompt()
        except Exception:
            messagebox.showerror("Invalid Input", "DPI and Max New Tokens must be integers.")
            return

        if max_new_tokens <= 0:
            messagebox.showerror("Invalid Input", "Max New Tokens must be greater than 0.")
            return

        self.run_ocr_button.state(["disabled"])
        self.ocr_output.delete("1.0", tk.END)
        self.ocr_saved_var.set("")

        cfg = {
            "pdf_path": pdf_path,
            "page_spec": page_spec,
            "dpi": dpi,
            "engine": engine,
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
            "glm_prompt": glm_prompt,
        }
        self.ocr_thread = threading.Thread(target=self._ocr_worker, args=(cfg,), daemon=True)
        self.ocr_thread.start()

    def _ocr_worker(self, cfg: dict):
        try:
            self._ocr_impl(cfg)
        except Exception as exc:
            print(traceback.format_exc())
            msg = f"{type(exc).__name__}: {exc}" if str(exc) else repr(exc)
            self.root.after(0, lambda m=msg: messagebox.showerror("OCR Error", m))
            self.root.after(0, lambda: self.set_status("OCR failed."))
        finally:
            self.root.after(0, lambda: self.run_ocr_button.state(["!disabled"]))

    def _ocr_impl(self, cfg: dict):
        pdf_path = cfg["pdf_path"]
        page_spec = cfg["page_spec"]
        dpi = cfg["dpi"]
        engine = cfg["engine"]
        model_id = cfg["model_id"]
        max_new_tokens = cfg["max_new_tokens"]
        glm_prompt = cfg["glm_prompt"]

        self.root.after(0, lambda: self.set_status("Loading OCR model..."))

        if engine == OCR_ENGINE_NEMOTRON:
            ocr_model, ocr_processor, ocr_generation_config = load_nemotron_ocr_model(model_id)
        else:
            ocr_model, ocr_processor = load_glm_ocr_model(model_id)
            ocr_generation_config = None

        doc = fitz.open(pdf_path)
        page_indices = parse_page_spec(page_spec, doc.page_count)
        doc.close()

        ocr_results: list[tuple[int, str]] = []
        for page_number, image in render_pdf_to_images(pdf_path, page_indices, dpi=dpi):
            self.root.after(0, lambda p=page_number: self.set_status(f"OCR page {p}..."))
            if engine == OCR_ENGINE_NEMOTRON:
                text = ocr_image_nemotron(ocr_model, ocr_processor, ocr_generation_config, image, max_new_tokens)
            else:
                text = ocr_image_glm(ocr_model, ocr_processor, image, glm_prompt, max_new_tokens)
            ocr_results.append((page_number, text))
            self.root.after(0, lambda p=page_number, t=text: self._append_ocr_output(f"\n\n# Page {p}\n{t}\n"))

        ocr_text = "\n\n".join([f"# Page {p}\n{t}" for p, t in ocr_results])
        self.state.ocr_text = ocr_text

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_path = output_dir / f"{Path(pdf_path).stem}_ocr_{timestamp}.md"
        text_path.write_text(ocr_text, encoding="utf-8")
        self.state.ocr_text_path = str(text_path)

        self.root.after(0, lambda: self.set_status("Unloading OCR model..."))
        ocr_model = unload_model(ocr_model)
        ocr_processor = unload_model(ocr_processor)
        ocr_generation_config = unload_model(ocr_generation_config)

        self.root.after(0, lambda p=text_path: self.ocr_saved_var.set(f"Saved OCR: {p} (engine: {engine}, model: {model_id})"))
        self.root.after(0, lambda: self.set_status("OCR done."))

    # -------------------------
    # TTS tab
    # -------------------------
    def _build_tts_tab(self):
        tab = self.tts_tab
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)

        cfg = ttk.Labelframe(tab, text="TTS Settings (Qwen3 CustomVoice)", padding=10)
        cfg.grid(row=0, column=0, columnspan=2, sticky="ew")
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="TTS Model Id").grid(row=0, column=0, sticky="w")
        self.tts_model_entry = ttk.Entry(cfg)
        self.tts_model_entry.insert(0, DEFAULT_TTS_MODEL_ID)
        self.tts_model_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0))

        ttk.Label(cfg, text="Speaker").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.tts_speaker_var = tk.StringVar(value=DEFAULT_TTS_SPEAKER)
        self.tts_speaker_combo = ttk.Combobox(cfg, textvariable=self.tts_speaker_var, values=[DEFAULT_TTS_SPEAKER], width=24)
        self.tts_speaker_combo.grid(row=1, column=1, sticky="w", pady=(6, 0), padx=(8, 0))

        ttk.Label(cfg, text="Language").grid(row=1, column=2, sticky="w", pady=(6, 0), padx=(12, 0))
        self.tts_language_var = tk.StringVar(value=DEFAULT_TTS_LANGUAGE)
        self.tts_language_combo = ttk.Combobox(cfg, textvariable=self.tts_language_var, values=[DEFAULT_TTS_LANGUAGE], width=18)
        self.tts_language_combo.grid(row=1, column=3, sticky="w", pady=(6, 0), padx=(8, 0))

        ttk.Label(cfg, text="Instruct (optional)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.tts_instruct_entry = ttk.Entry(cfg)
        self.tts_instruct_entry.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(6, 0), padx=(8, 0))

        ttk.Label(cfg, text="Chunk Max Chars").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.tts_max_chars_entry = ttk.Entry(cfg, width=10)
        self.tts_max_chars_entry.insert(0, str(DEFAULT_TTS_MAX_CHARS))
        self.tts_max_chars_entry.grid(row=3, column=1, sticky="w", pady=(6, 0), padx=(8, 0))

        ttk.Label(cfg, text="Silence Seconds").grid(row=3, column=2, sticky="w", pady=(6, 0), padx=(12, 0))
        self.tts_silence_entry = ttk.Entry(cfg, width=10)
        self.tts_silence_entry.insert(0, str(DEFAULT_TTS_SILENCE_SECONDS))
        self.tts_silence_entry.grid(row=3, column=3, sticky="w", pady=(6, 0), padx=(8, 0))

        adv = ttk.Labelframe(cfg, text="Generation Options", padding=8)
        adv.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        adv.columnconfigure(1, weight=1)

        ttk.Label(adv, text="max_new_tokens").grid(row=0, column=0, sticky="w")
        self.tts_max_new_tokens_entry = ttk.Entry(adv, width=10)
        self.tts_max_new_tokens_entry.insert(0, str(DEFAULT_TTS_MAX_NEW_TOKENS))
        self.tts_max_new_tokens_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(adv, text="top_p").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.tts_top_p_entry = ttk.Entry(adv, width=10)
        self.tts_top_p_entry.insert(0, str(DEFAULT_TTS_TOP_P))
        self.tts_top_p_entry.grid(row=0, column=3, sticky="w", padx=(8, 0))

        ttk.Label(adv, text="temperature").grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.tts_temp_entry = ttk.Entry(adv, width=10)
        self.tts_temp_entry.insert(0, str(DEFAULT_TTS_TEMPERATURE))
        self.tts_temp_entry.grid(row=0, column=5, sticky="w", padx=(8, 0))

        actions = ttk.Frame(tab)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        actions.columnconfigure(1, weight=1)

        self.tts_clean_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(actions, text="Clean text for TTS (recommended)", variable=self.tts_clean_var).grid(row=0, column=0, sticky="w")

        ttk.Button(actions, text="Load From OCR Output", command=self.load_from_ocr).grid(row=0, column=1, sticky="e")

        ttk.Label(tab, text="Source Text").grid(row=2, column=0, sticky="w")
        self.tts_text = tk.Text(tab, wrap="word", height=10)
        self.tts_text.grid(row=3, column=0, columnspan=2, sticky="nsew")
        tts_scroll = ttk.Scrollbar(tab, command=self.tts_text.yview)
        tts_scroll.grid(row=3, column=2, sticky="ns")
        self.tts_text.configure(yscrollcommand=tts_scroll.set)

        tts_btns = ttk.Frame(tab)
        tts_btns.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.generate_tts_button = ttk.Button(tts_btns, text="Generate Audio", command=self.generate_audio)
        self.generate_tts_button.grid(row=0, column=0, sticky="w")

        self.play_button = ttk.Button(tts_btns, text="Play", command=self.play_audio)
        self.play_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.play_button.state(["disabled"])

        self.stop_button = ttk.Button(tts_btns, text="Stop", command=self.stop_audio)
        self.stop_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.stop_button.state(["disabled"])

        ttk.Label(tab, textvariable=self.tts_saved_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        if sys.platform != "win32":
            self.play_button.state(["disabled"])
            self.stop_button.state(["disabled"])

    def load_from_ocr(self):
        if not self.state.ocr_text:
            messagebox.showinfo("No OCR Output", "Run OCR first (OCR tab) before loading text.")
            return

        text = self.state.ocr_text
        if self.tts_clean_var.get():
            text = clean_text_for_tts(text)

        self.tts_text.delete("1.0", tk.END)
        self.tts_text.insert("1.0", text)
        self.notebook.select(self.tts_tab)

    def generate_audio(self):
        if self.tts_thread and self.tts_thread.is_alive():
            messagebox.showinfo("Busy", "TTS is already running.")
            return

        text = self.tts_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Missing Text", "Please enter text (or load from OCR output).")
            return

        try:
            model_id = self.tts_model_entry.get().strip() or DEFAULT_TTS_MODEL_ID
            speaker = self.tts_speaker_var.get().strip() or DEFAULT_TTS_SPEAKER
            language = self.tts_language_var.get().strip() or DEFAULT_TTS_LANGUAGE
            instruct = self.tts_instruct_entry.get().strip()
            max_chars = int(self.tts_max_chars_entry.get().strip() or "0")
            silence_seconds = float(self.tts_silence_entry.get().strip() or "0")
            gen_kwargs = {
                "max_new_tokens": int(self.tts_max_new_tokens_entry.get().strip() or "0"),
                "top_p": float(self.tts_top_p_entry.get().strip() or "0"),
                "temperature": float(self.tts_temp_entry.get().strip() or "0"),
            }
            clean_text = bool(self.tts_clean_var.get())
        except Exception:
            messagebox.showerror("Invalid Input", "Chunk/Gen options must be numeric (ints/floats).")
            return

        if max_chars <= 0:
            messagebox.showerror("Invalid Input", "Chunk Max Chars must be greater than 0.")
            return

        pdf_path = (self.pdf_entry.get().strip() or self.state.pdf_path or "tts").strip()
        base = Path(pdf_path).stem if pdf_path else "tts"

        self.generate_tts_button.state(["disabled"])
        self.play_button.state(["disabled"])
        self.stop_button.state(["disabled"])
        self.tts_saved_var.set("")

        cfg = {
            "text": text,
            "model_id": model_id,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "max_chars": max_chars,
            "silence_seconds": silence_seconds,
            "gen_kwargs": gen_kwargs,
            "clean_text": clean_text,
            "output_base": base,
        }
        self.tts_thread = threading.Thread(target=self._tts_worker, args=(cfg,), daemon=True)
        self.tts_thread.start()

    def _tts_worker(self, cfg: dict):
        try:
            self._tts_impl(cfg)
        except Exception as exc:
            print(traceback.format_exc())
            msg = f"{type(exc).__name__}: {exc}" if str(exc) else repr(exc)
            self.root.after(0, lambda m=msg: messagebox.showerror("TTS Error", m))
            self.root.after(0, lambda: self.set_status("TTS failed."))
        finally:
            self.root.after(0, lambda: self.generate_tts_button.state(["!disabled"]))

    def _tts_impl(self, cfg: dict):
        text = cfg["text"]
        model_id = cfg["model_id"]
        speaker = cfg["speaker"]
        language = cfg["language"]
        instruct = cfg["instruct"]
        max_chars = cfg["max_chars"]
        silence_seconds = cfg["silence_seconds"]
        gen_kwargs = cfg["gen_kwargs"]
        clean_text = cfg["clean_text"]
        base = cfg["output_base"]

        self.root.after(0, lambda: self.set_status("Loading TTS model..."))
        tts_model = load_qwen_tts_model(model_id)

        # Populate voice/language lists (if supported by this qwen-tts version).
        try:
            speakers = list(tts_model.get_supported_speakers())
        except Exception:
            speakers = []
        try:
            langs = list(tts_model.get_supported_languages())
        except Exception:
            langs = []

        def _apply_lists():
            if speakers:
                self.tts_speaker_combo["values"] = speakers
                if self.tts_speaker_var.get().strip() not in speakers:
                    self.tts_speaker_var.set(DEFAULT_TTS_SPEAKER if DEFAULT_TTS_SPEAKER in speakers else speakers[0])
            if langs:
                values = ["Auto"] + [l for l in langs if l != "Auto"]
                self.tts_language_combo["values"] = values
                if self.tts_language_var.get().strip() not in values:
                    self.tts_language_var.set("Auto")

        self.root.after(0, _apply_lists)

        self.root.after(0, lambda: self.set_status("Generating speech..."))
        audio, sr = generate_qwen_tts_audio(
            tts_model,
            text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            max_chars=max_chars,
            silence_seconds=silence_seconds,
            gen_kwargs=gen_kwargs,
            clean_text=clean_text,
        )

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        audio_path = output_dir / f"{base}_tts_{timestamp}.wav"
        write_wav_pcm16(audio_path, audio, sr)

        self.state.audio_path = str(audio_path)
        self.state.audio_sr = int(sr)

        self.root.after(0, lambda p=audio_path, s=sr: self.tts_saved_var.set(f"Saved audio: {p} (sr: {s}, model: {model_id})"))
        self.root.after(0, lambda: self.set_status("TTS done."))

        if sys.platform == "win32":
            self.root.after(0, lambda: self.play_button.state(["!disabled"]))
            self.root.after(0, lambda: self.stop_button.state(["!disabled"]))

        tts_model = unload_model(tts_model)

    # -------------------------
    # Playback
    # -------------------------
    def play_audio(self):
        if sys.platform != "win32":
            messagebox.showinfo("Playback", "In-app playback is supported on Windows only.")
            return
        if not self.state.audio_path:
            messagebox.showinfo("No audio", "Generate audio first.")
            return
        try:
            import winsound

            winsound.PlaySound(self.state.audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            messagebox.showerror("Playback Error", str(exc))

    def stop_audio(self):
        if sys.platform != "win32":
            return
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


def run_smoke_test():
    import importlib

    print("Smoke test: imports")
    importlib.import_module("fitz")
    importlib.import_module("torch")
    importlib.import_module("transformers")
    importlib.import_module("soundfile")
    importlib.import_module("postprocessing")
    importlib.import_module("bs4")

    print("Smoke test: qwen-tts import")
    try:
        importlib.import_module("qwen_tts")
        print("qwen_tts_ok")
    except Exception as exc:
        print("qwen_tts_missing", repr(exc))

    print("Smoke test: GPU")
    print("cuda_available", torch.cuda.is_available())

    pdf_path = Path("test.pdf")
    if pdf_path.exists():
        print("Smoke test: render first page")
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=72, colorspace=fitz.csRGB)
        _ = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        print("render_ok")

    print("Smoke test: ok")


def main():
    if "--smoke" in sys.argv:
        run_smoke_test()
        return

    root = tk.Tk()
    app = OCRTTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

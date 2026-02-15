import gc
import os
import re
import shutil
import sys
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
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    VibeVoiceForConditionalGeneration,
)
from transformers.dynamic_module_utils import _sanitize_module_name, create_dynamic_module
from transformers.utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME

from postprocessing import extract_classes_bboxes, postprocess_text, remove_nemotron_formatting

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_OCR_MODEL_ID = os.getenv("OCR_MODEL_ID", "nvidia/NVIDIA-Nemotron-Parse-v1.1")
DEFAULT_TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "microsoft/VibeVoice-1.5B-hf")
FALLBACK_TTS_MODEL_ID = os.getenv("TTS_MODEL_ID_FALLBACK", "bezzam/VibeVoice-1.5B")

NEMOTRON_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"
TTS_SAMPLE_RATE = 24000


@dataclass
class AppState:
    pdf_path: str | None = None
    ocr_text: str = ""
    audio_path: str | None = None


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


def load_ocr_model(model_id: str):
    dtype = detect_dtype()
    ensure_c_radio_modules()
    ensure_mbart_layer_head_mask()
    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
    generation_config.use_cache = False
    return model, processor, generation_config


def unload_model(model):
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_tts_model(primary_id: str, fallback_id: str | None = None):
    dtype = detect_dtype()
    last_error = None
    for model_id in [primary_id, fallback_id]:
        if not model_id:
            continue
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            text_cfg = getattr(config, "text_config", None)
            if text_cfg and getattr(config, "hidden_size", None) is None:
                config.hidden_size = getattr(text_cfg, "hidden_size", None)
            acoustic_cfg = getattr(config, "acoustic_tokenizer_config", None)
            if acoustic_cfg and getattr(acoustic_cfg, "decoder_depths", None) is None:
                depths = getattr(acoustic_cfg, "depths", None)
                if depths:
                    acoustic_cfg.decoder_depths = list(reversed(depths))
            diff_cfg = getattr(config, "diffusion_head_config", None)
            if diff_cfg and text_cfg:
                target_hidden = getattr(text_cfg, "hidden_size", None)
                if target_hidden:
                    diff_cfg.hidden_size = target_hidden
                for attr in ("num_head_layers", "frequency_embedding_size", "hidden_act", "rms_norm_eps", "mlp_bias"):
                    if hasattr(config, attr):
                        setattr(diff_cfg, attr, getattr(config, attr))
                if hasattr(config, "intermediate_size") and target_hidden:
                    ratio = config.intermediate_size / float(target_hidden)
                    if ratio.is_integer():
                        diff_cfg.head_ffn_ratio = int(ratio)

            model = VibeVoiceForConditionalGeneration.from_pretrained(
                model_id,
                config=config,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                extra_special_tokens={},
            )
            return model_id, model, processor
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Failed to load VibeVoice model: {last_error}")


def ocr_image(model, processor, generation_config, image: Image.Image, max_new_tokens: int):
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


def generate_tts_audio(
    model,
    processor,
    text: str,
    max_new_tokens: int,
    max_chars: int = 1200,
    silence_seconds: float = 0.25,
):
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("No text to synthesize.")

    audio_segments = []
    for chunk in chunks:
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": chunk}]},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )
        audio = outputs.audio[0].detach().float().cpu().numpy()
        audio = np.squeeze(audio)
        audio_segments.append(audio)

    if len(audio_segments) == 1:
        return audio_segments[0]

    silence = np.zeros(int(TTS_SAMPLE_RATE * silence_seconds), dtype=audio_segments[0].dtype)
    stitched = []
    for idx, seg in enumerate(audio_segments):
        if idx > 0:
            stitched.append(silence)
        stitched.append(seg)
    return np.concatenate(stitched)


class OCRTTSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = AppState()
        self.thread: threading.Thread | None = None

        self.root.title("PDF OCR + TTS (NuMarkdown + VibeVoice)")
        self.root.geometry("980x720")

        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(7, weight=1)

        ttk.Label(main, text="PDF File").grid(row=0, column=0, sticky="w")
        self.pdf_entry = ttk.Entry(main)
        self.pdf_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(main, text="Browse", command=self.browse_pdf).grid(row=0, column=2)

        ttk.Label(main, text="Pages (e.g. 1-3,5 or all)").grid(row=1, column=0, sticky="w")
        self.pages_entry = ttk.Entry(main)
        self.pages_entry.insert(0, "all")
        self.pages_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(main, text="Render DPI").grid(row=2, column=0, sticky="w")
        self.dpi_entry = ttk.Entry(main, width=10)
        self.dpi_entry.insert(0, "200")
        self.dpi_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(main, text="OCR model id").grid(row=3, column=0, sticky="w")
        self.ocr_model_entry = ttk.Entry(main)
        self.ocr_model_entry.insert(0, DEFAULT_OCR_MODEL_ID)
        self.ocr_model_entry.grid(row=3, column=1, sticky="ew")

        ttk.Label(main, text="TTS model id").grid(row=4, column=0, sticky="w")
        self.tts_model_entry = ttk.Entry(main)
        self.tts_model_entry.insert(0, DEFAULT_TTS_MODEL_ID)
        self.tts_model_entry.grid(row=4, column=1, sticky="ew")

        ttk.Label(main, text="TTS fallback id").grid(row=5, column=0, sticky="w")
        self.tts_fallback_entry = ttk.Entry(main)
        self.tts_fallback_entry.insert(0, FALLBACK_TTS_MODEL_ID)
        self.tts_fallback_entry.grid(row=5, column=1, sticky="ew")

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=0, column=3, rowspan=6, sticky="ns")
        self.run_button = ttk.Button(btn_frame, text="Run OCR + Read", command=self.run_pipeline)
        self.run_button.grid(row=0, column=0, pady=(0, 8))
        self.play_button = ttk.Button(btn_frame, text="Open Audio", command=self.open_audio)
        self.play_button.grid(row=1, column=0, pady=(0, 8))
        self.play_button.state(["disabled"])

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(main, text="Status").grid(row=6, column=0, sticky="nw")
        self.status_label = ttk.Label(main, textvariable=self.status_var)
        self.status_label.grid(row=6, column=1, sticky="nw")

        ttk.Label(main, text="OCR Output").grid(row=7, column=0, sticky="nw", pady=(8, 0))
        self.output = tk.Text(main, wrap="word")
        self.output.grid(row=7, column=1, columnspan=3, sticky="nsew", pady=(8, 0))
        scrollbar = ttk.Scrollbar(main, command=self.output.yview)
        scrollbar.grid(row=7, column=4, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)

    def browse_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if path:
            self.pdf_entry.delete(0, tk.END)
            self.pdf_entry.insert(0, path)
            self.state.pdf_path = path

    def set_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    def append_output(self, text: str):
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.root.update_idletasks()

    def run_pipeline(self):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        pdf_path = self.pdf_entry.get().strip()
        if not pdf_path:
            messagebox.showerror("Missing PDF", "Please choose a PDF file.")
            return
        if not Path(pdf_path).exists():
            messagebox.showerror("Missing PDF", "The selected PDF does not exist.")
            return

        self.run_button.state(["disabled"])
        self.play_button.state(["disabled"])
        self.output.delete("1.0", tk.END)

        self.thread = threading.Thread(target=self._run_worker, daemon=True)
        self.thread.start()

    def _run_worker(self):
        try:
            self._run_pipeline_impl()
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}" if str(exc) else repr(exc)
            print(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Error", msg))
        finally:
            self.root.after(0, lambda: self.run_button.state(["!disabled"]))

    def _run_pipeline_impl(self):
        pdf_path = self.pdf_entry.get().strip()
        page_spec = self.pages_entry.get().strip()
        dpi = int(self.dpi_entry.get().strip() or "200")
        ocr_model_id = self.ocr_model_entry.get().strip() or DEFAULT_OCR_MODEL_ID
        tts_model_id = self.tts_model_entry.get().strip() or DEFAULT_TTS_MODEL_ID
        tts_fallback = self.tts_fallback_entry.get().strip() or None

        self.root.after(0, lambda: self.set_status("Loading OCR model..."))
        ocr_model, ocr_processor, ocr_generation_config = load_ocr_model(ocr_model_id)

        doc = fitz.open(pdf_path)
        page_indices = parse_page_spec(page_spec, doc.page_count)
        doc.close()

        ocr_results = []
        max_new_tokens = 2048

        for page_number, image in render_pdf_to_images(pdf_path, page_indices, dpi=dpi):
            self.root.after(0, lambda p=page_number: self.set_status(f"OCR page {p}..."))
            text = ocr_image(ocr_model, ocr_processor, ocr_generation_config, image, max_new_tokens)
            ocr_results.append((page_number, text))
            self.root.after(0, lambda p=page_number, t=text: self.append_output(f"\n\n# Page {p}\n{t}\n"))

        ocr_text = "\n\n".join([f"# Page {p}\n{t}" for p, t in ocr_results])
        self.state.ocr_text = ocr_text

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_path = output_dir / f"{Path(pdf_path).stem}_ocr_{timestamp}.md"
        text_path.write_text(ocr_text, encoding="utf-8")

        self.root.after(0, lambda: self.set_status("Unloading OCR model..."))
        unload_model(ocr_model)
        unload_model(ocr_processor)
        ocr_generation_config = None
        ocr_model = None
        ocr_processor = None

        self.root.after(0, lambda: self.set_status("Loading TTS model..."))
        selected_tts_id, tts_model, tts_processor = load_tts_model(tts_model_id, tts_fallback)

        self.root.after(0, lambda: self.set_status("Generating speech..."))
        audio = generate_tts_audio(
            tts_model,
            tts_processor,
            ocr_text,
            max_new_tokens=512,
            max_chars=1200,
        )

        audio_path = output_dir / f"{Path(pdf_path).stem}_tts_{timestamp}.wav"
        sf.write(audio_path, audio, TTS_SAMPLE_RATE)

        self.state.audio_path = str(audio_path)
        self.root.after(0, lambda: self.set_status(f"Done. Saved: {audio_path} (model: {selected_tts_id})"))
        self.root.after(0, lambda: self.play_button.state(["!disabled"]))

        unload_model(tts_model)
        unload_model(tts_processor)
        tts_model = None
        tts_processor = None

    def open_audio(self):
        if not self.state.audio_path:
            messagebox.showinfo("No audio", "No audio file available yet.")
            return
        try:
            os.startfile(self.state.audio_path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


def run_smoke_test():
    import importlib

    print("Smoke test: imports")
    importlib.import_module("fitz")
    importlib.import_module("torch")
    importlib.import_module("transformers")
    importlib.import_module("soundfile")
    importlib.import_module("postprocessing")
    importlib.import_module("bs4")

    print("Smoke test: transformers classes")
    from transformers import VibeVoiceForConditionalGeneration  # noqa: F401

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

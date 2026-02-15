import argparse
import traceback
from datetime import datetime
from pathlib import Path

import app


def run_pipeline(args) -> tuple[Path, Path]:
    print("Opening PDF...")
    doc = app.fitz.open(args.pdf)
    page_indices = app.parse_page_spec(args.pages, doc.page_count)
    doc.close()

    ocr_engine = args.ocr_engine
    if not args.ocr_model:
        args.ocr_model = app.DEFAULT_OCR_MODEL_ID_GLM if ocr_engine == "glm" else app.DEFAULT_OCR_MODEL_ID_NEMOTRON
    if not args.ocr_max_new_tokens:
        args.ocr_max_new_tokens = app.DEFAULT_GLM_MAX_NEW_TOKENS if ocr_engine == "glm" else app.DEFAULT_NEMOTRON_MAX_NEW_TOKENS

    if ocr_engine == "glm":
        print("Loading OCR model (GLM-OCR)...")
        ocr_model, ocr_processor = app.load_glm_ocr_model(args.ocr_model)
        ocr_generation_config = None
    else:
        print("Loading OCR model (Nemotron-Parse)...")
        ocr_model, ocr_processor, ocr_generation_config = app.load_nemotron_ocr_model(args.ocr_model)

    ocr_results = []
    for page_number, image in app.render_pdf_to_images(args.pdf, page_indices, dpi=args.dpi):
        print(f"OCR page {page_number} (size={image.size})...")
        if ocr_engine == "glm":
            text = app.ocr_image_glm(ocr_model, ocr_processor, image, args.glm_prompt, args.ocr_max_new_tokens)
        else:
            text = app.ocr_image_nemotron(ocr_model, ocr_processor, ocr_generation_config, image, args.ocr_max_new_tokens)
        ocr_results.append((page_number, text))

    ocr_text = "\n\n".join([f"# Page {p}\n{t}" for p, t in ocr_results])

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    text_path = output_dir / f"{Path(args.pdf).stem}_ocr_{timestamp}.md"
    text_path.write_text(ocr_text, encoding="utf-8")
    print(f"Saved OCR text to {text_path}")

    print("Unloading OCR model...")
    ocr_model = app.unload_model(ocr_model)
    ocr_processor = app.unload_model(ocr_processor)
    ocr_generation_config = app.unload_model(ocr_generation_config)

    print("Loading TTS model (Qwen3)...")
    tts_model = app.load_qwen_tts_model(args.tts_model)
    print(f"Loaded TTS model: {args.tts_model}")

    print("Generating speech...")
    audio, sr = app.generate_qwen_tts_audio(
        tts_model,
        ocr_text,
        speaker=args.tts_speaker,
        language=args.tts_language,
        instruct=args.tts_instruct,
        max_chars=args.tts_max_chars,
        silence_seconds=args.tts_silence_seconds,
        gen_kwargs={
            "max_new_tokens": args.tts_max_new_tokens,
            "top_p": args.tts_top_p,
            "temperature": args.tts_temperature,
        },
        clean_text=not args.no_tts_clean,
    )

    audio_path = output_dir / f"{Path(args.pdf).stem}_tts_{timestamp}.wav"
    app.write_wav_pcm16(audio_path, audio, sr)
    print(f"Saved audio to {audio_path} (sr={sr})")

    tts_model = app.unload_model(tts_model)
    return text_path, audio_path


def main():
    parser = argparse.ArgumentParser(description="End-to-end OCR + TTS test runner.")
    parser.add_argument("--pdf", default="test.pdf", help="Path to PDF")
    parser.add_argument("--pages", default="all", help="Pages spec, e.g. 1-3,5 or all")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI")

    parser.add_argument("--ocr-engine", choices=["glm", "nemotron"], default="glm")
    parser.add_argument("--ocr-model", default="", help="OCR model id (defaults depend on engine)")
    parser.add_argument("--ocr-max-new-tokens", type=int, default=0)
    parser.add_argument("--glm-prompt", dest="glm_prompt", default=app.DEFAULT_GLM_PROMPT)

    parser.add_argument("--tts-model", default=app.DEFAULT_TTS_MODEL_ID, help="TTS model id")
    parser.add_argument("--tts-speaker", default=app.DEFAULT_TTS_SPEAKER)
    parser.add_argument("--tts-language", default=app.DEFAULT_TTS_LANGUAGE)
    parser.add_argument("--tts-instruct", default="")
    parser.add_argument("--tts-max-new-tokens", type=int, default=app.DEFAULT_TTS_MAX_NEW_TOKENS)
    parser.add_argument("--tts-top-p", type=float, default=app.DEFAULT_TTS_TOP_P)
    parser.add_argument("--tts-temperature", type=float, default=app.DEFAULT_TTS_TEMPERATURE)
    parser.add_argument("--tts-max-chars", type=int, default=app.DEFAULT_TTS_MAX_CHARS)
    parser.add_argument("--tts-silence-seconds", type=float, default=app.DEFAULT_TTS_SILENCE_SECONDS)
    parser.add_argument("--no-tts-clean", action="store_true", help="Disable text cleaning for TTS")

    args = parser.parse_args()

    try:
        run_pipeline(args)
    except Exception:
        print("ERROR: pipeline failed")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

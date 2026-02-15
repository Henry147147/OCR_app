import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

import app


def run_pipeline(args):
    print("Loading OCR model...")
    ocr_model, ocr_processor, ocr_generation_config = app.load_ocr_model(args.ocr_model)

    print("Opening PDF...")
    doc = app.fitz.open(args.pdf)
    page_indices = app.parse_page_spec(args.pages, doc.page_count)
    doc.close()

    ocr_results = []
    for page_number, image in app.render_pdf_to_images(args.pdf, page_indices, dpi=args.dpi):
        print(f"OCR page {page_number} (size={image.size})...")
        text = app.ocr_image(
            ocr_model,
            ocr_processor,
            ocr_generation_config,
            image,
            args.ocr_max_new_tokens,
        )
        ocr_results.append((page_number, text))

    ocr_text = "\n\n".join([f"# Page {p}\n{t}" for p, t in ocr_results])

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = output_dir / f"{Path(args.pdf).stem}_ocr_{timestamp}.md"
    text_path.write_text(ocr_text, encoding="utf-8")
    print(f"Saved OCR text to {text_path}")

    print("Unloading OCR model...")
    app.unload_model(ocr_model)
    app.unload_model(ocr_processor)
    ocr_generation_config = None

    print("Loading TTS model...")
    selected_tts_id, tts_model, tts_processor = app.load_tts_model(args.tts_model, args.tts_fallback)
    print(f"Loaded TTS model: {selected_tts_id}")

    print("Generating speech...")
    audio = app.generate_tts_audio(
        tts_model,
        tts_processor,
        ocr_text,
        max_new_tokens=args.tts_max_new_tokens,
        max_chars=args.tts_max_chars,
    )

    audio_path = output_dir / f"{Path(args.pdf).stem}_tts_{timestamp}.wav"
    sf.write(audio_path, audio, app.TTS_SAMPLE_RATE)
    print(f"Saved audio to {audio_path}")

    app.unload_model(tts_model)
    app.unload_model(tts_processor)


def main():
    parser = argparse.ArgumentParser(description="End-to-end OCR + TTS test runner.")
    parser.add_argument("--pdf", default="test.pdf", help="Path to PDF")
    parser.add_argument("--pages", default="all", help="Pages spec, e.g. 1-3,5 or all")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI")
    parser.add_argument("--ocr-model", default=app.DEFAULT_OCR_MODEL_ID, help="OCR model id")
    parser.add_argument("--tts-model", default=app.DEFAULT_TTS_MODEL_ID, help="TTS model id")
    parser.add_argument("--tts-fallback", default=app.FALLBACK_TTS_MODEL_ID, help="TTS fallback model id")
    parser.add_argument("--ocr-max-new-tokens", type=int, default=2048)
    parser.add_argument("--tts-max-new-tokens", type=int, default=512)
    parser.add_argument("--tts-max-chars", type=int, default=1200)
    args = parser.parse_args()

    try:
        run_pipeline(args)
    except Exception:
        print("ERROR: pipeline failed")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Transcribe one clip with Whisper. Runs inside an engine venv that has openai-whisper.

    python transcribe_worker.py <model.pt> <clip.wav> [language]

Prints one JSON line: {"text": ..., "language": ...}.
"""
import json
import sys


def main() -> None:
    model_path, clip = sys.argv[1], sys.argv[2]
    language = (sys.argv[3] if len(sys.argv) > 3 else "").strip().lower() or None

    import torch
    import whisper
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    if language and language not in LANGUAGES:
        language = TO_LANGUAGE_CODE.get(language)

    def run(device: str) -> dict:
        model = whisper.load_model(model_path, device=device)
        return model.transcribe(clip, language=language, fp16=device == "cuda")

    if torch.cuda.is_available():
        try:
            result = run("cuda")
        except RuntimeError:  # out of memory while a voice model or the LLM holds the GPU
            torch.cuda.empty_cache()
            result = run("cpu")
    else:
        result = run("cpu")
    print(json.dumps({"text": result["text"].strip(), "language": result.get("language")}))


if __name__ == "__main__":
    main()

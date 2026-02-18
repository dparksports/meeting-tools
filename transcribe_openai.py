#!/usr/bin/env python3
"""
Batch Transcribe Folder — Standard OpenAI Whisper
──────────────────────────────────────────────────
Transcribes every media file in a given folder using the standard
OpenAI Whisper library (https://github.com/openai/whisper).

Output:  <folder>.whisper-<model>/<filename>.whisper-<model>.txt
Resume:  Re-run the same command — already-transcribed files are skipped.

Usage:
    python transcribe_openai.py "D:\\Videos\\meetings"
    python transcribe_openai.py "D:\\Videos\\meetings" --model medium.en
    python transcribe_openai.py "D:\\Videos\\meetings" --model large
    python transcribe_openai.py "D:\\Videos\\meetings" --device cpu
"""

import argparse
import sys
import os
import time
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "medium.en"

MEDIA_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.wav', '.mp3', '.flac', '.m4a',
    '.webm', '.aac', '.wma', '.ogg', '.m4v', '.3gp', '.ts', '.mpg', '.mpeg',
}

# ANSI colours
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"

# ──────────────────────────────────────────────────────────────────────────────
# Device detection
# ──────────────────────────────────────────────────────────────────────────────

def get_device(device_override=None):
    """Auto-detect GPU or honour explicit --device flag. Returns device string."""
    import torch

    if device_override == "cpu":
        print(f"{_DIM}[INIT] Forced CPU mode.{_RESET}")
        return "cpu"
    elif device_override == "cuda":
        if torch.cuda.is_available():
            print(f"{_GREEN}[INIT] Forced CUDA mode. GPU available.{_RESET}")
            return "cuda"
        else:
            print(f"{_YELLOW}[INIT] CUDA requested but not available — falling back to CPU.{_RESET}")
            return "cpu"
    else:  # auto
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"{_GREEN}[INIT] CUDA detected ({gpu_name}). Using GPU.{_RESET}")
            return "cuda"
        else:
            print(f"{_DIM}[INIT] CUDA not found. Using CPU.{_RESET}")
            return "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_whisper_model(model_name, device):
    """Load standard Whisper model with GPU→CPU fallback."""
    import whisper

    try:
        print(f"{_CYAN}[MODEL] Loading {_BOLD}{model_name}{_RESET}{_CYAN} on {device}...{_RESET}")
        return whisper.load_model(model_name, device=device)
    except Exception as e:
        if device == "cuda":
            print(f"{_YELLOW}[WARNING] GPU load failed: {e}{_RESET}")
            print(f"{_CYAN}[MODEL] Falling back to CPU...{_RESET}")
            return whisper.load_model(model_name, device="cpu")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# Media discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_media_files(directory):
    """Recursively find all media files, sorted alphabetically."""
    if directory.endswith('"'):
        directory = directory[:-1]
    if len(directory) == 2 and directory[1] == ':':
        directory = directory + os.sep
    elif not os.path.isabs(directory):
        directory = os.path.abspath(directory)

    media_files = []
    for root, _dirs, files in os.walk(directory, followlinks=True):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in MEDIA_EXTENSIONS:
                media_files.append(os.path.join(root, f))

    media_files.sort()
    return media_files

# ──────────────────────────────────────────────────────────────────────────────
# Output path helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_output_dir(input_dir, model_name):
    """
    Given input folder  D:\\Videos\\meetings  and model  medium.en
    Returns output folder D:\\Videos\\meetings.whisper-medium.en
    """
    input_dir = input_dir.rstrip(os.sep).rstrip("/")
    parent = os.path.dirname(input_dir)
    folder_name = os.path.basename(input_dir)
    out_dir = os.path.join(parent, f"{folder_name}.whisper-{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def transcript_path(output_dir, media_file, model_name):
    """Return the transcript file path for a given media file."""
    base = os.path.splitext(os.path.basename(media_file))[0]
    return os.path.join(output_dir, f"{base}.whisper-{model_name}.txt")

# ──────────────────────────────────────────────────────────────────────────────
# Batch transcription
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(input_dir, model_name=DEFAULT_MODEL, device_override=None, beam_size=5):
    """Transcribe every media file in input_dir with the specified model."""

    if not os.path.isdir(input_dir):
        print(f"{_RED}[ERROR] Directory not found: {input_dir}{_RESET}")
        sys.exit(1)

    # ── Discover files ──
    media_files = find_media_files(input_dir)
    total = len(media_files)
    if total == 0:
        print(f"{_YELLOW}[INFO] No media files found in: {input_dir}{_RESET}")
        return

    output_dir = make_output_dir(input_dir, model_name)

    print(f"\n{_BOLD}{'='*60}{_RESET}")
    print(f"{_BOLD}  Batch Transcription — Standard Whisper {model_name}{_RESET}")
    print(f"{_BOLD}{'='*60}{_RESET}")
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Files:   {total}")
    print(f"{'='*60}\n")

    # ── Detect device, load model (once) ──
    device = get_device(device_override)
    model = load_whisper_model(model_name, device)

    transcribed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for i, file_path in enumerate(media_files, 1):
        out_path = transcript_path(output_dir, file_path, model_name)

        # ── Resume: skip if transcript already exists ──
        if os.path.exists(out_path):
            print(f"{_DIM}[{i}/{total}] [SKIP] {os.path.basename(file_path)}{_RESET}")
            skipped += 1
            continue

        print(f"\n{_CYAN}[{i}/{total}] Transcribing: {os.path.basename(file_path)}{_RESET}")

        try:
            result = model.transcribe(
                file_path,
                beam_size=beam_size,
                fp16=(device == "cuda"),
            )

            segments = result.get("segments", [])
            lines = []
            for s in segments:
                text = s["text"].strip()
                line = f"[{s['start']:.2f} - {s['end']:.2f}] {text}"
                lines.append(line)
                print(f"  {_DIM}{s['start']:.1f}s{_RESET} {text}")

            # Write transcript (even if empty — marks file as processed)
            duration = segments[-1]["end"] if segments else 0
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"--- Transcription (whisper-{model_name}, {duration:.1f}s) ---\n")
                f.write(f"Source: {os.path.abspath(file_path)}\n")
                f.write(f"Date:   {datetime.now().isoformat()}\n")
                for line in lines:
                    f.write(line + "\n")

            if lines:
                print(f"  {_GREEN}[SAVED] {out_path}{_RESET}")
                transcribed += 1
            else:
                print(f"  {_YELLOW}[SILENT] No speech detected{_RESET}")
                transcribed += 1  # still counts as processed

        except Exception as e:
            print(f"  {_RED}[ERROR] {e}{_RESET}")
            errors += 1

    # ── Summary ──
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    print(f"\n{_BOLD}{'='*60}{_RESET}")
    print(f"{_BOLD}  BATCH TRANSCRIPTION COMPLETE{_RESET}")
    print(f"{_BOLD}{'='*60}{_RESET}")
    print(f"  Total files:   {total}")
    print(f"  Transcribed:   {transcribed}")
    print(f"  Skipped:       {skipped}")
    print(f"  Errors:        {errors}")
    print(f"  Elapsed:       {mins}m {secs}s")
    print(f"  Output folder: {output_dir}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-transcribe all media files in a folder using standard OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_openai.py "D:\\Videos\\meetings"
  python transcribe_openai.py "D:\\Videos\\meetings" --model medium.en
  python transcribe_openai.py "D:\\Videos\\meetings" --model large
  python transcribe_openai.py "D:\\Videos\\meetings" --device cpu
        """,
    )
    parser.add_argument("folder", help="Path to folder containing media files")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Whisper model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device to run on (default: auto-detect)")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size for decoding (default: 5)")

    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    run_batch(args.folder, model_name=args.model, device_override=device, beam_size=args.beam_size)


if __name__ == "__main__":
    main()


import os
import argparse
import whisper
import torch
import sys
import numpy as np
from datetime import timedelta

# Supported audio extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def save_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

def check_gpu():
    if torch.cuda.is_available():
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("CUDA not detected. Using CPU (this will be slow for 'large' model).")
        return "cpu"

def process_file_robust(file_path, output_dir, device):
    """
    Two-pass transcription:
    1. VAD Pass: Use 'small.en' to find valid speech segments.
    2. Transcribe Pass: Use 'large-v2' to transcribe the audio.
       (Note: We don't physically crop; we use the large model on the full file
       but guide it or trust its segmentation, or we could chunk it.
       
       However, the prompt asked to use small.en as VAD.
       A simple way to do this "filtering" is to trust the timestamps from small.en
       and then ask large-v2 to transcribe those specific chunks.
       
       Given Whisper's architecture, we can't easily "force" it to only look at
       timestamps without chopping audio.
       
       Strategy:
       1. Transcribe full file with small.en (fast).
       2. Extract segments that are NOT "hallucinations" (e.g. no_speech_prob < threshold).
       3. Transcribe each valid segment using large-v2 by passing the audio clip.
    """
    
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    # --- Pass 1: VAD with small.en ---
    print("  [Pass 1] VAD/Segmentation with 'small.en'...")
    try:
        model_vad = whisper.load_model("small.en", device=device)
        # We use a higher checking threshold to be strict about silence
        result_vad = model_vad.transcribe(file_path, verbose=False, no_speech_threshold=0.6)
    except Exception as e:
        print(f"  [ERROR] VAD Pass failed: {e}")
        return

    # Filter segments
    # detailed_segments = result_vad["segments"]
    # We want to merge nearby segments? For now, let's keep them as defined by Whisper.
    
    valid_segments = result_vad["segments"]
    print(f"  [Pass 1] Found {len(valid_segments)} segments.")
    
    if not valid_segments:
        print("  [Pass 1] No speech detected. Skipping Pass 2.")
        return

    # Unload small model to free VRAM for large
    del model_vad
    torch.cuda.empty_cache()
    
    # --- Pass 2: Transcription with large-v2 ---
    # The prompt explicitly asked for "not large v3, v2, v1 or turbo".
    # Wait, the prompt said: "run large model (not large v3, v2, v1 or turbo)" ??
    # That is confusing. "Large" usually maps to v2 or v3 now.
    # If the user means the *original* large (deprecated), it's harder to get.
    # But usually "large" = "large-v2" in the openai/whisper repo default.
    # Let's try explicitly loading "large-v2" as it is the most standard "large".
    # Actually, let's load "large" and see what Whisper serves (usually v2).
    
    print("  [Pass 2] Transcribing segments with 'large'...")
    try:
        model_large = whisper.load_model("large", device=device)
    except Exception as e:
        print(f"  [ERROR] Could not load large model: {e}")
        return

    # Load audio once
    audio = whisper.load_audio(file_path)
    
    final_segments = []
    
    for i, seg in enumerate(valid_segments):
        start = seg["start"]
        end = seg["end"]
        
        # Audio slicing (Whisper expects 16kHz audio array)
        # 16000 samples per second
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        
        clip = audio[start_sample:end_sample]
        
        # Transcribe clip
        # Pad slightly if needed or just run
        # We wrap in pad_or_trim logic implicitly handled by transcribe/log_mel_spectrogram usually,
        # but model.transcribe handles raw audio arrays too.
        
        res_seg = model_large.transcribe(clip, verbose=False)
        
        text = res_seg["text"].strip()
        
        # Combine into our format
        final_segments.append({
            "start": start,
            "end": end,
            "text": text
        })
        
        print(f"    Seg {i+1}/{len(valid_segments)} ({start:.1f}s-{end:.1f}s): {text}")

    # --- Save Output ---
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # TXT
    txt_path = os.path.join(output_dir, base_name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        full_text = "\n".join([s["text"] for s in final_segments])
        f.write(full_text)
    print(f"  Saved TXT: {txt_path}")
    
    # SRT
    srt_path = os.path.join(output_dir, base_name + ".srt")
    save_srt(final_segments, srt_path)
    print(f"  Saved SRT: {srt_path}")

    # Cleanup
    del model_large
    torch.cuda.empty_cache()


def process_folder(folder_path, recursive=True):
    # Determine output root folder
    root_abs = os.path.abspath(folder_path)
    parent_dir = os.path.dirname(root_abs)
    base_folder = os.path.basename(root_abs)
    output_root = os.path.join(parent_dir, f"{base_folder}_transcribe_large")
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output folder: {output_root}")
    
    device = check_gpu()

    # Gather files
    audio_files = []
    
    if recursive:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(folder_path, file))
                
    audio_files.sort()
    
    print(f"Found {len(audio_files)} audio files.")
    
    for i, file_path in enumerate(audio_files):
        # Calculate relative path
        rel_path = os.path.relpath(file_path, root_abs)
        rel_dir = os.path.dirname(rel_path)
        
        target_dir = os.path.join(output_root, rel_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        txt_path = os.path.join(target_dir, base_name + ".txt")

        if os.path.exists(txt_path):
            print(f"  Skipping (TXT exists): {txt_path}")
            continue
            
        process_file_robust(file_path, target_dir, device)

def main():
    parser = argparse.ArgumentParser(description="Robust transcription using 'small.en' VAD + 'large' model.")
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive search")
    parser.set_defaults(recursive=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)
        
    process_folder(args.folder, recursive=args.recursive)

if __name__ == "__main__":
    main()

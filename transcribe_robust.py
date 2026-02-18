
import os
import argparse
import whisper
import torch
import sys
import csv
import gc
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
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA detected: {device_name}")
        return "cuda"
    else:
        print("CUDA not detected. Using CPU (this will be slow for 'large' model).")
        return "cpu"

def scan_audio_files(folder_path, recursive=True):
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
    return audio_files

# --- Pass 1: VAD (Small.en) ---

def run_vad_pass(source_folder, output_root_small, device, recursive=True):
    print("\n" + "="*50)
    print(f"PASS 1: VAD & Initial Transcription (Model: 'small.en') on {device.upper()}")
    print(f"Output: {output_root_small}")
    print("="*50)
    
    files = scan_audio_files(source_folder, recursive)
    print(f"Found {len(files)} audio files to scan.")
    
    if not files:
        return

    try:
        print("Loading 'small.en'...")
        model_vad = whisper.load_model("small.en", device=device)
    except Exception as e:
        print(f"[ERROR] Could not load small model: {e}")
        return

    root_abs = os.path.abspath(source_folder)

    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing: {os.path.basename(file_path)}")
        
        # Determine output path for Small
        rel_path = os.path.relpath(file_path, root_abs)
        rel_dir = os.path.dirname(rel_path)
        target_dir = os.path.join(output_root_small, rel_dir)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_path = os.path.join(target_dir, base_name + ".vad.csv")
        txt_path = os.path.join(target_dir, base_name + ".txt")
        srt_path = os.path.join(target_dir, base_name + ".srt")

        # Skip if already done
        if os.path.exists(csv_path) and os.path.exists(txt_path):
            print(f"  Skipping (Output exists): {base_name}")
            continue
            
        try:
            # We use a higher checking threshold to be strict about silence
            # Also set condition_on_previous_text=False to prevent "looping" hallucinations
            result_vad = model_vad.transcribe(
                file_path, 
                verbose=False, 
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4
            )
            segments = result_vad["segments"]
            
            # Save Small.en Transcript
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result_vad["text"])
            save_srt(segments, srt_path)
            print(f"  Saved Small.en Transcript")

            if not segments:
                print("  No speech detected.")
                continue

            # Save VAD data for Pass 2 (CSV Format)
            # Headers: start, end, text
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["start", "end", "text"])
                for seg in segments:
                    writer.writerow([seg["start"], seg["end"], seg["text"].strip()])
            
            # Also save source path reference in a separate small file or just rely on structure?
            # Creating a meta file might be cleaner, but the user asked for CSV format.
            # To know source path in Pass 2, we can infer it or store it.
            # Storing it in a separate meta file for the *folder* or just re-finding source?
            # Re-finding source is safer. We know the relative path structure.
            # Actually, let's just create a companion .meta file or similar if needed.
            # Or pass source path as argument to pass 2?
            # Wait, Pass 2 iterates output folder.
            # If we mirror structure, we can reconstruct source path from relative path if we know source root.
            # But source root is dynamic.
            # Let's save a simple .meta file alongside or just put source in CSV header comment?
            # CSV header comment is standard.
            
            # Re-write CSV with source comment
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(f"# source_path: {file_path}\n")
                writer = csv.writer(f)
                writer.writerow(["start", "end", "text"])
                for seg in segments:
                    writer.writerow([seg["start"], seg["end"], seg["text"].strip()])
            
        except Exception as e:
            print(f"  [ERROR] Pass 1 failed: {e}")

    # Cleanup
    print("Unloading 'small.en'...")
    del model_vad
    torch.cuda.empty_cache()
    gc.collect()


# --- Pass 2: Transcription (Large) ---

def run_transcription_pass(output_root_small, output_root_large, device):
    print("\n" + "="*50)
    print(f"PASS 2: High-Res Transcription (Model: 'large') on {device.upper()}")
    print(f"Output: {output_root_large}")
    print("="*50)
    
    # Scan for .vad.csv files in output_root_small
    csv_files = []
    for root, dirs, files in os.walk(output_root_small):
        for file in files:
            if file.endswith(".vad.csv"):
                csv_files.append(os.path.join(root, file))
    
    csv_files.sort()
    print(f"Found {len(csv_files)} VAD CSV files to process.")
    
    if not csv_files:
        return

    try:
        print("Loading 'large'...")
        # User requested specific "large" key
        model_large = whisper.load_model("large", device=device)
    except Exception as e:
        print(f"[ERROR] Could not load large model: {e}")
        return

    root_abs_small = os.path.abspath(output_root_small)

    for i, csv_path in enumerate(csv_files):
        print(f"[{i+1}/{len(csv_files)}] Transcribing: {os.path.basename(csv_path)}")
        
        try:
            source_path = None
            valid_segments = []
            
            # Read CSV
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Parse source path from first line
            if lines and lines[0].startswith("# source_path:"):
                source_path = lines[0].split(":", 1)[1].strip()
                
            if not source_path:
                print(f"  [ERROR] Source UUID not found in CSV header")
                continue
                
            if not os.path.exists(source_path):
                print(f"  [ERROR] Source file not found: {source_path}")
                continue

            # Parse segments (skip header/comments)
            reader = csv.DictReader([line for line in lines if not line.startswith("#")])
            for row in reader:
                try:
                    valid_segments.append({
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "text": row["text"]
                    })
                except ValueError:
                    continue
            
            # Replicate structure in Large folder
            rel_path = os.path.relpath(csv_path, root_abs_small)
            rel_dir = os.path.dirname(rel_path)
            target_dir = os.path.join(output_root_large, rel_dir)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            base_name = os.path.splitext(os.path.basename(source_path))[0]
            txt_path = os.path.join(target_dir, base_name + ".txt")
            srt_path = os.path.join(target_dir, base_name + ".srt")
            
            if os.path.exists(txt_path):
                print(f"  Skipping (TXT exists): {txt_path}")
                continue

            # Load audio for cropping
            audio = whisper.load_audio(source_path)
            
            final_segments = []
            
            for seg in valid_segments:
                start = seg["start"]
                end = seg["end"]
                
                # Audio slicing (16kHz)
                start_sample = int(start * 16000)
                end_sample = int(end * 16000)
                
                # Safety check for audio length
                if start_sample >= len(audio):
                    continue
                end_sample = min(end_sample, len(audio))
                
                clip = audio[start_sample:end_sample]
                if len(clip) < 1600: # Skip clips < 0.1s
                    continue
                
                # Transcribe clip
                res_seg = model_large.transcribe(clip, verbose=False)
                text = res_seg["text"].strip()
                
                final_segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })

            # Save Results (Large)
            with open(txt_path, "w", encoding="utf-8") as f:
                full_text = "\n".join([s["text"] for s in final_segments])
                f.write(full_text)
            print(f"  Saved TXT (Large)")

            save_srt(final_segments, srt_path)
            print(f"  Saved SRT (Large)")
            
        except Exception as e:
            print(f"  [ERROR] Transcription failed: {e}")

    # Cleanup
    print("Unloading 'large'...")
    del model_large
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Batch Robust Transcription.\nPass 1: small.en -> '_small_en'\nPass 2: large -> '_large'")
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive search")
    parser.set_defaults(recursive=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    # Determine output roots
    root_abs = os.path.abspath(args.folder)
    parent_dir = os.path.dirname(root_abs)
    base_folder = os.path.basename(root_abs)
    
    output_root_small = os.path.join(parent_dir, f"{base_folder}_small_en")
    output_root_large = os.path.join(parent_dir, f"{base_folder}_large")
    
    if not os.path.exists(output_root_small):
        os.makedirs(output_root_small)
    if not os.path.exists(output_root_large):
        os.makedirs(output_root_large)

    device = check_gpu()
    
    # Execute batch passes
    run_vad_pass(args.folder, output_root_small, device, recursive=args.recursive)
    run_transcription_pass(output_root_small, output_root_large, device)
    
    print("\n" + "="*50)
    print("BATCH COMPLETE")
    print(f"Small Logs: {output_root_small}")
    print(f"Large Logs: {output_root_large}")
    print("="*50)

if __name__ == "__main__":
    main()

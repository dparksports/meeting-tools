
import os
import argparse
import whisper
import torch
import sys
import glob

# Supported audio extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}

def transcribe_file(model, file_path, output_dir):
    print(f"Transcribing: {file_path}")
    
    # Transcribe
    result = model.transcribe(file_path, verbose=False) # verbose=False to avoid cluttered output
    
    # Base filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save as TXT
    txt_path = os.path.join(output_dir, base_name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"  Saved TXT: {txt_path}")
    
    # Save as SRT (simple implementation)
    srt_path = os.path.join(output_dir, base_name + ".srt")
    save_srt(result["segments"], srt_path)
    print(f"  Saved SRT: {srt_path}")

def save_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def process_folder(folder_path, model_name="medium.en", recursive=True):
    # Determine output root folder
    # If folder_path is "C:\Audio", output will be "C:\Audio_transcript"
    root_abs = os.path.abspath(folder_path)
    parent_dir = os.path.dirname(root_abs)
    base_folder = os.path.basename(root_abs)
    output_root = os.path.join(parent_dir, f"{base_folder}_transcript")
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output folder: {output_root}")
    
    # Load model once
    print(f"Loading Whisper model '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Gather files
    audio_files = []
    
    if recursive:
        # Walk recursively
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    audio_files.append(os.path.join(root, file))
    else:
        # Flat scan
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(folder_path, file))
                
    audio_files.sort()
    
    print(f"Found {len(audio_files)} audio files.")
    
    for i, file_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] Processing {os.path.basename(file_path)}...")
        
        # Calculate relative path to maintain structure
        # Source: C:\Audio\Sub\file.mp3
        # Rel:    Sub\file.mp3
        # Dest:   C:\Audio_transcript\Sub\file.txt
        
        rel_path = os.path.relpath(file_path, root_abs)
        rel_dir = os.path.dirname(rel_path)
        
        # Target output directory for this file
        target_dir = os.path.join(output_root, rel_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        txt_path = os.path.join(target_dir, base_name + ".txt")

        # Check if output already exists to skip
        if os.path.exists(txt_path):
            print(f"  Skipping (TXT exists): {txt_path}")
            continue
            
        try:
            transcribe_file(model, file_path, target_dir)
        except Exception as e:
            print(f"  Failed to transcribe: {e}")

def main():
    parser = argparse.ArgumentParser(description="Recursively transcribe audio files using OpenAI Whisper.")
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--model", default="medium.en", help="Whisper model size (small, medium, large, etc.)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive search")
    parser.set_defaults(recursive=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)
        
    process_folder(args.folder, model_name=args.model, recursive=args.recursive)

if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import sys
import shutil
from glob import glob

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.m4v'}

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def has_audio_stream(video_path):
    """Check if video has an audio stream using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "a", 
        "-show_entries", "stream=index", 
        "-of", "csv=p=0", 
        video_path
    ]
    try:
        # If output is not empty, there is an audio stream
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return len(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: if ffprobe fails/missing, assume uncertain and let ffmpeg try (or return False safely)
        # Since we checked ffmpeg, ffprobe likely exists.
        return False

def extract_audio(video_path, output_path):
    """
    Extract audio from video using ffmpeg.
    """
    if not has_audio_stream(video_path):
        print("  [SKIP] No audio stream found.")
        return "NO_AUDIO"

    cmd = [
        "ffmpeg", 
        "-y", 
        "-v", "error",       # Less verbose
        "-i", video_path, 
        "-vn", 
        "-acodec", "libmp3lame", 
        "-q:a", "2", 
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] ffmpeg failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch extract audio from video files to a new folder.")
    parser.add_argument("source_folder", help="Folder containing video files")
    parser.add_argument("--suffix", default="_audio", help="Suffix for output folder (default: _audio)")
    
    args = parser.parse_args()
    
    source_dir = os.path.abspath(args.source_folder)
    
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source folder not found: {source_dir}")
        sys.exit(1)
        
    if not check_ffmpeg():
        print("[ERROR] 'ffmpeg' is not found on your system PATH.")
        print("Please install FFmpeg to use this script.")
        sys.exit(1)
        
    # Determine output directory
    # If source is "C:\Videos\MyTrip", output is "C:\Videos\MyTrip_audio"
    parent_dir = os.path.dirname(source_dir)
    base_name = os.path.basename(source_dir)
    output_dir = os.path.join(parent_dir, f"{base_name}{args.suffix}")
    
    print(f"Source: {source_dir}")
    print(f"Dest:   {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output folder: {output_dir}")
    else:
        print(f"Output folder already exists.")
        
    # Find files recursively
    video_files = []
    print(f"Searching for video files in {source_dir}...")
    for ext in VIDEO_EXTENSIONS:
        # Recursive search using **
        video_files.extend(glob(os.path.join(source_dir, "**", f"*{ext}"), recursive=True))
        video_files.extend(glob(os.path.join(source_dir, "**", f"*{ext.upper()}"), recursive=True))
        
    # De-duplicate
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"No video files found in {source_dir}")
        sys.exit(0)
        
    print(f"Found {len(video_files)} video files.")
    print("-" * 60)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, vid_path in enumerate(video_files):
        # Calculate relative path to maintain structure
        rel_path = os.path.relpath(vid_path, source_dir)
        rel_dir = os.path.dirname(rel_path)
        
        filename = os.path.basename(vid_path)
        name_no_ext = os.path.splitext(filename)[0]
        out_name = f"{name_no_ext}.mp3"
        
        # Create corresponding output directory
        current_out_dir = os.path.join(output_dir, rel_dir)
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
            
        out_path = os.path.join(current_out_dir, out_name)
        
        print(f"[{i+1}/{len(video_files)}] {rel_path} -> {out_name} ... ", end="", flush=True)
        
        if os.path.exists(out_path):
            print("SKIPPED (Exists)")
            skip_count += 1
            continue
            
        result = extract_audio(vid_path, out_path)
        if result == True:
            print("DONE")
            success_count += 1
        elif result == "NO_AUDIO":
            skip_count += 1
        else:
            print("FAILED")
            error_count += 1
            
    print("-" * 60)
    print(f"Finished.")
    print(f"Extracted: {success_count}")
    print(f"Skipped:   {skip_count}")
    print(f"Errors:    {error_count}")

if __name__ == "__main__":
    main()

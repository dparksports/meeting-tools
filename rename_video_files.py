"""
Rename Video Files — Extract burned-in timestamps from video files using Qwen2.5-VL-7B.
Standalone script for the TurboScribe Timestamps tab.

Usage:
  python rename_video_files.py <video_file> [--num-frames 5] [--crop-ratio 0.08]
  python rename_video_files.py --batch-folder <folder> [--crop-ratio 0.08]
"""

import argparse
import json
import os
import re
import sys
import subprocess
import tempfile
import shutil
import sys
import subprocess

# Auto-switch to 'pymeetings' venv if not already running in it
def _ensure_venv_execution():
    """If not running in the 'pymeetings' venv, restart this script using the venv python."""
    # Check if we are inside a venv (sys.prefix != sys.base_prefix generally true for venvs)
    # Also check specific venv name to be sure
    in_venv = (sys.prefix != sys.base_prefix) or ("pymeetings" in sys.executable)
    
    if in_venv:
        return

    # Look for the venv python
    venv_name = "pymeetings"
    if sys.platform == "win32":
        venv_python = os.path.join(os.getcwd(), venv_name, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(os.getcwd(), venv_name, "bin", "python")

    if os.path.exists(venv_python):
        # enhancing UI feedback
        # print(f"[INFO] Switching to '{venv_name}' virtual environment...")
        
        # Re-execute the script with the same arguments
        try:
            cmd = [venv_python] + sys.argv
            subprocess.run(cmd, check=True)
            sys.exit(0) # Exit the original process
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)
        except Exception as e:
            print(f"[ERROR] Failed to switch to venv: {e}")
            sys.exit(1)
    else:
        # If venv not found, just proceed (or maybe warn)
        # We'll proceed — maybe user installed deps globally
        pass

# Run venv check immediately
_ensure_venv_execution()

# Tesla dashcam filename pattern: YYYY-MM-DD_HH-MM-SS-<camera>.mp4
_TESLA_DASHCAM_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(front|back|left_repeater|right_repeater|left_pillar|right_pillar|narrow_front)\.mp4$',
    re.IGNORECASE
)

# Already-renamed pattern: YYMMDD-DAY-HHMM-HHMM-ampm-camera.mp4
_ALREADY_RENAMED_RE = re.compile(
    r'^\d{6}-[A-Z]{2,3}-\d{3,4}-\d{3,4}-(am|pm)-.*\.mp4$',
    re.IGNORECASE
)

# Force unbuffered output for real-time UI updates
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# Globals for lazy-loaded VLM
_vlm_model = None
_vlm_processor = None


def _load_vlm():
    """Load Qwen2.5-VL-7B model (cached after first load)."""
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print("[ERROR] Required packages not installed. Run:")
        print("[ERROR]   pip install transformers accelerate qwen-vl-utils torchvision Pillow")
        return None, None

    try:
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

        # Try to use GPU — do a quick smoke test to confirm it works
        use_cuda = False
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                use_cuda = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[TIMESTAMP] GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
            except RuntimeError as e:
                print(f"[TIMESTAMP] GPU available but unusable ({e}) — falling back to CPU.")

        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "auto" if use_cuda else "cpu"

        # Try loading from local cache first (no network calls)
        try:
            print(f"[TIMESTAMP] Loading model from cache on {'GPU' if use_cuda else 'CPU'}...")
            _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                local_files_only=True,
            )
            _vlm_processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
            print("[TIMESTAMP] Model loaded from cache.")
            return _vlm_model, _vlm_processor
        except Exception:
            pass  # Not cached yet — download below

        # First time: download from HuggingFace Hub
        print("[TIMESTAMP] Model not cached — downloading (~15 GB, one-time)...")
        print(f"[TIMESTAMP] Loading on {'GPU (CUDA)' if use_cuda else 'CPU'}...")
        _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        _vlm_processor = AutoProcessor.from_pretrained(model_name)
        print("[TIMESTAMP] Model downloaded and loaded successfully.")
        return _vlm_model, _vlm_processor
    except Exception as e:
        print(f"[ERROR] Failed to load VLM: {e}")
        return None, None


def _extract_frames(video_path, num_frames=5):
    """Extract evenly-spaced frames from a video file using ffmpeg."""
    # Get video duration via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path
    ]
    try:
        output = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
        if output == "N/A":
            duration = 0.0
        else:
            duration = float(output)
    except Exception as e:
        print(f"[ERROR] ffprobe failed for {video_path}: {e}")
        # print("[ERROR] Make sure ffmpeg/ffprobe is installed and on PATH.") # Reduced noise
        return [], 0

    if duration <= 0:
        print("[ERROR] Video has zero duration")
        return [], 0

    print(f"[TIMESTAMP] Video duration: {duration:.1f}s")

    # Calculate timestamps for evenly spaced frames
    if num_frames == 1:
        timestamps = [duration / 2]
    elif num_frames == 2:
        # Batch mode: first + last frame (avoid black frames at edges)
        start = min(2.0, duration * 0.05)
        end = max(duration - 2.0, duration * 0.95)
        timestamps = [start, end]
    else:
        # Avoid first/last 1 second to skip potential black frames
        start = min(1.0, duration * 0.05)
        end = max(duration - 1.0, duration * 0.95)
        step = (end - start) / (num_frames - 1)
        timestamps = [start + i * step for i in range(num_frames)]

    frames = []
    tmp_dir = tempfile.mkdtemp(prefix="ts_frames_")

    for i, ts in enumerate(timestamps):
        out_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.2f}",
            "-i", video_path,
            "-vframes", "1", "-q:v", "2",
            out_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                frames.append({"path": out_path, "timestamp_sec": ts})
            else:
                print(f"[WARN] Frame at {ts:.1f}s produced empty file")
        except Exception as e:
            print(f"[WARN] Frame at {ts:.1f}s failed: {e}")

    return frames, duration


def _crop_timestamp_region(image_path, crop_ratio=0.08):
    """Crop the top portion of an image where timestamps typically appear."""
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        crop_h = int(h * crop_ratio)
        if crop_h < 10:
            crop_h = 10
        cropped = img.crop((0, 0, w, crop_h))
        cropped_path = image_path.replace(".jpg", "_crop.jpg")
        cropped.save(cropped_path, quality=95)
        return cropped_path
    except ImportError:
        print("[WARN] Pillow not installed, using full frame")
        return image_path
    except Exception as e:
        print(f"[WARN] Crop failed: {e}, using full frame")
        return image_path


def _crop_camera_label_region(image_path, ratio=0.12):
    """Crop the lower-right corner of the frame where camera labels usually appear."""
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        # Lower-right corner: bottom 'ratio' of height, right 40% of width
        left = int(w * 0.60)
        top = int(h * (1 - ratio))
        cropped = img.crop((left, top, w, h))
        cropped_path = image_path.replace(".jpg", "_camera.jpg").replace(".png", "_camera.png")
        cropped.save(cropped_path)
        return cropped_path
    except Exception as e:
        print(f"[ERROR] Camera label crop failed: {e}")
        return image_path


def _read_camera_label(model, processor, image_path):
    """Use VLM to read camera name/label from a cropped image region."""
    try:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path)

        prompt = (
            "Read the camera name or location label shown in this image. "
            "Return ONLY the camera name text (e.g. 'driveway1104', 'backyard', 'garage_cam'), nothing else. "
            "If no camera name is visible, respond with 'NONE'."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text if output_text and output_text != "NONE" else None

    except Exception as e:
        print(f"[ERROR] Camera label extraction failed: {e}")
        return None


def _sanitize_label(label):
    """Sanitize a camera label for use in filenames."""
    import re
    if not label:
        return None
    s = label.strip()
    s = re.sub(r'[<>:"/\\|?*]', '', s)   # remove invalid filename chars
    s = re.sub(r'\s+', '_', s)             # spaces to underscores
    s = s.strip('_.-')
    return s if s else None


def _read_timestamp_from_image(model, processor, image_path):
    """Use VLM to read timestamp text from an image."""
    try:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path)

        prompt = (
            "Read the exact timestamp and camera name shown in this image. "
            "Return ONLY the timestamp text in the format it appears, nothing else. "
            "If no timestamp is visible, respond with 'NONE'."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text

    except Exception as e:
        print(f"[ERROR] VLM inference failed: {e}")
        return "ERROR"


def _find_consensus(timestamps):
    """Find consensus among extracted timestamps using majority voting."""
    from collections import Counter

    valid = [t for t in timestamps if t and t != "NONE" and t != "ERROR"]
    if not valid:
        return None

    counter = Counter(valid)
    most_common = counter.most_common(1)[0][0]
    return most_common


def run_extract_timestamps(file_path, num_frames=5, crop_ratio=0.08):
    """
    Extract burned-in timestamps from video frames using Qwen2.5-VL.
    Samples multiple frames and uses majority voting for accuracy.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    print(f"[TIMESTAMP] Processing: {file_path}")
    print(f"[TIMESTAMP] Extracting {num_frames} frames...")

    # Step 1: Extract frames
    frames, duration = _extract_frames(file_path, num_frames)
    if not frames:
        print("[ERROR] No frames could be extracted. Is ffmpeg installed?")
        return

    print(f"[TIMESTAMP] Extracted {len(frames)} frames")

    # Step 2: Load VLM
    model, processor = _load_vlm()
    if model is None:
        return

    # Step 3: Process each frame
    results = []
    for i, frame in enumerate(frames):
        ts_sec = frame["timestamp_sec"]
        print(f"[TIMESTAMP] Reading frame {i+1}/{len(frames)} (at {ts_sec:.1f}s)...")

        # Crop timestamp region
        cropped = _crop_timestamp_region(frame["path"], crop_ratio)

        # Read timestamp with VLM
        raw_text = _read_timestamp_from_image(model, processor, cropped)

        confidence = "high" if raw_text and raw_text != "NONE" and raw_text != "ERROR" else "low"
        results.append({
            "frame_sec": round(ts_sec, 1),
            "raw_text": raw_text,
            "confidence": confidence,
        })
        print(f"[TIMESTAMP] Frame {i+1}: {raw_text} ({confidence})")

    # Step 4: Find consensus
    all_texts = [r["raw_text"] for r in results]
    consensus = _find_consensus(all_texts)

    # Build output
    high_conf = [r for r in results if r["confidence"] == "high"]
    output = {
        "file": os.path.abspath(file_path),
        "timestamps": results,
        "consensus": consensus,
        "frames_extracted": len(frames),
        "frames_readable": len(high_conf),
        "video_duration_sec": round(duration, 1),
    }

    if len(high_conf) >= 2:
        output["first_timestamp"] = high_conf[0]["raw_text"]
        output["last_timestamp"] = high_conf[-1]["raw_text"]

    print(f"\n[TIMESTAMP] === Results ===")
    print(f"[TIMESTAMP] Consensus: {consensus}")
    print(f"[TIMESTAMP] Readable frames: {len(high_conf)}/{len(frames)}")

    # Output JSON for the C# app to parse
    print(f"[TIMESTAMP_RESULT] {json.dumps(output)}")

    # Cleanup temp frames
    for frame in frames:
        tmp_dir = os.path.dirname(frame["path"])
        if "ts_frames_" in tmp_dir:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
            break

    return output


def _parse_timestamp_parts(timestamp_text):
    """
    Parse a raw timestamp string into (date_str, time_str, ampm, day_of_week).
    Handles formats like '10/21/2025 09:06:03 am TUE', '2024-01-15 10:30:22', etc.
    Returns e.g. ('251021', '0906', 'am', 'TUE') or (None, None, None, None).
    """
    if not timestamp_text:
        return None, None, None, None
    import re
    s = timestamp_text.strip()

    # Try MM/DD/YYYY format first (US-style)
    m = re.match(
        r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\s+'  # date MM/DD/YYYY
        r'(\d{1,2}[:\-]\d{2}(?:[:\-]\d{2})?)'             # time
        r'(?:\s*(am|pm))?'                                  # optional am/pm
        r'(?:\s*([A-Za-z]{2,3}))?',                         # optional day name
        s, re.IGNORECASE
    )
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        date_part = f"{year[2:]}{month.zfill(2)}{day.zfill(2)}"  # YYMMDD
        raw_time = m.group(4)
        tp = re.split(r'[:\-]', raw_time)
        time_part = f"{tp[0].zfill(2)}{tp[1].zfill(2) if len(tp) > 1 else '00'}"
        ampm = m.group(5).lower() if m.group(5) else None
        day_of_week = m.group(6).upper() if m.group(6) else None
        return date_part, time_part, ampm, day_of_week

    # Try YYYY/MM/DD or YYYY-MM-DD format
    m = re.match(
        r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\s+'  # date YYYY-MM-DD
        r'(\d{1,2}[:\-]\d{2}(?:[:\-]\d{2})?)'             # time
        r'(?:\s*(am|pm))?'                                  # optional am/pm
        r'(?:\s*([A-Za-z]{2,3}))?',                         # optional day name
        s, re.IGNORECASE
    )
    if m:
        year, month, day = m.group(1), m.group(2), m.group(3)
        date_part = f"{year[2:]}{month.zfill(2)}{day.zfill(2)}"  # YYMMDD
        raw_time = m.group(4)
        tp = re.split(r'[:\-]', raw_time)
        time_part = f"{tp[0].zfill(2)}{tp[1].zfill(2) if len(tp) > 1 else '00'}"
        ampm = m.group(5).lower() if m.group(5) else None
        day_of_week = m.group(6).upper() if m.group(6) else None
        return date_part, time_part, ampm, day_of_week

    # Fallback: no valid timestamp
    return None, None, None, None


def _build_rename(start_text, end_text, camera_label=None):
    """
    Build a compact filename from start/end timestamps and optional camera label.
    Format: YYMMDD-day_of_week-hhmm-hhmm-ampm-camera_location.mp4
    E.g. '251021-TUE-0906-0912-am-driveway1104.mp4'
    Requires both a valid date and camera label to rename.
    """
    start_date, start_time, start_ampm, start_day = _parse_timestamp_parts(start_text)
    end_date, end_time, end_ampm, end_day = _parse_timestamp_parts(end_text)

    # Pick the date and day of week (prefer start, fall back to end)
    date = start_date or end_date
    day_of_week = start_day or end_day
    ampm = start_ampm or end_ampm
    label = _sanitize_label(camera_label)

    # Skip rename if no date or no camera label
    if not date or not label:
        return None

    # Build name parts list
    parts = []

    # Date (YYMMDD)
    parts.append(date)

    # Day of week
    if day_of_week:
        parts.append(day_of_week)

    # Start time (HHMM)
    if start_time:
        parts.append(start_time)

    # End time (HHMM)
    if end_time:
        parts.append(end_time)

    # AM/PM
    if ampm:
        parts.append(ampm)

    # Camera label (at the end)
    parts.append(label)

    return "-".join(parts) + ".mp4"


def run_batch_rename(folder_path, crop_ratio=0.08, recursive=False, prefix=None, do_rename=True):
    """
    Process all .mp4 files in a folder, extracting start/end timestamps
    for batch renaming. Uses only 2 frames per video for speed.
    Renames files by default and saves results to CSV.
    """
    import csv
    import glob

    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    if recursive:
        mp4_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.mp4"), recursive=True))
        scope = f"{folder_path} (including subfolders)"
    else:
        mp4_files = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))
        scope = folder_path

    # Filter by prefix if specified
    if prefix:
        mp4_files = [f for f in mp4_files if os.path.basename(f).lower().startswith(prefix.lower())]
        scope += f" [prefix: {prefix}]"

    if not mp4_files:
        print(f"[ERROR] No .mp4 files found in {scope}")
        return

    print(f"[BATCH] Found {len(mp4_files)} video(s) in {scope}")
    if do_rename:
        print(f"[BATCH] Files will be renamed after timestamp extraction.")
    else:
        print(f"[BATCH] Rename disabled (--no-rename). Files will not be renamed.")

    # Resume support: skip files already recorded in the CSV
    csv_dir = os.getcwd()
    all_csv_path = os.path.join(csv_dir, "batch_timestamps.csv")
    already_done = set()
    if os.path.exists(all_csv_path):
        try:
            with open(all_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Track by both original and new filename + folder
                    folder = row.get("folder", "")
                    orig = row.get("original_file", "")
                    new = row.get("new_file", "")
                    if orig:
                        already_done.add(os.path.join(folder, orig))
                    if new and new != orig:
                        already_done.add(os.path.join(folder, new))
        except Exception as e:
            print(f"[BATCH] Warning: could not read existing CSV for resume: {e}")

    before_skip = len(mp4_files)
    mp4_files = [f for f in mp4_files if os.path.abspath(f) not in already_done]
    skipped = before_skip - len(mp4_files)

    if skipped:
        print(f"[BATCH] Resuming — skipped {skipped} already-processed file(s)")

    if not mp4_files:
        print(f"[BATCH] All files already processed. Nothing to do.")
        return

    # Pre-load VLM once for all videos
    model, processor = _load_vlm()
    if model is None:
        print("[ERROR] Cannot load VLM model — aborting batch.")
        return

    all_results = []
    renamed_count = 0

    for idx, video_path in enumerate(mp4_files):
        filename = os.path.basename(video_path)
        print(f"\n[BATCH] Processing {idx+1}/{len(mp4_files)}: {filename}")

        # Skip Tesla dashcam files (no burned-in timestamps)
        if _TESLA_DASHCAM_RE.match(filename):
            print(f"[BATCH] Skipping {filename} — Tesla dashcam file")
            result = {
                "original_file": filename,
                "new_file": filename,
                "camera_label": None,
                "folder": os.path.dirname(os.path.abspath(video_path)),
                "start_timestamp": None,
                "end_timestamp": None,
                "duration_sec": 0,
                "renamed": False,
                "error": "Tesla dashcam — skipped"
            }
            all_results.append(result)
            print(f"[BATCH_RESULT] {json.dumps(result)}")
            continue

        # Skip files already renamed by this script
        if _ALREADY_RENAMED_RE.match(filename):
            print(f"[BATCH] Skipping {filename} — already renamed")
            continue

        # Extract only 2 frames: first + last
        frames, duration = _extract_frames(video_path, num_frames=2)
        if len(frames) < 2:
            print(f"[BATCH] Skipping {filename} — could not extract 2 frames")
            result = {
                "original_file": filename,
                "new_file": None,
                "camera_label": None,
                "folder": os.path.dirname(os.path.abspath(video_path)),
                "start_timestamp": None,
                "end_timestamp": None,
                "duration_sec": round(duration, 1) if duration else 0,
                "renamed": False,
                "error": "Could not extract frames"
            }
            all_results.append(result)
            print(f"[BATCH_RESULT] {json.dumps(result)}")
            _cleanup_frames(frames)
            continue

        start_text = None
        end_text = None
        camera_label = None

        for i, frame in enumerate(frames):
            # Extract timestamp from top of frame
            cropped = _crop_timestamp_region(frame["path"], crop_ratio)
            raw = _read_timestamp_from_image(model, processor, cropped)
            if raw and raw != "NONE" and raw != "ERROR":
                if i == 0:
                    start_text = raw
                else:
                    end_text = raw
            label = "start" if i == 0 else "end"
            print(f"[BATCH]   {label}: {raw}")

            # Extract camera label from lower-right of first frame only
            if i == 0:
                cam_crop = _crop_camera_label_region(frame["path"])
                camera_label = _read_camera_label(model, processor, cam_crop)
                if camera_label:
                    print(f"[BATCH]   camera: {camera_label}")

        # Build new filename from timestamps + camera label
        new_filename = None
        was_renamed = False

        if do_rename and (start_text or end_text):
            new_filename = _build_rename(start_text, end_text, camera_label=camera_label)

            if new_filename and new_filename != filename:
                new_path = os.path.join(os.path.dirname(video_path), new_filename)
                # Avoid overwriting existing files
                if os.path.exists(new_path):
                    base, ext = os.path.splitext(new_filename)
                    counter = 2
                    while os.path.exists(new_path):
                        new_filename = f"{base}_{counter}{ext}"
                        new_path = os.path.join(os.path.dirname(video_path), new_filename)
                        counter += 1

                try:
                    os.rename(video_path, new_path)
                    was_renamed = True
                    renamed_count += 1
                    print(f"[BATCH] ✓ Renamed → {new_filename}")
                except OSError as e:
                    print(f"[BATCH] ✗ Rename failed: {e}")
            elif new_filename == filename:
                print(f"[BATCH] Already named correctly, skipping rename.")

        result = {
            "original_file": filename,
            "new_file": new_filename if was_renamed else filename,
            "camera_label": camera_label,
            "folder": os.path.dirname(os.path.abspath(video_path)),
            "start_timestamp": start_text,
            "end_timestamp": end_text,
            "duration_sec": round(duration, 1),
            "renamed": was_renamed,
            "error": None
        }
        all_results.append(result)
        print(f"[BATCH_RESULT] {json.dumps(result)}")

        # Cleanup temp frames
        _cleanup_frames(frames)

    # Split results into succeeded and failed
    succeeded = [r for r in all_results if not r.get("error")]
    failed = [r for r in all_results if r.get("error")]

    fieldnames = [
        "original_file", "new_file", "camera_label", "folder",
        "start_timestamp", "end_timestamp",
        "duration_sec", "renamed", "error"
    ]

    def _write_csv(path, rows):
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    # Save all 3 CSVs to current working directory
    csv_dir = os.getcwd()
    try:
        all_csv = os.path.join(csv_dir, "batch_timestamps.csv")
        _write_csv(all_csv, all_results)
        print(f"\n[BATCH] CSV (all)       → {all_csv} ({len(all_results)} rows)")

        ok_csv = os.path.join(csv_dir, "batch_succeeded.csv")
        _write_csv(ok_csv, succeeded)
        print(f"[BATCH] CSV (succeeded) → {ok_csv} ({len(succeeded)} rows)")

        fail_csv = os.path.join(csv_dir, "batch_failed.csv")
        _write_csv(fail_csv, failed)
        print(f"[BATCH] CSV (failed)    → {fail_csv} ({len(failed)} rows)")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")

    print(f"[BATCH] Done — processed {len(mp4_files)} videos, renamed {renamed_count}, failed {len(failed)}.")


def _cleanup_frames(frames):
    """Remove temporary frame files."""
    for frame in frames:
        tmp_dir = os.path.dirname(frame["path"])
        if "ts_frames_" in tmp_dir:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract burned-in timestamps from video files using Qwen2.5-VL-7B")
    parser.add_argument("file", nargs="?", help="Path to video file")
    parser.add_argument("--batch-folder", help="Process all .mp4 files in a folder (batch rename mode)")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Disable recursive search (default: enabled)")
    parser.add_argument("--prefix", help="Only process files starting with this prefix (e.g., 'reo')")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to extract (default: 5)")
    parser.add_argument("--crop-ratio", type=float, default=0.08, help="Fraction of frame height to crop from top (default: 0.08)")
    parser.add_argument("--no-rename", action="store_true", help="Do not rename files, only extract timestamps and save CSV")
    args = parser.parse_args()

    if args.batch_folder:
        run_batch_rename(args.batch_folder, crop_ratio=args.crop_ratio, recursive=args.recursive,
                         prefix=args.prefix, do_rename=not args.no_rename)
    elif args.file:
        run_extract_timestamps(args.file, num_frames=args.num_frames, crop_ratio=args.crop_ratio)
    else:
        parser.print_help()

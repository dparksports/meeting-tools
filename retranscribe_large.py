
import os
import re
import sys
import csv
import gc
import time
import argparse
from collections import Counter

# Supported audio extensions (for finding source audio)
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}


# ============================================================
# SRT Parsing
# ============================================================

def parse_srt(srt_path):
    """
    Parse an SRT file into a list of segment dicts:
    [{"index": 1, "start": 0.0, "end": 1.5, "text": "Hello"}, ...]
    """
    segments = []
    if not os.path.exists(srt_path):
        return segments

    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return segments

    # Split on blank lines
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Parse timestamp line: "00:19:00,000 --> 00:19:01,599"
        ts_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
            lines[1].strip()
        )
        if not ts_match:
            continue

        g = ts_match.groups()
        start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
        end = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000
        text = ' '.join(lines[2:]).strip()

        segments.append({
            "start": start,
            "end": end,
            "text": text,
        })

    return segments


# ============================================================
# Segment-Level Classification
# ============================================================

def is_segment_hallucination(text):
    """
    Check if a single SRT segment's text looks like hallucination.
    Returns True if the segment is likely hallucinated.
    """
    text = text.strip()
    if not text:
        return True

    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    words = clean.split()

    if not words:
        return True

    # Single word or very short
    if len(words) <= 2:
        # Single filler words are hallucination
        fillers = {'yeah', 'yes', 'no', 'oh', 'hmm', 'um', 'uh', 'hey',
                   'now', 'so', 'ok', 'okay', 'whoa', 'wow', 'poop',
                   'easy', 'hello', 'hi', 'bye', 'thanks', 'thank'}
        if all(w in fillers for w in words):
            return True
        return False

    # Check word repetition within this segment
    unique = set(words)
    unique_ratio = len(unique) / len(words)

    # Dominant single word
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    top_pct = top_count / len(words)

    # Strong hallucination: very low diversity or one word dominates
    if unique_ratio < 0.25 or top_pct > 0.6:
        return True

    return False


def classify_segments(segments):
    """
    Classify each SRT segment as good or hallucination.
    Returns (good_segments, hallucination_count, total_count)
    """
    good = []
    hallucination_count = 0

    for seg in segments:
        if is_segment_hallucination(seg["text"]):
            hallucination_count += 1
        else:
            good.append(seg)

    return good, hallucination_count, len(segments)


# ============================================================
# File-Level Classification (with segment awareness)
# ============================================================

def classify_transcript(txt_path, srt_path, junk_threshold=0.3, good_threshold=0.6):
    """
    Classify a transcript using both file-level text metrics
    AND segment-level SRT analysis.

    Returns: (classification, metrics_dict, good_segments)
      - good_segments: list of SRT segments that look like real speech
        (only relevant for 'retranscribe' classification)
    """
    # Read the full text
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except Exception:
        return "junk", {"unique_ratio": 0.0, "word_count": 0, "top_word_pct": 1.0,
                        "good_segments": 0, "total_segments": 0}, []

    if not text:
        return "junk", {"unique_ratio": 0.0, "word_count": 0, "top_word_pct": 1.0,
                        "good_segments": 0, "total_segments": 0}, []

    # --- File-level metrics ---
    clean = re.sub(r'[^\w\s]', '', text.lower())
    words = clean.split()
    word_count = len(words)

    if word_count == 0:
        return "junk", {"unique_ratio": 0.0, "word_count": 0, "top_word_pct": 1.0,
                        "good_segments": 0, "total_segments": 0}, []

    unique_words = set(words)
    unique_ratio = len(unique_words) / word_count

    word_counts = Counter(words)
    _, most_common_count = word_counts.most_common(1)[0]
    top_word_pct = most_common_count / word_count

    if word_count >= 4:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(word_count - 1)]
        bigram_ratio = len(set(bigrams)) / len(bigrams)
    else:
        bigram_ratio = 1.0

    # --- Segment-level analysis ---
    segments = parse_srt(srt_path)
    good_segments, hallucination_count, total_segments = classify_segments(segments)

    metrics = {
        "unique_ratio": round(unique_ratio, 3),
        "word_count": word_count,
        "top_word_pct": round(top_word_pct, 3),
        "bigram_ratio": round(bigram_ratio, 3),
        "good_seg_count": len(good_segments),
        "total_segments": total_segments,
    }

    # --- Classification logic ---

    # If SRT has good segments, this file has SOME real speech
    has_good_segments = len(good_segments) > 0

    # Very short transcripts
    if word_count <= 5 and not has_good_segments:
        return "junk", metrics, []

    # File-level says "good" — clean transcript
    if (unique_ratio >= good_threshold and top_word_pct < 0.15
            and bigram_ratio > 0.5):
        return "good", metrics, good_segments

    # File-level says "junk" BUT segments say there's real speech
    # -> upgrade to retranscribe (only the good segments)
    if has_good_segments:
        return "retranscribe", metrics, good_segments

    # File-level junk with no good segments
    if unique_ratio < junk_threshold or top_word_pct > 0.5 or bigram_ratio < 0.2:
        return "junk", metrics, []

    # Middle ground — retranscribe the whole file
    return "retranscribe", metrics, good_segments


# ============================================================
# Scan & Report
# ============================================================

def find_audio_path(transcript_path, transcript_root):
    """
    Given a transcript .txt path under *_transcript folder,
    find the matching audio file under *_audio folder.

    Example:
      transcript: C:\\Users\\k2\\videos-daytime_audio_transcript\\200126-daytime\\file.txt
      audio:      C:\\Users\\k2\\videos-daytime_audio\\200126-daytime\\file.mp3
    """
    transcript_root_abs = os.path.abspath(transcript_root)
    parent_dir = os.path.dirname(transcript_root_abs)
    root_name = os.path.basename(transcript_root_abs)

    # Strip _transcript suffix to get the audio root
    # e.g. "videos-daytime_audio_transcript" -> "videos-daytime_audio"
    if root_name.endswith("_transcript"):
        audio_root_name = root_name[:-len("_transcript")]
    elif "_transcript" in root_name:
        audio_root_name = root_name.replace("_transcript", "", 1)
    else:
        audio_root_name = root_name + "_audio"

    audio_root = os.path.join(parent_dir, audio_root_name)

    # Get relative path within transcript tree
    rel_path = os.path.relpath(transcript_path, transcript_root_abs)
    base_name = os.path.splitext(rel_path)[0]

    # Try each audio extension
    for ext in AUDIO_EXTENSIONS:
        candidate = os.path.join(audio_root, base_name + ext)
        if os.path.exists(candidate):
            return candidate

    return None


def scan_and_classify(transcript_folder, junk_threshold=0.3, good_threshold=0.6):
    """
    Walk the transcript folder, classify each .txt file
    using both text-level and SRT segment-level analysis.
    """
    results = []
    transcript_root = os.path.abspath(transcript_folder)

    for root, dirs, files in os.walk(transcript_root):
        for file in sorted(files):
            if not file.lower().endswith('.txt'):
                continue

            txt_path = os.path.join(root, file)
            rel_path = os.path.relpath(txt_path, transcript_root)

            # Find matching SRT
            srt_path = os.path.splitext(txt_path)[0] + '.srt'

            classification, metrics, good_segments = classify_transcript(
                txt_path, srt_path, junk_threshold, good_threshold
            )

            audio_path = find_audio_path(txt_path, transcript_root)

            results.append({
                "rel_path": rel_path,
                "txt_path": txt_path,
                "srt_path": srt_path,
                "audio_path": audio_path,
                "classification": classification,
                "good_segments": good_segments,
                **metrics
            })

    return results


def write_triage_report(results, output_path):
    """Write classification results to a CSV report."""
    fieldnames = [
        "rel_path", "classification", "unique_ratio",
        "word_count", "top_word_pct", "bigram_ratio",
        "good_segments", "total_segments",
        "audio_path", "audio_found"
    ]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "rel_path": r["rel_path"],
                "classification": r["classification"],
                "unique_ratio": r["unique_ratio"],
                "word_count": r["word_count"],
                "top_word_pct": r["top_word_pct"],
                "bigram_ratio": r.get("bigram_ratio", ""),
                "good_segments": r.get("good_seg_count", 0),
                "total_segments": r.get("total_segments", ""),
                "audio_path": r.get("audio_path", ""),
                "audio_found": "yes" if r.get("audio_path") else "NO",
            })

    print(f"Triage report saved: {output_path}")


def print_summary(results):
    """Print a summary of classification results."""
    counts = Counter(r["classification"] for r in results)
    total = len(results)
    missing_audio = sum(1 for r in results
                        if r["classification"] == "retranscribe" and not r.get("audio_path"))

    # Count how many retranscribe files will use segment-only mode
    segment_mode = sum(1 for r in results
                       if r["classification"] == "retranscribe"
                       and r.get("good_segments")
                       and r.get("total_segments", 0) > 0
                       and len(r["good_segments"]) < r["total_segments"])

    print(f"\n{'='*60}")
    print(f"TRIAGE SUMMARY  ({total} transcripts)")
    print(f"{'='*60}")
    retrans = counts.get('retranscribe', 0) + counts.get('good', 0)
    print(f"  junk:          {counts.get('junk', 0):>5}  (skip — no real speech)")
    print(f"  retranscribe:  {retrans:>5}  (re-transcribe with large)")
    if counts.get('retranscribe', 0) and counts.get('good', 0):
        print(f"    noisy:         {counts.get('retranscribe', 0):>3}  (mixed quality)")
        print(f"    clean:         {counts.get('good', 0):>3}  (good small.en, upgrade to large)")
    if segment_mode:
        print(f"    segment-only:  {segment_mode:>3}  (only good segments, not full file)")

    if missing_audio:
        print(f"\n  [WARNING] {missing_audio} retranscribe files have no matching audio!")

    print(f"{'='*60}\n")


# ============================================================
# Re-transcribe with Large Model
# ============================================================

def format_eta(seconds):
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s:02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = format_timestamp(segment.get("start", 0))
            end = format_timestamp(segment.get("end", 0))
            text = segment.get("text", "").strip()
            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def retranscribe(results, transcript_root, model_name="large", force=False, include_junk=False):
    """
    Re-transcribe files classified as 'retranscribe' using the large model.

    For files where only some SRT segments are good, only those time ranges
    are sent to the large model (segment-only mode). For files where all/most
    segments are good, the full file is transcribed.
    """
    import whisper
    import torch

    # Determine output folder
    transcript_root_abs = os.path.abspath(transcript_root)
    parent_dir = os.path.dirname(transcript_root_abs)
    root_name = os.path.basename(transcript_root_abs)

    # Strip _transcript and append _large
    if "_transcript" in root_name:
        base = root_name.replace("_transcript", "", 1)
        output_root_name = base + "_large"
    else:
        output_root_name = root_name + "_large"

    output_root = os.path.join(parent_dir, output_root_name)
    os.makedirs(output_root, exist_ok=True)

    # Setup junk output folder
    output_root_junk = os.path.join(parent_dir, output_root_name + "_junk")
    if include_junk:
        os.makedirs(output_root_junk, exist_ok=True)

    # Filter files to process
    allowed_classes = ("retranscribe", "good")
    if include_junk:
        allowed_classes += ("junk",)

    to_process = [
        r for r in results
        if r["classification"] in allowed_classes and r.get("audio_path")
    ]

    if not to_process:
        print("No files to re-transcribe.")
        return

    # Check for existing outputs (resume support)
    queue = []
    for r in to_process:
        base_name = os.path.splitext(r["rel_path"])[0]
        
        # Check specific output folder based on class
        if r["classification"] == "junk":
            target_output_root = output_root_junk
        else:
            target_output_root = output_root

        out_txt = os.path.join(target_output_root, base_name + ".txt")

        if os.path.exists(out_txt) and not force:
            continue
        queue.append(r)

    skipped = len(to_process) - len(queue)
    if skipped:
        print(f"Skipping {skipped} already-transcribed files (use --force to redo).")

    if not queue:
        print("All retranscribe files already have large-model output.")
        return

    # Count segment-only vs full-file for ETA estimate
    segment_only_count = sum(
        1 for r in queue
        if r.get("good_segments") and r.get("total_segments", 0) > 0
        and len(r["good_segments"]) < r["total_segments"]
    )
    full_file_count = len(queue) - segment_only_count

    print(f"\n{'='*60}")
    print(f"RE-TRANSCRIBING {len(queue)} files with '{model_name}' model")
    print(f"  Full file:     {full_file_count}")
    print(f"  Segment-only:  {segment_only_count}")
    print(f"  Output:        {output_root}")
    if include_junk:
        print(f"  Junk Output:   {output_root_junk}")
    print(f"{'='*60}\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading '{model_name}' model...")

    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # --- ETA estimate (rough, before first file) ---
    # Rough heuristic: ~45s per full file, ~10s per segment-only on GPU; 5x slower on CPU
    speed_mult = 1 if device == "cuda" else 5
    est_seconds = (full_file_count * 45 + segment_only_count * 10) * speed_mult
    print(f"\nEstimated time: ~{format_eta(est_seconds)} ({device.upper()})")
    print(f"(This is a rough estimate — actual ETA will update after the first file.)\n")

    job_start = time.time()
    file_times = []  # track per-file durations for rolling average

    for i, r in enumerate(queue):
        file_start = time.time()
        audio_path = r["audio_path"]
        base_name = os.path.splitext(r["rel_path"])[0]
        rel_dir = os.path.dirname(r["rel_path"])
        good_segs = r.get("good_segments", [])
        total_segs = r.get("total_segments", 0)

        # Decide mode: segment-only vs full-file
        use_segment_mode = (
            good_segs
            and total_segs > 0
            and len(good_segs) < total_segs
        )
        # Junk files are always full-file retranscribe (since no good segments found)
        if r["classification"] == "junk":
            use_segment_mode = False
            target_root = output_root_junk
        else:
            target_root = output_root

        mode_label = f"segments {len(good_segs)}/{total_segs}" if use_segment_mode else "full file"

        # Progress header with ETA
        elapsed = time.time() - job_start
        if file_times:
            avg_time = sum(file_times) / len(file_times)
            remaining = avg_time * (len(queue) - i)
            eta_str = f"  ETA: {format_eta(remaining)}"
        else:
            eta_str = ""

        print(f"[{i+1}/{len(queue)}] {r['rel_path']}  ({mode_label})  elapsed: {format_eta(elapsed)}{eta_str}")
        print(f"  Audio: {audio_path}")
        if r["classification"] == "junk":
             print(f"  Target: [JUNK] {target_root}")

        # Ensure output subdirectory exists
        target_dir = os.path.join(target_root, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        out_txt = os.path.join(target_root, base_name + ".txt")
        out_srt = os.path.join(target_root, base_name + ".srt")

        try:
            if use_segment_mode:
                # --- Segment-only mode ---
                # Load audio, slice to good time ranges, transcribe each
                audio = whisper.load_audio(audio_path)
                final_segments = []

                for seg in good_segs:
                    start_sample = int(seg["start"] * 16000)
                    end_sample = int(seg["end"] * 16000)

                    # Safety bounds
                    if start_sample >= len(audio):
                        continue
                    end_sample = min(end_sample, len(audio))

                    clip = audio[start_sample:end_sample]
                    if len(clip) < 1600:  # < 0.1s
                        continue

                    res = model.transcribe(clip, verbose=False)
                    final_text = res["text"].strip()

                    if final_text:
                        final_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": final_text,
                        })

                # Save
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(s["text"] for s in final_segments))
                save_srt(final_segments, out_srt)
                print(f"  Saved ({len(final_segments)} segments)")

            else:
                # --- Full-file mode ---
                result = model.transcribe(audio_path, verbose=False)

                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                save_srt(result["segments"], out_srt)
                print(f"  Saved (full file)")

        except Exception as e:
            print(f"  [ERROR] Failed: {e}")

        # Track timing
        file_duration = time.time() - file_start
        file_times.append(file_duration)
        print(f"  Time: {format_eta(file_duration)}")

    # Cleanup
    print(f"\nUnloading '{model_name}' model...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    total_elapsed = time.time() - job_start
    print(f"\n{'='*60}")
    print(f"COMPLETE in {format_eta(total_elapsed)}")
    print(f"  Files processed: {len(file_times)}")
    if file_times:
        print(f"  Avg per file:    {format_eta(sum(file_times) / len(file_times))}")
    print(f"  Output:          {output_root}")
    print(f"{'='*60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Triage small.en transcripts and re-transcribe with large Whisper model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Report only (no GPU needed):
  python retranscribe_large.py C:\\Users\\k2\\videos-daytime_audio_transcript --report-only

  # Full run (classify + re-transcribe):
  python retranscribe_large.py C:\\Users\\k2\\videos-daytime_audio_transcript

  # Adjust thresholds:
  python retranscribe_large.py C:\\Users\\k2\\videos-daytime_audio_transcript --junk-threshold 0.25 --good-threshold 0.7
        """
    )
    parser.add_argument("transcript_folder",
                        help="Folder containing small.en transcripts (.txt files)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only classify and generate triage CSV, skip re-transcription")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe even if large-model output already exists")
    parser.add_argument("--model", default="large",
                        help="Whisper model for re-transcription (default: large)")
    parser.add_argument("--junk-threshold", type=float, default=0.3,
                        help="Unique word ratio below this = junk (default: 0.3)")
    parser.add_argument("--good-threshold", type=float, default=0.6,
                        help="Unique word ratio above this = good (default: 0.6)")
    parser.add_argument("--include-junk", action="store_true",
                        help="Also re-transcribe junk files (saved to *_large_junk folder)")

    args = parser.parse_args()

    if not os.path.isdir(args.transcript_folder):
        print(f"Error: Not a directory: {args.transcript_folder}")
        sys.exit(1)

    # --- Stage 1 & 2: Scan, classify, report ---
    print(f"Scanning transcripts in: {args.transcript_folder}")
    results = scan_and_classify(
        args.transcript_folder,
        junk_threshold=args.junk_threshold,
        good_threshold=args.good_threshold,
    )

    if not results:
        print("No .txt files found.")
        sys.exit(0)

    print_summary(results)

    # Save triage report next to the transcript folder
    transcript_root_abs = os.path.abspath(args.transcript_folder)
    parent_dir = os.path.dirname(transcript_root_abs)
    report_path = os.path.join(parent_dir, "triage_report.csv")
    write_triage_report(results, report_path)

    if args.report_only:
        print("Done (report-only mode).")
        return

    # --- Stage 3: Re-transcribe ---
    retranscribe(
        results,
        args.transcript_folder,
        model_name=args.model,
        force=args.force,
        include_junk=args.include_junk,
    )


if __name__ == "__main__":
    main()

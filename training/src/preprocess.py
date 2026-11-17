import os
import glob
import hashlib
import subprocess
import tempfile
import librosa
import soundfile as sf
import numpy as np
import imageio_ffmpeg

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
TARGET_CLASSES = ["Hunger", "Pain_Discomfort", "Sleepiness"]
TARGET_SR = 16000  # Required sample rate for YAMNet
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def load_audio_any_format(file_path, target_sr=TARGET_SR):
    """Loads WAV natively or decodes .3gp/.ogg/.mp3/.m4a via bundled FFmpeg."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        y, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return y

    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            FFMPEG_EXE, "-y", "-i", file_path,
            "-ac", "1", "-ar", str(target_sr),
            "-loglevel", "error", tmp_wav
        ]
        subprocess.run(cmd, check=True)
        y, _ = librosa.load(tmp_wav, sr=target_sr, mono=True)
        return y
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def get_audio_fingerprint(y):
    """Creates a unique MD5 fingerprint from the sound wave to catch renamed duplicate files."""
    rounded = np.round(y, decimals=3).tobytes()
    return hashlib.md5(rounded).hexdigest()


def clean_dataset():
    print("=== Step 1: Removing Duplicates & Standardizing Audio ===")
    seen_hashes = set()
    stats = {cls: {"kept": 0, "duplicates": 0, "too_short": 0, "errors": 0} for cls in TARGET_CLASSES}

    for cls in TARGET_CLASSES:
        in_folder = os.path.join(RAW_DIR, cls)
        out_folder = os.path.join(CLEAN_DIR, cls)

        if not os.path.exists(in_folder):
            print(f"[Warning] Folder not found: {in_folder}")
            continue

        os.makedirs(out_folder, exist_ok=True)
        for old_file in glob.glob(os.path.join(out_folder, "*.wav")):
            os.remove(old_file)

        files = []
        for ext in ("*.wav", "*.3gp", "*.mp3", "*.m4a", "*.ogg"):
            files.extend(glob.glob(os.path.join(in_folder, "**", ext), recursive=True))

        # Prioritize original UUID WAV recordings (longer stems) before short augmented copies (e.g., 248c.wav)
        files = sorted(
            set(files),
            key=lambda p: (0 if p.lower().endswith(".wav") else 1, -len(os.path.basename(p)), os.path.basename(p))
        )
        print(f"Checking {len(files)} files in {cls}...")

        for file_path in files:
            try:
                y = load_audio_any_format(file_path, target_sr=TARGET_SR)

                # Normalize volume
                peak = np.max(np.abs(y))
                if peak > 0:
                    y = y / peak

                # Trim silence from start and end (28 dB threshold preserves high-SNR cry frames)
                y_trimmed, _ = librosa.effects.trim(y, top_db=28)

                # Skip clips shorter than 0.8 seconds
                if len(y_trimmed) < int(0.8 * TARGET_SR):
                    stats[cls]["too_short"] += 1
                    continue

                # Skip exact duplicate recordings
                fp = get_audio_fingerprint(y_trimmed)
                if fp in seen_hashes:
                    stats[cls]["duplicates"] += 1
                    continue
                seen_hashes.add(fp)

                orig_stem = os.path.splitext(os.path.basename(file_path))[0]
                out_name = f"{stats[cls]['kept'] + 1:04d}__{orig_stem}.wav"
                sf.write(os.path.join(out_folder, out_name), y_trimmed, TARGET_SR)
                stats[cls]["kept"] += 1

            except Exception as e:
                stats[cls]["errors"] += 1
                print(f"[Error] {os.path.basename(file_path)}: {e}")

    print("\n--- Clean Dataset Report ---")
    total_kept = 0
    for cls, s in stats.items():
        total_kept += s["kept"]
        print(f"{cls:<16} | Kept: {s['kept']:<4} | Duplicates Removed: {s['duplicates']:<4} | Too Short: {s['too_short']:<3} | Errors: {s['errors']}")
    print(f"Total Clean Recordings Ready: {total_kept}\n")


if __name__ == "__main__":
    clean_dataset()
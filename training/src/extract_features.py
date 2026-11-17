import os
import glob
import librosa
import numpy as np
import pandas as pd
import tensorflow_hub as hub

CLEAN_DIR = os.path.join("data", "clean")
OUTPUT_CSV = os.path.join("data", "yamnet_features.csv")
TARGET_CLASSES = ["Hunger", "Pain_Discomfort", "Sleepiness"]
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

def extract_yamnet_features():
    print("=== Step 2: Loading Pre-Trained YAMNet AI ===")
    yamnet = hub.load(YAMNET_URL)
    print("YAMNet loaded successfully!\n")

    rows = []

    for cls in TARGET_CLASSES:
        folder = os.path.join(CLEAN_DIR, cls)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        print(f"Processing {len(files)} {cls} recordings...")

        for idx, file_path in enumerate(files, start=1):
            try:
                # Load 16kHz mono audio as float32
                y, _ = librosa.load(file_path, sr=16000, mono=True)
                y = y.astype(np.float32)

                # Pad to at least 1 second if slightly shorter
                if len(y) < 16000:
                    y = np.pad(y, (0, 16000 - len(y)))

                # Run audio through pre-trained YAMNet
                _, embeddings, _ = yamnet(y)
                emb_np = embeddings.numpy()  # Shape: (frames, 1024)

                # Combine average and peak sound patterns across the cry (2048 features)
                emb_mean = np.mean(emb_np, axis=0)
                emb_max = np.max(emb_np, axis=0)
                feature_vec = np.concatenate([emb_mean, emb_max])

                row = {f"yam_{i}": float(val) for i, val in enumerate(feature_vec)}
                row["filename"] = os.path.basename(file_path)
                row["label"] = cls
                rows.append(row)

                if idx % 100 == 0:
                    print(f"  -> Finished {idx}/{len(files)} in {cls}")

            except Exception as e:
                print(f"[Error] {os.path.basename(file_path)}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved {len(df)} recordings with {df.shape[1] - 2} AI features to '{OUTPUT_CSV}'.\n")

if __name__ == "__main__":
    extract_yamnet_features() 
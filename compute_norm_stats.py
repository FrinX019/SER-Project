"""
Recomputes the training-set normalization stats (mean & std) from the local
CREMA-D folder using the exact same split logic as the training notebook.
Saves: norm_stats.npz  (keys: mean, std)
"""
import os, random
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CREMA_DIR   = os.path.join(os.path.dirname(__file__), 'Crema')
OUT_PATH    = os.path.join(os.path.dirname(__file__), 'norm_stats.npz')
SEED        = 42
SAMPLE_RATE = 22050
DURATION    = 3.0
N_MFCC      = 40
MAX_FRAMES  = 200
EMOTION_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}

random.seed(SEED)
np.random.seed(SEED)

# ── 1. Parse filenames (same as notebook) ──────────────────
records = []
for fname in sorted(os.listdir(CREMA_DIR)):
    if not fname.endswith('.wav'):
        continue
    parts = fname.replace('.wav', '').split('_')
    if len(parts) != 4:
        continue
    _, _, emotion_code, _ = parts
    if emotion_code not in EMOTION_MAP:
        continue
    records.append({
        'filepath':      os.path.join(CREMA_DIR, fname),
        'emotion_label': EMOTION_MAP[emotion_code],
    })

labels = np.array([r['emotion_label'] for r in records])

# ── 2. Same 70/15/15 split ─────────────────────────────────
idx       = np.arange(len(records))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=labels)
print(f"Train split: {len(idx_train)} files — extracting MFCCs...")

# ── 3. Extract MFCCs for training files ────────────────────
def load_mfcc(filepath):
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    target = int(SAMPLE_RATE * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    y    = y[:target]
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfcc.shape[0] < MAX_FRAMES:
        mfcc = np.vstack([mfcc, np.zeros((MAX_FRAMES - mfcc.shape[0], N_MFCC))])
    return mfcc[:MAX_FRAMES].astype(np.float32)

X = np.zeros((len(idx_train), MAX_FRAMES, N_MFCC), dtype=np.float32)
for i, idx_i in enumerate(tqdm(idx_train, desc="Extracting")):
    try:
        X[i] = load_mfcc(records[idx_i]['filepath'])
    except Exception as e:
        print(f"  Error: {e}")

# ── 4. Compute & save stats ────────────────────────────────
# Shape: (1, 1, 40) — same as notebook: mean over samples and time, per coeff
mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, 40)
std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8

np.savez(OUT_PATH, mean=mean, std=std)
print(f"\nSaved normalization stats → {OUT_PATH}")
print(f"  mean shape: {mean.shape}, std shape: {std.shape}")
print(f"  mean range: [{mean.min():.3f}, {mean.max():.3f}]")
print(f"  std  range: [{std.min():.3f},  {std.max():.3f}]")

"""
Speech Emotion Recognition — FastAPI Backend
Run: /opt/anaconda3/bin/uvicorn backend:app --reload --port 8000
"""

import os, math, tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────
#  Model Architecture
# ──────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SpeechEmotionTransformer(nn.Module):
    def __init__(self, input_size=40, d_model=128, nhead=8,
                 num_layers=4, dim_ff=256, dropout=0.2, num_classes=6):
        super().__init__()
        self.input_proj   = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
SAMPLE_RATE  = 22050
DURATION     = 3.0
N_MFCC       = 40
MAX_FRAMES   = 200
EMOTION_FULL = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'best_transformer.pth')
NORM_STATS   = os.path.join(os.path.dirname(__file__), 'norm_stats.npz')
TRAIN_FEAT   = os.path.join(os.path.dirname(__file__), 'features', 'mfcc_train.npy')

# ──────────────────────────────────────────────
#  Load model at startup
# ──────────────────────────────────────────────
model     = None
norm_mean = None
norm_std  = None

def load_model():
    global model, norm_mean, norm_std
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model weights not found at {MODEL_PATH}")
        print("       Place best_transformer.pth next to backend.py and restart.")
        return
    m = SpeechEmotionTransformer().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    model = m
    if os.path.exists(NORM_STATS):
        stats     = np.load(NORM_STATS)
        norm_mean = torch.FloatTensor(stats['mean'])   # (1, 1, 40)
        norm_std  = torch.FloatTensor(stats['std'])    # (1, 1, 40)
        print(f"[INFO] Model + normalization stats loaded from norm_stats.npz. Device: {DEVICE}")
    elif os.path.exists(TRAIN_FEAT):
        X_tr      = torch.FloatTensor(np.load(TRAIN_FEAT))
        norm_mean = X_tr.mean(dim=(0, 1), keepdim=True)
        norm_std  = X_tr.std(dim=(0, 1),  keepdim=True) + 1e-8
        print(f"[INFO] Model + normalization stats loaded from train features. Device: {DEVICE}")
    else:
        print(f"[WARN] Model loaded WITHOUT proper normalization — predictions may be inaccurate.")

load_model()

# ──────────────────────────────────────────────
#  Audio helpers
# ──────────────────────────────────────────────
def extract_features(filepath: str) -> torch.Tensor:
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    target = int(SAMPLE_RATE * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    y = y[:target]

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfcc.shape[0] < MAX_FRAMES:
        mfcc = np.vstack([mfcc, np.zeros((MAX_FRAMES - mfcc.shape[0], N_MFCC))])
    mfcc = mfcc[:MAX_FRAMES].astype(np.float32)

    x = torch.FloatTensor(mfcc).unsqueeze(0)   # (1, 200, 40)
    if norm_mean is not None:
        x = (x - norm_mean) / norm_std
    else:
        x = (x - x.mean()) / (x.std() + 1e-8)
    return x


# ──────────────────────────────────────────────
#  FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(title="SER API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
def status():
    return {"model_loaded": model is not None, "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place best_transformer.pth next to backend.py and restart."
        )

    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        x      = extract_features(tmp_path)
        with torch.no_grad():
            logits = model(x.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return {
            "emotion":     EMOTION_FULL[idx],
            "confidence":  round(float(probs[idx]) * 100, 2),
            "all_emotions": [
                {"label": label, "probability": round(float(p) * 100, 2)}
                for label, p in zip(EMOTION_FULL, probs)
            ]
        }
    finally:
        os.unlink(tmp_path)

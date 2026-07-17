# ============================================================
# ULTRA ADVANCED EMG RESIDUAL MODELING
# CNN + TCN + TRANSFORMER + ATTENTION
# OPTIMIZED FOR KAGGLE P100 GPU
# TARGET: PREDICT a_emg1-4 USING s_emg1-6 + BIOMECHANICS
# ============================================================

# ============================================================
# INSTALLS (KAGGLE)
# ============================================================
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install scikit-learn scipy pandas matplotlib joblib

# ============================================================
# IMPORTS
# ============================================================

import os
import gc
import math
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from scipy.stats import pearsonr

import joblib

# ============================================================
# GPU CHECK
# ============================================================

print("="*60)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

else:
    print("GPU NOT DETECTED")
    device = torch.device("cpu")

print("Using device:", device)
print("="*60)

# ============================================================
# CONFIG
# ============================================================

class CFG:

    # DATA
    DATA_PATH = "/kaggle/input/datasets/devilns/modeling/merged_with_weights.csv"

    # WINDOWING
    SEQ_LEN = 264
    STRIDE = 4

    # TRAINING
    BATCH_SIZE = 264
    EPOCHS = 100
    LR = 1e-3
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-5

    # MODEL
    D_MODEL = 512
    N_HEADS = 8
    N_LAYERS = 3
    DROPOUT = 0.10

    # TRAINING
    PATIENCE = 20
    CLIP = 1.0
    SEED = 42
    TEST_SIZE = 0.20

    # AMP
    USE_AMP = True

    # TTA
    TTA = 3

    # SAVE
    MODEL_NAME = "best_emg_residual_model.pth"

cfg = CFG()

# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(cfg.SEED)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

df = pd.read_csv(cfg.DATA_PATH, low_memory=False)

print("Raw shape:", df.shape)

# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED = [

    'e_angle',
    'e_tau',
    's_angle',
    's_tau',

    's_emg1',
    's_emg2',
    's_emg3',
    's_emg4',
    's_emg5',
    's_emg6',

    'a_emg1',
    'a_emg2',
    'a_emg3',
    'a_emg4',

    'Weight'
]

for c in REQUIRED:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=REQUIRED).reset_index(drop=True)

print("Clean shape:", df.shape)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nFeature engineering...")

# ------------------------------------------------------------
# DERIVATIVES
# ------------------------------------------------------------

for col in ['e_angle', 'e_tau', 's_angle', 's_tau']:

    df[f'{col}_vel'] = df[col].diff().fillna(0)

    df[f'{col}_acc'] = (
        df[f'{col}_vel']
        .diff()
        .fillna(0)
    )

# ------------------------------------------------------------
# EMG FEATURES
# ------------------------------------------------------------

SIM_COLS = [
    's_emg1',
    's_emg2',
    's_emg3',
    's_emg4',
    's_emg5',
    's_emg6'
]

df['emg_mean'] = df[SIM_COLS].mean(axis=1)
df['emg_std']  = df[SIM_COLS].std(axis=1)
df['emg_max']  = df[SIM_COLS].max(axis=1)
df['emg_min']  = df[SIM_COLS].min(axis=1)
df['emg_sum']  = df[SIM_COLS].sum(axis=1)

# ------------------------------------------------------------
# RATIOS
# ------------------------------------------------------------

df['tau_emg_ratio'] = df['s_tau'] / (df['emg_sum'] + 1e-6)
df['angle_diff']    = df['e_angle'] - df['s_angle']
df['tau_diff']      = df['e_tau'] - df['s_tau']

# ------------------------------------------------------------
# ROLLING FEATURES
# ------------------------------------------------------------

for col in ['s_emg1', 's_emg2', 's_emg3', 's_emg4']:

    df[f'{col}_roll_mean'] = (
        df[col]
        .rolling(10, min_periods=1)
        .mean()
    )

    df[f'{col}_roll_std'] = (
        df[col]
        .rolling(10, min_periods=1)
        .std()
        .fillna(0)
    )

# ============================================================
# FEATURES
# ============================================================

FEATURE_COLS = [

    's_emg1',
    's_emg2',
    's_emg3',
    's_emg4',
    's_emg5',
    's_emg6',

    'e_angle',
    'e_tau',
    's_angle',
    's_tau',

    'Weight',

    'e_angle_vel',
    'e_tau_vel',
    's_angle_vel',
    's_tau_vel',

    'e_angle_acc',
    'e_tau_acc',
    's_angle_acc',
    's_tau_acc',

    'emg_mean',
    'emg_std',
    'emg_max',
    'emg_min',
    'emg_sum',

    'tau_emg_ratio',
    'angle_diff',
    'tau_diff',

    's_emg1_roll_mean',
    's_emg2_roll_mean',
    's_emg3_roll_mean',
    's_emg4_roll_mean',

    's_emg1_roll_std',
    's_emg2_roll_std',
    's_emg3_roll_std',
    's_emg4_roll_std',
]

TARGET_ACTUAL = [
    'a_emg1',
    'a_emg2',
    'a_emg3',
    'a_emg4'
]

TARGET_SIM = [
    's_emg1',
    's_emg2',
    's_emg3',
    's_emg4'
]

# ============================================================
# ARRAYS
# ============================================================

X = df[FEATURE_COLS].values.astype(np.float32)

actual = df[TARGET_ACTUAL].values.astype(np.float32)

simulated = df[TARGET_SIM].values.astype(np.float32)

# RESIDUAL TARGET
residual = actual - simulated

print("\nInput dim:", X.shape[1])

# ============================================================
# SCALING
# ============================================================

print("\nScaling...")

x_scaler = RobustScaler()
r_scaler = RobustScaler()

X = x_scaler.fit_transform(X).astype(np.float32)

residual = (
    r_scaler
    .fit_transform(residual)
    .astype(np.float32)
)

joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(r_scaler, "r_scaler.save")

# ============================================================
# WINDOWING
# ============================================================

print("\nCreating windows...")

def create_windows(X, Y, SIM, ACT, seq_len, stride):

    Xs = []
    Ys = []
    Ss = []
    As = []

    for i in range(0, len(X) - seq_len, stride):

        Xs.append(X[i:i+seq_len])
        Ys.append(Y[i:i+seq_len])
        Ss.append(SIM[i:i+seq_len])
        As.append(ACT[i:i+seq_len])

    return (
        np.array(Xs),
        np.array(Ys),
        np.array(Ss),
        np.array(As)
    )

X_seq, Y_seq, SIM_seq, ACT_seq = create_windows(
    X,
    residual,
    simulated,
    actual,
    cfg.SEQ_LEN,
    cfg.STRIDE
)

print("X windows:", X_seq.shape)
print("Y windows:", Y_seq.shape)

# ============================================================
# SPLIT
# ============================================================

idx = np.arange(len(X_seq))

tr_idx, te_idx = train_test_split(
    idx,
    test_size=cfg.TEST_SIZE,
    random_state=cfg.SEED,
    shuffle=True
)

X_train = X_seq[tr_idx]
X_test  = X_seq[te_idx]

Y_train = Y_seq[tr_idx]
Y_test  = Y_seq[te_idx]

SIM_train = SIM_seq[tr_idx]
SIM_test  = SIM_seq[te_idx]

ACT_train = ACT_seq[tr_idx]
ACT_test  = ACT_seq[te_idx]

print("\nTrain:", len(X_train))
print("Test :", len(X_test))

# ============================================================
# DATASET
# ============================================================

class EMGDataset(Dataset):

    def __init__(
        self,
        X,
        Y,
        SIM,
        ACT,
        augment=False
    ):

        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

        self.SIM = torch.FloatTensor(SIM)
        self.ACT = torch.FloatTensor(ACT)

        self.augment = augment

    def __len__(self):
        return len(self.X)

    def augment_signal(self, x):

        # gaussian noise
        x = x + torch.randn_like(x) * 0.003

        # scale jitter
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.04
        x = x * scale

        # shift
        if torch.rand(1) > 0.5:

            shift = torch.randint(1, 5, (1,)).item()

            x = torch.roll(x, shift, dims=0)

        return x

    def __getitem__(self, idx):

        x = self.X[idx]

        if self.augment:
            x = self.augment_signal(x)

        return (
            x,
            self.Y[idx],
            self.SIM[idx],
            self.ACT[idx]
        )

# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    EMGDataset(
        X_train,
        Y_train,
        SIM_train,
        ACT_train,
        augment=True
    ),
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    EMGDataset(
        X_test,
        Y_test,
        SIM_test,
        ACT_test,
        augment=False
    ),
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer(
            'pe',
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]

# ============================================================
# SE BLOCK
# ============================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels, channels // reduction),

            nn.GELU(),

            nn.Linear(channels // reduction, channels),

            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _ = x.shape

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y

# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel=3,
        dilation=1
    ):

        super().__init__()

        pad = ((kernel - 1) * dilation) // 2

        self.conv = nn.Sequential(

            nn.Conv1d(
                in_ch,
                out_ch,
                kernel,
                padding=pad,
                dilation=dilation
            ),

            nn.BatchNorm1d(out_ch),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Conv1d(
                out_ch,
                out_ch,
                kernel,
                padding=pad,
                dilation=dilation
            ),

            nn.BatchNorm1d(out_ch)
        )

        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

        self.act = nn.GELU()

    def forward(self, x):

        return self.act(
            self.conv(x) + self.skip(x)
        )

# ============================================================
# MULTI SCALE BLOCK
# ============================================================

class MultiScaleBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        branch = out_ch // 4

        self.b1 = ResBlock(in_ch, branch, kernel=3)
        self.b2 = ResBlock(in_ch, branch, kernel=5)
        self.b3 = ResBlock(in_ch, branch, kernel=9)
        self.b4 = ResBlock(in_ch, branch, kernel=15)

        self.fuse = nn.Sequential(

            nn.Conv1d(out_ch, out_ch, 1),

            nn.BatchNorm1d(out_ch),

            nn.GELU()
        )

    def forward(self, x):

        x = torch.cat([
            self.b1(x),
            self.b2(x),
            self.b3(x),
            self.b4(x)
        ], dim=1)

        return self.fuse(x)

# ============================================================
# MAIN MODEL
# ============================================================

class UltraEMGModel(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim=4
    ):

        super().__init__()

        # ----------------------------------------------------
        # STEM
        # ----------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv1d(input_dim, 64, 1),

            nn.BatchNorm1d(64),

            nn.GELU()
        )

        # ----------------------------------------------------
        # MULTISCALE
        # ----------------------------------------------------

        self.ms1 = MultiScaleBlock(64, 128)

        self.se1 = SEBlock(128)

        self.ms2 = MultiScaleBlock(128, cfg.D_MODEL)

        self.se2 = SEBlock(cfg.D_MODEL)

        # ----------------------------------------------------
        # DILATED TCN
        # ----------------------------------------------------

        self.tcn = nn.Sequential(

            ResBlock(
                cfg.D_MODEL,
                cfg.D_MODEL,
                dilation=1
            ),

            ResBlock(
                cfg.D_MODEL,
                cfg.D_MODEL,
                dilation=2
            ),

            ResBlock(
                cfg.D_MODEL,
                cfg.D_MODEL,
                dilation=4
            ),

            ResBlock(
                cfg.D_MODEL,
                cfg.D_MODEL,
                dilation=8
            ),
        )

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        self.pos = PositionalEncoding(cfg.D_MODEL)

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=cfg.D_MODEL,

            nhead=cfg.N_HEADS,

            dim_feedforward=cfg.D_MODEL * 4,

            dropout=cfg.DROPOUT,

            activation='gelu',

            batch_first=True,

            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.N_LAYERS
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.head = nn.Sequential(

            nn.Linear(cfg.D_MODEL, 256),

            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(256, 128),

            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):

        # [B,T,F] -> [B,F,T]
        x = x.permute(0, 2, 1)

        x = self.stem(x)

        x = self.ms1(x)

        x = self.se1(x)

        x = self.ms2(x)

        x = self.se2(x)

        x = self.tcn(x)

        # [B,C,T] -> [B,T,C]
        x = x.permute(0, 2, 1)

        x = self.pos(x)

        x = self.transformer(x)

        out = self.head(x)

        return out

# ============================================================
# MODEL
# ============================================================

model = UltraEMGModel(
    input_dim=len(FEATURE_COLS),
    output_dim=4
).to(device)

print("\nModel parameters:")

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"{params:,}")

# ============================================================
# LOSSES
# ============================================================

huber = nn.HuberLoss(delta=0.5)

def pearson_loss(pred, target):

    pred = pred.reshape(-1, pred.shape[-1])

    target = target.reshape(-1, target.shape[-1])

    loss = 0

    for i in range(pred.shape[-1]):

        vx = pred[:, i] - pred[:, i].mean()

        vy = target[:, i] - target[:, i].mean()

        corr = (vx * vy).sum() / (
            vx.norm() * vy.norm() + 1e-8
        )

        loss += (1 - corr)

    return loss / pred.shape[-1]

def smoothness_loss(pred):

    diff = pred[:, 1:] - pred[:, :-1]

    return diff.pow(2).mean()

def combined_loss(pred, target):

    l1 = huber(pred, target)

    l2 = pearson_loss(pred, target)

    l3 = smoothness_loss(pred)

    return (
        0.65 * l1 +
        0.30 * l2 +
        0.05 * l3
    )

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=cfg.LR,

    weight_decay=cfg.WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=cfg.EPOCHS,

    eta_min=cfg.MIN_LR
)

scaler = torch.cuda.amp.GradScaler(
    enabled=cfg.USE_AMP
)

# ============================================================
# TRAINING
# ============================================================

print("\n" + "="*60)
print("TRAINING")
print("="*60)

best_loss = np.inf
patience = 0

train_losses = []
val_losses = []

for epoch in range(cfg.EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0

    for Xb, Yb, _, _ in train_loader:

        Xb = Xb.to(device, non_blocking=True)

        Yb = Yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(
            enabled=cfg.USE_AMP
        ):

            pred = model(Xb)

            loss = combined_loss(pred, Yb)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.CLIP
        )

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for Xb, Yb, _, _ in test_loader:

            Xb = Xb.to(device)

            Yb = Yb.to(device)

            pred = model(Xb)

            loss = combined_loss(pred, Yb)

            val_loss += loss.item()

    val_loss /= len(test_loader)

    scheduler.step()

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    # ========================================================
    # SAVE
    # ========================================================

    if val_loss < best_loss:

        best_loss = val_loss

        patience = 0

        torch.save(
            model.state_dict(),
            cfg.MODEL_NAME
        )

    else:

        patience += 1

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.5f} | "
        f"Val {val_loss:.5f} | "
        f"LR {optimizer.param_groups[0]['lr']:.2e}"
    )

    if patience >= cfg.PATIENCE:

        print("\nEarly stopping")

        break

    gc.collect()

    torch.cuda.empty_cache()

# ============================================================
# LOAD BEST
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        cfg.MODEL_NAME,
        map_location=device
    )
)

model.eval()

# ============================================================
# INFERENCE
# ============================================================

print("\nRunning inference...")

all_preds = []
all_sims = []
all_acts = []

with torch.no_grad():

    for Xb, _, SIMb, ACTb in test_loader:

        Xb = Xb.to(device)

        tta_preds = []

        for _ in range(cfg.TTA):

            noisy = Xb + (
                torch.randn_like(Xb) * 0.002
            )

            pred = model(noisy)

            pred = pred.cpu().numpy()

            pred = (
                r_scaler
                .inverse_transform(
                    pred.reshape(-1, 4)
                )
                .reshape(pred.shape)
            )

            tta_preds.append(pred)

        pred = np.mean(tta_preds, axis=0)

        all_preds.append(pred)

        all_sims.append(SIMb.numpy())

        all_acts.append(ACTb.numpy())

preds = np.concatenate(all_preds)

sims = np.concatenate(all_sims)

acts = np.concatenate(all_acts)

# ============================================================
# CORRECTED SIGNAL
# ============================================================

corrected = sims + preds

# ============================================================
# FLATTEN
# ============================================================

pred_flat = corrected.reshape(-1, 4)

sim_flat = sims.reshape(-1, 4)

act_flat = acts.reshape(-1, 4)

# ============================================================
# METRICS
# ============================================================

print("\n" + "="*60)
print("RESULTS")
print("="*60)

MUSCLE = [
    "EMG1",
    "EMG2",
    "EMG3",
    "EMG4"
]

def metrics(actual, pred):

    rmse = np.sqrt(
        mean_squared_error(actual, pred)
    )

    mae = mean_absolute_error(actual, pred)

    r2 = r2_score(actual, pred)

    corr = pearsonr(actual, pred)[0]

    return rmse, mae, r2, corr

for i in range(4):

    print(f"\n{MUSCLE[i]}")

    b_rmse, b_mae, b_r2, b_corr = metrics(
        act_flat[:, i],
        sim_flat[:, i]
    )

    a_rmse, a_mae, a_r2, a_corr = metrics(
        act_flat[:, i],
        pred_flat[:, i]
    )

    print(
        f"Before -> "
        f"RMSE={b_rmse:.5f} "
        f"MAE={b_mae:.5f} "
        f"R2={b_r2:.5f} "
        f"r={b_corr:.5f}"
    )

    print(
        f"After  -> "
        f"RMSE={a_rmse:.5f} "
        f"MAE={a_mae:.5f} "
        f"R2={a_r2:.5f} "
        f"r={a_corr:.5f}"
    )

# ============================================================
# OVERALL
# ============================================================

print("\n" + "="*60)

overall_r2 = r2_score(
    act_flat,
    pred_flat
)

overall_rmse = np.sqrt(
    mean_squared_error(
        act_flat,
        pred_flat
    )
)

overall_mae = mean_absolute_error(
    act_flat,
    pred_flat
)

print("OVERALL RESULTS")
print("="*60)

print("RMSE:", round(overall_rmse, 5))
print("MAE :", round(overall_mae, 5))
print("R2  :", round(overall_r2, 5))

# ============================================================
# VISUALIZATION
# ============================================================

print("\nSaving plots...")

N = 1000

fig, axes = plt.subplots(
    4,
    1,
    figsize=(18, 14)
)

for i in range(4):

    axes[i].plot(
        act_flat[:N, i],
        label="Actual",
        linewidth=1.5
    )

    axes[i].plot(
        sim_flat[:N, i],
        label="Simulated",
        alpha=0.8
    )

    axes[i].plot(
        pred_flat[:N, i],
        label="Corrected",
        alpha=0.8
    )

    axes[i].set_title(MUSCLE[i])

    axes[i].legend()

    axes[i].grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    "emg_results.png",
    dpi=200
)

# ============================================================
# TRAINING CURVE
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(
    train_losses,
    label="Train"
)

plt.plot(
    val_losses,
    label="Validation"
)

plt.legend()

plt.grid(alpha=0.2)

plt.title("Training Curve")

plt.savefig(
    "training_curve.png",
    dpi=200
)

# ============================================================
# SAVE CSV
# ============================================================

out = pd.DataFrame(

    np.hstack([
        act_flat,
        sim_flat,
        pred_flat
    ]),

    columns=[

        "actual_1",
        "actual_2",
        "actual_3",
        "actual_4",

        "sim_1",
        "sim_2",
        "sim_3",
        "sim_4",

        "corrected_1",
        "corrected_2",
        "corrected_3",
        "corrected_4"
    ]
)

out.to_csv(
    "emg_predictions.csv",
    index=False
)

print("\nSaved:")
print(" - emg_results.png")
print(" - training_curve.png")
print(" - emg_predictions.csv")

print("\nDONE")
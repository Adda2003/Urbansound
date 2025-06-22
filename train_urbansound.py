import sys
import os
import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional: for a nicer progress bar (install via `pip install tqdm`)
try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

# === 1) Configuration & Paths ===
root_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(root_dir, "cache", "urbansound8k", "raw")
metadata_path = os.path.join(raw_dir, "UrbanSound8K.csv")

# Verify paths
assert os.path.isfile(metadata_path), f"Metadata CSV not found: {metadata_path}"
assert os.path.isdir(raw_dir), f"Raw data dir not found: {raw_dir}"

# === 2) Hyperparameters ===
sr = 22050
n_mfcc = 13
win_ms = 25
hop_ms = 10
T_max = 400
batch_size = 16
epochs = 5
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 3) Dataset Definition ===
class UrbanSoundDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_fft = int(sr * win_ms / 1000)
        self.hop_length = int(sr * hop_ms / 1000)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold = row['fold']
        fname = row['slice_file_name']
        label = int(row['classID'])
        wav_path = os.path.join(raw_dir, f"fold{fold}", fname)
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        y, _ = librosa.load(wav_path, sr=sr)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft
        ).T
        if mfcc.shape[0] < T_max:
            pad_amt = T_max - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_amt), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:T_max, :]
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# === 4) Prepare DataLoaders ===
df = pd.read_csv(metadata_path)
train_df = df[df['fold'] <= 8]
val_df = df[df['fold'] == 9]

dataset_train = UrbanSoundDataset(train_df)
dataset_val = UrbanSoundDataset(val_df)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

print(f"Loaded {len(dataset_train)} training and {len(dataset_val)} validation samples.")

# === 5) Model & Optimizer ===
model = nn.LSTM(input_size=n_mfcc, hidden_size=128, num_layers=2, batch_first=True).to(device)
classifier = nn.Linear(128, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)

print("Model and classifier initialized.")
# sys.exit(0)
# === 6) Training & Validation Loop ===
for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    model.train()
    running_loss = 0.0
    iterator = tqdm(train_loader, desc="Training", unit="batch") if use_tqdm else train_loader
    for i, (X, Y) in enumerate(iterator, 1):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        _, (h_n, _) = model(X)
        logits = classifier(h_n[-1])
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        if not use_tqdm and i % 50 == 0:
            print(f"  Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")
    train_loss = running_loss / len(dataset_train)
    print(f"Epoch {epoch} Training Loss: {train_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    for X, Y in val_loader:
        X, Y = X.to(device), Y.to(device)
        _, (h_n, _) = model(X)
        preds = classifier(h_n[-1]).argmax(dim=1)
        correct += (preds == Y).sum().item()
        total += Y.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch} Validation Accuracy: {val_acc:.4f}")

# === 7) Save Checkpoint ===
ckpt = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
torch.save(ckpt, os.path.join(root_dir, 'urbansound_lstm.pth'))
print("\nTraining complete. Model saved to 'urbansound_lstm.pth'.")

import os
import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==== Paths ====
root_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(root_dir, "cache", "urbansound8k", "raw")
metadata_path = os.path.join(raw_dir, "UrbanSound8K.csv")
ckpt_path = os.path.join(root_dir, "urbansound_lstm.pth")

# ==== Hyperparameters (should match train script) ====
sr = 22050
n_mfcc = 13
win_ms = 25
hop_ms = 10
T_max = 400
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ====
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

# ==== Model & Load Weights ====
model = nn.LSTM(input_size=n_mfcc, hidden_size=128, num_layers=2, batch_first=True).to(device)
classifier = nn.Linear(128, 10).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])
model.eval()
classifier.eval()

# ==== Prepare Test Data ====
df = pd.read_csv(metadata_path)
test_df = df[df['fold'] == 10]
test_ds = UrbanSoundDataset(test_df)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# ==== Evaluate ====
correct = 0
total = 0
predictions = []
with torch.no_grad():
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        _, (h_n, _) = model(X)
        logits = classifier(h_n[-1])
        preds = logits.argmax(dim=1)
        correct += (preds == Y).sum().item()
        total += Y.size(0)
        predictions.extend(zip(preds.cpu().numpy(), Y.cpu().numpy()))

test_acc = correct / total
print(f"Test accuracy on fold 10: {test_acc:.4f}")

# ==== Print some predictions ====
print("\nSample predictions (predicted_label, true_label):")
for i in range(10):
    print(predictions[i])

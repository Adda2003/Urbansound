# download_urbansound.py
"""
Locate and display a random sample from a locally unpacked UrbanSound8K dataset.
"""
import os
import pandas as pd
import random

# === Configuration ===
base_dir = os.path.dirname(os.path.abspath(__file__))
cache_root = os.path.join(base_dir, "cache", "urbansound8k")

# Locate metadata CSV anywhere under cache_root
metadata_csv = None
for root, _, files in os.walk(cache_root):
    if "UrbanSound8K.csv" in files:
        metadata_csv = os.path.join(root, "UrbanSound8K.csv")
        break
if not metadata_csv:
    raise FileNotFoundError(f"Could not find metadata CSV under {cache_root}")

# Load metadata
df = pd.read_csv(metadata_csv)
required_cols = {"slice_file_name", "fold"}
if not required_cols.issubset(df.columns):
    raise KeyError(f"Metadata CSV missing columns: {required_cols - set(df.columns)}")

# Pick a random example
row = df.sample(1, random_state=42).iloc[0]
fname = row["slice_file_name"]
fold  = row["fold"]

# Search for the audio file under any 'foldX' directory
audio_path = None
for root, dirs, _ in os.walk(cache_root):
    if f"fold{fold}" in dirs:
        candidate = os.path.join(root, f"fold{fold}", fname)
        if os.path.isfile(candidate):
            audio_path = candidate
            break
if not audio_path:
    raise FileNotFoundError(f"Audio file {fname} not found under any fold{fold} directory in {cache_root}")

# Print sample info
print("Metadata CSV used:", metadata_csv)
print("Audio root base :", cache_root)
print("\nSample clip info:")
print(f"  audio_path      : {audio_path}")
print(f"  slice_file_name : {fname}")
# Optional fields
for key in ["classID", "class", "start", "end", "salience"]:
    if key in row.index:
        print(f"  {key:<16}: {row[key]}")

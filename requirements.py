import importlib
import subprocess
import sys
import os
import time
import zipfile
import urllib.error
import soundata

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# List only external packages (std-lib modules like os stay out)
for _pkg in ("pandas", "librosa", "numpy", "torch"):
    try:
        print(f"Checking package '{_pkg}'...")
        importlib.import_module(_pkg)
        print(f"Package '{_pkg}' is already installed.")
    except ImportError:
        print(f"Package '{_pkg}' not found. Installing…")
        _install(_pkg)

# Now safe to import everything
import os
import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# …rest of your code…
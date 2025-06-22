# UrbanSound8K LSTM Training Pipeline
# 1. Dataset & Problem Overview
Goal: Classify short (≤4s) audio clips from the UrbanSound8K dataset into 10 urban sound categories (e.g., siren, dog bark, drilling).

Dataset structure:

Each clip belongs to one of 10 classes.

Metadata CSV contains info about each clip, including filename, fold, label, etc.

Audio files are organized as fold1 to fold10 folders, each containing .wav files.

# 2. Feature Extraction
MFCCs (Mel-frequency cepstral coefficients):

For each clip, we extract 13 MFCC features per time step, capturing the timbral texture of sound.

MFCC extraction uses a window of 25ms and a hop of 10ms, so each audio clip is turned into a sequence of feature vectors.

All feature sequences are padded or truncated to a maximum length of 400 frames (covers the max clip length).

# 3. Labels
The target label is an integer classID from 0–9, each mapping to a class:

ini
Copy
Edit
0 = air_conditioner
1 = car_horn
2 = children_playing
3 = dog_bark
4 = drilling
5 = engine_idling
6 = gun_shot
7 = jackhammer
8 = siren
9 = street_music

# 4. Train/Validation Split
Training set: folds 1–8 (majority of data)

Validation set: fold 9 (to measure generalization and tune model)

# 5. Neural Network Model
Architecture:

A two-layer LSTM (Long Short-Term Memory) recurrent neural network reads the MFCC sequence, summarizing the audio’s temporal information.

The last hidden state of the LSTM is fed into a linear (fully connected) layer to classify the sequence into one of the 10 sound classes.

Why LSTM?

LSTMs can model temporal dependencies in the sound, making them effective for sequence data like audio.

# 6. Hyperparameters
Sample rate: 22,050 Hz

Number of MFCCs: 13

LSTM hidden size: 128

LSTM layers: 2

Batch size: 16

Epochs: 5

Learning rate: 0.001 (1e-3)

Optimizer: AdamW (Adam with weight decay for regularization)

Loss function: Cross-entropy loss (standard for multi-class classification)

# 7. Loss Function
Cross-entropy loss:

Measures the difference between the model’s predicted class probabilities and the true class labels.

The optimizer updates model weights to minimize this loss, improving classification accuracy.

# 8. Training & Validation Loop
For each epoch:

The model processes batches of training data, updating weights via backpropagation.

After each epoch, validation accuracy is computed on the held-out validation set.

Logs show per-epoch training loss and validation accuracy, helping to monitor learning progress.

# 9. Saving the Model
After training, the LSTM and classifier weights are saved to urbansound_lstm.pth for later use or further evaluation.

Summary
This pipeline uses MFCCs as input features, an LSTM to capture temporal dynamics, and a linear classifier to predict the sound category. Hyperparameters like learning rate, batch size, and feature extraction settings were chosen to balance model capacity and training efficiency. Validation accuracy provides feedback on model generalization, and the cross-entropy loss guides the learning process.


import torch
import torch.nn as nn

# 1) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Hyperparameters
input_size   = 13    # MFCC feature dim
hidden_size  = 128
num_layers   = 2
num_classes  = 10
batch_size   = 8
seq_length   = 100   # placeholder time-steps
learning_rate= 1e-3

# 3) Model definition (LSTM + classifier) inline
lstm      = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
classifier= nn.Linear(hidden_size, num_classes).to(device)

# 4) Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(list(lstm.parameters()) + list(classifier.parameters()),
                              lr=learning_rate)

# 5) Dummy forward pass to verify shapes
#    
dummy_input = torch.randn(batch_size, seq_length, input_size).to(device)  # (B, T, 13)
lstm_out, (h_n, c_n) = lstm(dummy_input)
#   lstm_out: (B, T, hidden_size)
#   h_n: (num_layers, B, hidden_size) → take last layer's hidden state:
final_hidden = h_n[-1]  # (B, hidden_size)
logits = classifier(final_hidden)  # (B, num_classes)
print("Logits shape (batch×classes):", logits.shape)

# 6) Single backward pass to verify loss flow
dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
loss = criterion(logits, dummy_labels)
print("Initial loss:", loss.item())
loss.backward()
print("Backward pass successful, gradients computed.")

# 7) (Next) Hook this model into your full training loop or Trainer.

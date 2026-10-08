import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import Net

torch.manual_seed(42)

data = load_breast_cancer()

X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

X_tr = torch.tensors(X_tr, dtype=torch.float32)
X_te = torch.tensors(X_te, dtype=torch.float32)
y_tr = torch.tensors(y_tr, dtype=torch.float32).view(-1,1)
y_te = torch.tensors(y_te, dtype=torch.float32).view(-1,1)

train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

model = Net(X_tr.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(60):
    model.train()
    running = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:3d}, Loss: {running/ len(train_dl):.4f}")

    
model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(X_te)) > 0.5).float()
    acc = (preds == y_te).float().mean().item()
print(f"\n Test Accuracy: {acc:.2%}")

torch.save(model.state_dict(), "model.pth")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("meta.pkl", "wb") as f:
    pickle.dump({
        "feature_names":
        list(data.feature_names),
         "target_names":
        list(data.target_names),
        "accuracy": acc,
        "medians": np.median(data.data, axis= 0).tolist(),
        "mins": data.data.min(axis=0).tolist(),
        "maxs": data.data.max(axis=0).tolist(),
    }, f, )

print("Saved model.pth, scaler.pkl and meta.pkl")

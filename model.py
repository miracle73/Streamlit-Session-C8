import torch.nn as nn

class Net(nn.Module):
    def _init_(self, n_features):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        return self.net(x)
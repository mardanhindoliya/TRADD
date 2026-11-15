import torch
import torch.nn as nn

class TemporalTRADD(nn.Module):
    def __init__(self, input_size=704, hidden_size=128):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])

"""
Simple phoneme classifier for ECoG data.
Takes x_common from frozen pretrained encoder → outputs 1 phoneme class per sample.
"""

import torch.nn as nn


class SimplePhonemeClassifier(nn.Module):
    """
    Takes x_common [batch, channels, time] → outputs [batch, num_classes]
    One classification per full time series.
    """
    def __init__(self, input_channels=32, num_classes=40, hidden_dim=128, dropout=0.3):
        super(SimplePhonemeClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),           # [B, C, T] → [B, C, 1] (pool over time)
            nn.Flatten(),                       # → [B, C]
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_common):
        """
        Args:
            x_common: [batch, channels, time]
        Returns:
            logits: [batch, num_classes]
        """
        return self.classifier(x_common)

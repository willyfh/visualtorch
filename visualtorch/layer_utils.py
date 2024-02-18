import torch.nn as nn


class SpacingDummyLayer(nn.Module):
    def __init__(self, spacing: int = 50):
        super().__init__()
        self.spacing = spacing

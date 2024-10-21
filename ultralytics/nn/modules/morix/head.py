import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .ops import *
from .wrapper import *

class ClassificationHead(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2):
        """Initializes YOLOv8 classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c1, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(x).flatten(1)))
        return x if self.training else x.softmax(1)


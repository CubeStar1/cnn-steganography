import torch
from torch import nn
from .base import BasicEncoder

class ResidualEncoder(BasicEncoder):
    """
    The ResidualEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image.
    """
    add_image = True

    def _build_models(self, device):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)

        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
        ).to(device)

        return self.features, self.layers 
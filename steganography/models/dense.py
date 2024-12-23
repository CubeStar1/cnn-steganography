import torch
from torch import nn
from .base import BasicEncoder

class DenseEncoder(BasicEncoder):
    """
    The DenseEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image using dense connections.
    """
    add_image = True

    def _build_models(self, device):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)

        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)

        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)

        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3),
            nn.Tanh(),
        ).to(device)

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image, data):
        image = image.to(self.device)
        data = data.to(self.device)

        batch_size, data_depth = data.size()
        height, width = image.size(2), image.size(3)
        data = data.view(batch_size, data_depth, 1, 1).expand(batch_size, data_depth, height, width)

        x1 = self.conv1(image)
        x2_input = torch.cat([x1, data], dim=1)
        x2 = self.conv2(x2_input)

        x3_input = torch.cat([x1, x2, data], dim=1)
        x3 = self.conv3(x3_input)

        x4_input = torch.cat([x1, x2, x3, data], dim=1)
        x4 = self.conv4(x4_input)

        if self.add_image:
            x4 = image + x4

        return x4 
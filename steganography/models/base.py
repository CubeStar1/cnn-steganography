import torch
from torch import nn

class BasicEncoder(nn.Module):
    def __init__(self, data_depth, hidden_size, device):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.device = device
        self._models = self._build_models(device)

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

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
            nn.Tanh(),
        ).to(device)

        return self.features, self.layers

    def forward(self, image, data):
        image = image.to(self.device)
        data = data.to(self.device)

        batch_size, data_depth = data.size()
        height, width = image.size(2), image.size(3)
        data = data.view(batch_size, data_depth, 1, 1).expand(batch_size, data_depth, height, width)

        x = self.features(image)
        x = torch.cat([x, data], dim=1)
        x = self.layers(x)

        if self.add_image:
            x = image + x

        return x 
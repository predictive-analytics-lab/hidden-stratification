from __future__ import annotations
from collections import OrderedDict

from torch import Tensor
import torch.nn as nn

__all__ = ["Mp64x64Net"]


class Mp64x64Net(nn.Module):
    def __init__(self, num_classes: int, batch_norm: bool = True):
        super().__init__()
        self.batch_norm = batch_norm

        layers = OrderedDict()
        layers["conv1"] = self._conv_block(3, 64, 5, 1, 0)
        layers["mp1"] = nn.MaxPool2d(2, 2)

        layers["conv2"] = self._conv_block(64, 128, 3, 1, 1)
        layers["mp2"] = nn.MaxPool2d(2, 2)

        layers["conv3"] = self._conv_block(128, 128, 3, 1, 1)
        layers["mp3"] = nn.MaxPool2d(2, 2)

        layers["conv4"] = self._conv_block(128, 256, 3, 1, 1)
        layers["mp4"] = nn.MaxPool2d(2, 2)

        layers["conv5"] = self._conv_block(256, 512, 3, 1, 1)
        layers["mp5"] = nn.MaxPool2d(2, 2)

        layers["flatten"] = nn.Flatten
        self.convnet = nn.Sequential(layers)
        self.activation_layer_name = "convnet.mp5"
        self.fc = nn.Linear(512, num_classes)

    def _conv_block(
        self, in_dim: int, out_dim: int, kernel_size: int, stride: int, padding: int
    ) -> nn.Sequential:
        _block: list[nn.Module] = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if self.batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return nn.Sequential(*_block)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.convnet(x))

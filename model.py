import torch
import torch.nn as nn


class ConvLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, pool=True):
    super().__init__()
    padding = kernel_size[0] // 2 if kernel_size[0] % 2 == 1 else 0
    self.net = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if pool else nn.Identity()
    )

  def forward(self, x):
    return self.net(x)


class DNNLayer(nn.Module):
  def __init__(self, in_features, out_features, output_layer=False):
    super().__init__()
    self.net = nn.Sequential(
      nn.BatchNorm1d(num_features=in_features),
      nn.Dropout(p=0.2),
      nn.Linear(in_features, out_features),
      nn.Sigmoid() if output_layer else nn.ReLU()
    )

  def forward(self, x):
    return self.net(x)


class CNN_DNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn = nn.Sequential(
      ConvLayer(in_channels=1, out_channels=16, kernel_size=(2, 2), stride=(1, 1)),
      ConvLayer(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
      ConvLayer(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(1, 1), pool=False),
      ConvLayer(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(1, 1), pool=False),
      ConvLayer(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(1, 1), pool=False),
    )
    self.dnn = nn.Sequential(
      DNNLayer(3264, 1024),
      DNNLayer(1024, 512),
      DNNLayer(512, 256),
      DNNLayer(256, 161*2, output_layer=True)
    )

  def forward(self, x):
    cnn_out = self.cnn(x)
    cnn_out = torch.flatten(cnn_out, start_dim=1)
    dnn_out = self.dnn(cnn_out)
    (real, imag) = torch.chunk(dnn_out, chunks=2, dim=1)
    return real, imag

import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder and Decoder
# class Encoder(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(nn.Linear(input_size, 1024), nn.PReLU(),
#                                      nn.Linear(1024, 512), nn.PReLU(),
#                                      nn.Linear(512, 256), nn.PReLU(),
#                                      nn.Linear(256, 128), nn.PReLU(),
#                                      nn.Linear(128, output_size), nn.Sigmoid())
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(nn.Linear(input_size, 128), nn.PReLU(),
#                                      nn.Linear(128, 256), nn.PReLU(),
#                                      nn.Linear(256, 512), nn.PReLU(),
#                                      nn.Linear(512, 1024), nn.PReLU(),
#                                      nn.Linear(1024, output_size))
#
#     def forward(self, x):
#         x = self.decoder(x)
#         return x


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 2048), nn.PReLU(),
                                     nn.Linear(2048, 1024), nn.PReLU(),
                                     nn.Linear(1024, 512), nn.PReLU(),
                                     nn.Linear(512, 256), nn.PReLU(),
                                     nn.Linear(256, 128), nn.PReLU(),
                                     nn.Linear(128, output_size), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(input_size, 128), nn.PReLU(),
                                     nn.Linear(128, 256), nn.PReLU(),
                                     nn.Linear(256, 512), nn.PReLU(),
                                     nn.Linear(512, 1024), nn.PReLU(),
                                     nn.Linear(1024, 2048), nn.PReLU(),
                                     nn.Linear(2048, output_size))

    def forward(self, x):
        x = self.decoder(x)
        return x

class Encoder_S2D(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder_S2D, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 512), nn.PReLU(),
                                     nn.Linear(512, 256), nn.PReLU(),
                                     nn.Linear(256, 128), nn.PReLU(),
                                     nn.Linear(128, output_size))

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder_S2D(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder_S2D, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(input_size, 128), nn.PReLU(),
                                     nn.Linear(128, 256), nn.PReLU(),
                                     nn.Linear(256, 512), nn.PReLU(),
                                     nn.Linear(512, output_size))

    def forward(self, x):
        x = self.decoder(x)
        return x


class PtNet(nn.Module):
    def __init__(self, dim):
        super(PtNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 28)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(128)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)

        return x
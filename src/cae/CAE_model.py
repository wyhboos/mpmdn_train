import torch
import torch.nn as nn


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



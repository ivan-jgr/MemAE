import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            self.upsampling_block(256, 256),
            self.upsampling_block(256, 128),
            self.upsampling_block(128, 64),
            self.upsampling_block(64, 32),
            self.upsampling_block(32, 16),
            self.upsampling_block(16, 1, last=True)
        )

    def upsampling_block(self, inc, outc, last=False):
        if last:
            return nn.ConvTranspose2d(inc, outc, 4, 2, 1, bias=False)

        return nn.Sequential(
            nn.ConvTranspose2d(inc, outc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.decoder(x)


if __name__ == '__main__':
    import torch
    x = torch.rand(1, 64, 3, 3)
    model = Decoder()
    out = model(x)
    print(out.shape)
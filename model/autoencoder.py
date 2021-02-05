import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            self.conv_block(1, 16),
            self.conv_block(16, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 256)
        )

    def conv_block(self, inc, outc):
        return nn.Sequential(
            nn.Conv2d(inc, outc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.encoder(x)
        return out.view(-1, 1).squeeze(1)


if __name__ == '__main__':
    import torch

    img = torch.rand(1, 1, 224, 224)
    model = Encoder()
    out = model(img)
    print(out.shape)

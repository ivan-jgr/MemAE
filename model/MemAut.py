import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from torch.nn import functional as F


class MemAut(nn.Module):
    def __init__(self):
        super(MemAut, self).__init__()
        # Encoder, Decoder
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.num_memories = 100
        self.feature_size = 64 * 3 * 3
        init_mem = torch.zeros(self.num_memories, self.feature_size)
        nn.init.kaiming_normal_(init_mem)

        self.memory = nn.Parameter(init_mem)
        self.cosine_similarity = nn.CosineSimilarity(dim=2, )
        self.relu = nn.ReLU(inplace=True)
        self.threshold = 1 / self.memory.size(0)
        self.epsilon = 1e-15

    def forward(self, x):
        b, c, h, w = x.size()

        z = self.encoder(x).view(b, -1)
        # copias de la memoria, una por cada ejemplo
        ex_mem = self.memory.repeat(b, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)

        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)
        d = mem_weight - self.threshold
        mem_weight = (self.relu(d) * mem_weight) / (torch.abs(d) + self.epsilon)
        mem_weight = mem_weight / mem_weight.norm(p=1, dim=1).unsqueeze(1).expand(b, self.num_memories)
        z_hat = torch.mm(mem_weight, self.memory)

        rec_x = self.decoder(z_hat.view(b, 64, 3, 3))
        return rec_x

def test():
    import matplotlib.pyplot as plt
    import torchvision
    from torchvision import datasets

    model = MemAut()
    model.load_state_dict(torch.load('../checkpoints.pth'))

    transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    dataset = datasets.MNIST('../files/', train=True, download=False, transform=transform)

    idx = dataset.targets == 7
    train_data = dataset.data[idx]

    for i in range(10):
        x = transform(train_data[i].view(1, 28, 28).float()).view(1, 1, 28, 28)
        x = F.interpolate(x, (24, 24))
        out = model(x)

        plt.subplot(1, 3, 1)
        plt.imshow(x[0, 0, ...], cmap='gray_r')
        plt.subplot(1, 3, 2)
        plt.imshow(out[0, 0, ...].detach().numpy(), cmap='gray_r')
        plt.subplot(1, 3, 3)
        e = x - out
        plt.imshow( e[0, 0, ...].detach().numpy(), cmap='jet' )
        plt.colorbar()
        plt.show()

test()

if __name__ == '__mains__':
    import matplotlib.pyplot as plt
    import torchvision
    from torchvision import datasets
    import torch.optim as optim

    model = MemAut()
    transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    dataset = datasets.MNIST('../files/', train=True, download=False, transform=transform)

    idx = dataset.targets == 1
    train_data = dataset.data[idx]

    n = train_data.size(0)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for i in range(10):
        print("--"*20)
        print("EPOCH: ", i)
        print("--" * 20)
        for sample in range(n):
            optimizer.zero_grad()
            x = transform(train_data[sample].view(1, 28, 28).float()).view(1, 1, 28, 28)
            x = F.interpolate(x, (24, 24))
            out = model(x)
            loss = criterion(x, out)
            loss.backward()
            optimizer.step()

            if sample % 100 == 0:
                print("loss: ", loss)

        plt.imshow(out[0, 0, ...].detach().numpy(), cmap='gray')
        plt.show()




    """
    x = torch.rand(2, 1, 256, 256)
    model = MemAut()
    out = model(x)

    plt.imshow(x[0, 0, ...])
    plt.show()
    print(out.shape)
    print(model.memory[0].shape)
    plt.imshow(model.memory[0].view(64, -1).detach().numpy())
    #plt.imshow(out[0, 0, ...].detach().numpy())
    plt.show()
    """
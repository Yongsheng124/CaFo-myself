import torch
import torch.nn as nn
from torch.optim import Adam
# from utils import one_hot, loss_MSE, jaccobian_MSE, loss_cross_entropy, jaccobian_cross_entropy, loss_sparsemax, \
#     jaccobian_sparsemax
from tqdm import tqdm


class Block(nn.Module):
    def __init__(self, loss_fn, num_classes, num_epochs, num_batches, conv_module, out_features, step, lamda):
        super().__init__()
        self.loss_fn = loss_fn
        self.conv_module = conv_module
        self.out_features = out_features
        self.conv_module.to('cpu')
        self.num_classes = num_classes
        self.lamda = lamda
        assert loss_fn in ['BP', 'CE', 'SL', 'MSE']
        if loss_fn in ['BP', 'CE', 'SL']:
            self.num_epochs = num_epochs
            self.num_batches = num_batches
            self.step = step
            if loss_fn == 'BP':
                self.fc = nn.Linear(in_features=self.out_features, out_features=num_classes, bias=False)
                self.fc.to('cuda')
                self.opt = Adam(self.fc.parameters(), lr=step, eps=1e-4)  # lr=0.001

    def forward(self, x):
        x = self.conv_module(x)
        return x

    def batch_forward(self, x, num_batches):
        input = x[:1, :, :, :]
        output = self.forward(input)

        fea = torch.zeros(size=(x.shape[0], output.shape[1], output.shape[2], output.shape[3]), device=x.device)
        batch_size = int(x.shape[0] / num_batches)
        for i in range(num_batches):
            batch_x = x[i * batch_size:(i + 1) * batch_size, :, :, :]
            batch_y = self.forward(batch_x).detach()
            fea[i * batch_size:(i + 1) * batch_size, :, :, :] = batch_y

        return fea

    def linear(self, x):
        return torch.matmul(x, self.W)


if __name__ == '__main__':
    layers = []
    inp_channels, out_channels = 3, 32
    wp = 32
    n_blocks = 3
    assert 0 < n_blocks < 4
    for i in range(n_blocks):
        out_channels = 32 * pow(4, i)

        conv = torch.nn.Sequential(
            torch.nn.Conv2d(inp_channels, out_channels, 3, 1, 1),
            torch.nn.ReLU(True),
            # torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(out_channels, eps=1e-4)
        )
        wc = int((wp - 3 + 2 * 1) / 1 + 1)
        wp = int((wc - 2 + 2 * 0) / 2 + 1)
        out_features = int(out_channels * wp * wp)
        inp_channels = out_channels
        step = 0.01
        lamda = 0.0
        loss_fn = "MSE"
        num_classes = 10
        num_epochs = 5000
        num_batches = 32
        layers += [Block(loss_fn, num_classes, num_epochs, num_batches, conv, out_features, step, lamda)]
    x = torch.randn(size=(50000, 3, 32, 32))
    print(len(layers))
    for block in layers:
        x = block.batch_forward(x, num_batches)
        print("this", x.shape)

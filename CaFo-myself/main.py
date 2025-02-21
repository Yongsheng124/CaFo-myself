import argparse
from loader import load_data_cafo
import torch.cuda

from utils import evaluate, set_seed

parser = argparse.ArgumentParser(description='Cascaded Forward Algorithm')
parser.add_argument('--data', type=str, default='CIFAR10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU id')
parser.add_argument('--num_epochs', type=int, default=5000,
                    help='number of epochs for GD (default: 10000)')
parser.add_argument('--step', type=float, default=0.01,
                    help='optimization step for GD (default: 0.001)/learning rate for BP')
parser.add_argument('--loss_fn', type=str, default="MSE",
                    help='MSE(mean square error)/CE(cross entropy)/SL(sparsemax loss)/BP(backpropagation)')
parser.add_argument('--num_batches', type=int, default=1,
                    help='number of batches for BP, CE and SL')
parser.add_argument('--lamda', type=float, default=0.0,
                    help='regularization factor')
parser.add_argument('--num_blocks', type=int, default=3,
                    help='number of blocks')
args = parser.parse_args()

set_seed(args.seed)

torch.cuda.set_device(args.gpu)
train_loader, test_loader, num_classes, input_channels, input_size = load_data_cafo(args)
x, y = next(iter(train_loader))
# x : 50000 * 3* 32 *32
# y : 50000
x_test, y_test = next(iter(test_loader))
# x : 10000 * 3* 32 *32
# y : 10000
layers = []
input_channels, output_channels = input_channels, 32
wp = input_size
print("WP=", wp)
n_blocks = 3
assert 0 < n_blocks < 4
for i in range(n_blocks):
    output_channels = 32 * pow(4, i)

    conv = torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, output_channels, 3, 1, 1),
        torch.nn.ReLU(True),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.BatchNorm2d(output_channels, eps=1e-4)
    )

    wc = int((wp - 3 + 2 * 1) / 1 + 1)
    wp = int((wc - 2 + 2 * 0) / 2 + 1)
    print("wc=", wc)
    print("WP=", wp)
    out_features = int(output_channels * wp * wp)
    print(out_features)
    inp_channels = output_channels
    layers += [Block(loss_fn, num_classes, num_epochs, num_batches, conv, out_features, step, lamda)]

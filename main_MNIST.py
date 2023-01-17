import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

# 3.3 A simple supervised example of FF

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        # change image to tensor;
        ToTensor(),
        # Normalize the image by subtracting a known mean and standard deviation;
        Normalize((0.1307,), (0.3081,)),
        # flatten the image;
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


# For x: 0_th dimension：Sample Size; 1_st dimension：Flattened Features
# y is a scaler: 0 <= y <= 9
# Cover each row of x (each sample in x) with the same label,
# by writing x.max() into the position which corresponds to the label value.
def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0   # REPLACE the first 10 pixels by the label representation.
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

    def predict(self, x):
        goodness_per_label = []
        # Iterate over all the 10 labels and find the one with highest accumulated goodness.
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        # return the index of the label with maximum accumulated goodness for each sample,
        # as the final prediction for all testing samples.
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


# All the hyper-parameters are determined by Layer.
# The concrete implementation of Forward and Train are also implemented in Layer.
class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = AdamW(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        # normalize the input x (Chapter 2.1 in the original paper)
        # Why an additional 1e-4 ?
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))
        # Unsqueeze here: expand a new dimension of size 1; then apply broadcasting mechanism in the addition.

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            # positive forward pass for each layer.
            g_pos = self.forward(x_pos).pow(2).mean(1)
            # negative forward pass for each layer
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()   # calculate the mean of all elements in the resulting tensor
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            # Update the parameter layer by layer respectively.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

if __name__ == "__main__":

    torch.manual_seed(3241)
    train_loader, test_loader = MNIST_loaders()

    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    # Generate positive data and negative data
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())

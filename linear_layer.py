import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, AdamW


# 最简单的线性层: 无 weight_decay; 也无 dropout; 其余所有参数都为默认值
class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 lr, threshold, num_epochs,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=lr)
        self.threshold = threshold
        self.num_epochs = num_epochs

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


# 带有 weight_decay 的线性层 ( AdamW 实现, 默认值为 0.01 )
class CIFAR_Layer(Layer):
    # Override INIT method in Layer in order to implement weight_decay
    def __init__(self, in_features, out_features,
                 lr, threshold, num_epochs, weight_decay,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, lr, threshold, num_epochs, bias, device, dtype)
        self.opt = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
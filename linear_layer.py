import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, AdamW


# The simplest linear layer, without dropout or weight-decay
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
        # Under L-2 based goodness: normalize the input x
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        
        # # Under L-1 based goodness: normalize the input x
        # x_direction = x / (x.norm(1, 1, keepdim=True) + 1e-4)
        
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))
        # Unsqueeze here: expand a new dimension of size 1; then apply broadcasting mechanism in the addition.

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            # The following goodness is L-2 based:
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            
            # # The following goodness is L-1 based:
            # g_pos = self.forward(x_pos).abs().mean(1)
            # g_neg = self.forward(x_neg).abs().mean(1)
            
            # The following loss pushes pos (neg) samples to values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()   # calculate the mean of both positive and negative samples
            
            # # The following loss pushes pos (neg) samples to values smaller (larger) than the self.threshold.
            # loss = torch.log(1 + torch.exp(torch.cat([
            #     g_pos - self.threshold,
            #     -g_neg + self.threshold]))).mean() 
            
            self.opt.zero_grad()
            # This backward just compute the derivative and hence is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


# Linear layer with weight-decay, default to 0.01
class CIFAR_Layer(Layer):
    # Override INIT method in Layer in order to implement weight_decay
    def __init__(self, in_features, out_features,
                 lr, threshold, num_epochs, weight_decay,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, lr, threshold, num_epochs, bias, device, dtype)
        self.opt = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
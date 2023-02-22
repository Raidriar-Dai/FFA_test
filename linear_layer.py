import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

class Layer(nn.Linear):
    '''最简单的线性层, 无 weight decay 或 dropout, 一般用于 MNIST 实验'''
    def __init__(self, in_features, out_features, lr, threshold,
                bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=lr)
        self.threshold = threshold

    def forward(self, x):
        # Under L-2 based goodness: normalize the input x
        # 加上 1e-4 的原因: 防止除 0 导致数值爆炸.
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)

        # # Under L-1 based goodness: normalize the input x
        # x_direction = x / (x.norm(1, 1, keepdim=True) + 1e-4)

        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))
        # Unsqueeze here: expand a new dimension of size 1; then apply broadcasting mechanism in the addition.

    def train(self, x_pos, x_neg):
        '''layer.train() 函数只会对一组给定的输入(即单独的一个 batch), 做1次更新'''
        # The following goodness is L-2 based:
        g_pos = self.forward(x_pos.detach()).pow(2).mean(1)
        g_neg = self.forward(x_neg.detach()).pow(2).mean(1)

        # # The following goodness is L-1 based:
        # g_pos = self.forward(x_pos.detach()).abs().mean(1)
        # g_neg = self.forward(x_neg.detach()).abs().mean(1)

        # The following loss pushes pos (neg) samples to values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()   # calculate the mean of both positive and negative samples

        # # The following loss pushes pos (neg) samples to values smaller (larger) than the self.threshold.
        # loss = torch.log(1 + torch.exp(torch.cat([
        #     g_pos - self.threshold,
        #     -g_neg + self.threshold]))).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class CIFAR_Layer(Layer):
    '''用 AdamW 来实现 weight decay 的线性层'''
    def __init__(self, in_features, out_features,
                lr, threshold, weight_decay,
                bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, lr, threshold, bias, device, dtype)
        self.opt = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)


class CIFAR_LocalReceptiveLayer(CIFAR_Layer):
    '''同时实现 weight decay 和 local receptive field 的线性层'''
    def __init__(self, in_features, out_features,
                lr, threshold, weight_decay, mask,
                bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, lr, threshold, weight_decay, bias, device, dtype)
        self.mask = mask

    # 在每次参数更新之后, 给参数矩阵套上蒙板(mask), 再用修正后的参数矩阵进行下一次训练
    def train(self, x_pos, x_neg):
        # with torch.enable_grad():

        # The following goodness is L-2 based:
        g_pos = self.forward(x_pos.detach()).pow(2).mean(1)
        g_neg = self.forward(x_neg.detach()).pow(2).mean(1)

        # # The following goodness is L-1 based:
        # g_pos = self.forward(x_pos.detach()).abs().mean(1)
        # g_neg = self.forward(x_neg.detach()).abs().mean(1)

        # The following loss pushes pos (neg) samples to values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()   # calculate the mean of both positive and negative samples

        # # The following loss pushes pos (neg) samples to values smaller (larger) than the self.threshold.
        # loss = torch.log(1 + torch.exp(torch.cat([
        #     g_pos - self.threshold,
        #     -g_neg + self.threshold]))).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            self.weight.masked_fill_(self.mask, 0)

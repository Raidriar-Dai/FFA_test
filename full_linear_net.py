import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from linear_layer import Layer, CIFAR_Layer
from data_overlay import overlay_y_on_x
from data_load import Labeled_Dataset
from tqdm import tqdm

# Fully-connected Linear net, without dropout layer.
class Net(torch.nn.Module):

    # 新增了 Net 的参数: batch_size, 用来控制 pos_loader 与 neg_loader 的批次大小
    def __init__(self, dims, lr, threshold, num_epochs, batch_size):
        super().__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], lr, threshold).cuda()]

    def predict(self, x):
        goodness_per_label = []
        # Iterate over all the 10 labels and find the one with highest accumulated goodness.
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                if isinstance(layer, nn.Dropout):
                    continue
                else:
                    h = layer(h)
                    goodness += [h.pow(2).mean(1)]  # squared_mean
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        # For each testing sample, return the index of the label with maximum / minimum accumulated goodness
        # as the final prediction.
        return goodness_per_label.argmax(1)     # MAXIMUM accumulated goodness
        # return goodness_per_label.argmin(1)     # MINIMUM accumulated goodness

    # 在 Net 的 training 函数中, 实现 layer -> epoch -> batch 的结构,
    # 而非让每层 layer 在给定的一个批次输入 x_pos 和 x_neg 之上跑很多个 epochs
    def forward_train(self, x_pos_all, x_neg_all):
        # 初始化训练第一层所需的 pos_loader 与 neg_loader
        pos_loader = DataLoader(Labeled_Dataset(x_pos_all), batch_size=self.batch_size, shuffle=True)
        neg_loader = DataLoader(Labeled_Dataset(x_neg_all), batch_size=self.batch_size, shuffle=True)
        # 再对每层 layer 一层层地训练下来
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Dropout):
                print('training layer', i, ': Dropout Layer ...')
            else:
                print('training layer', i, '...')
                for epoch in tqdm(range(self.num_epochs)):
                    # 每一份 x_pos 与 x_neg 都是一个 batch 的数据
                    for x_pos, x_neg in zip(pos_loader, neg_loader):
                        x_pos, x_neg = x_pos.cuda(), x_neg.cuda()
                        layer.train(x_pos, x_neg)
            # 该 layer 训练完毕后, 更新训练下一 layer 所要使用的 pos_loader 与 neg_loader
            # 注意: 为了消除传到下一层 layer 的 tensor 在前一层上 forward 的计算图, 需要用 Tensor.detach()
            # x_pos_all, x_neg_all = layer(x_pos_all).detach(), layer(x_neg_all).detach()
            x_pos_all, x_neg_all = ([layer.forward(x_pos).detach() for x_pos in pos_loader],
                                    [layer.forward(x_neg).detach() for x_neg in neg_loader])
            x_pos_all, x_neg_all = torch.cat(x_pos_all, 0), torch.cat(x_neg_all, 0)
            pos_loader = DataLoader(Labeled_Dataset(x_pos_all), batch_size=self.batch_size, shuffle=True)
            neg_loader = DataLoader(Labeled_Dataset(x_neg_all), batch_size=self.batch_size, shuffle=True)


# Fully-connected Linear layer, with dropout layer and weight-decay.
class CIFAR_Net(Net):
    # Override INIT method of Net in order to implement <Dropout> and <local_receptive_field>
    def __init__(self, dims, lr, threshold, num_epochs, batch_size, weight_decay, dropout):
        super(Net, self).__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.layers = []
        self.mask = generate_mask().cuda()  # 整个 CIFAR_Net 的所有 layer 共享同一张 mask
        for d in range(len(dims) - 1):
            self.layers += [CIFAR_Layer(dims[d], dims[d + 1], lr, threshold, weight_decay, self.mask).cuda()]
            if (len(dims) == 4 and (d == 0 or d == 1)) or (len(dims) == 3 and d == 0):
                # Apply dropback after the 1st and 2nd layer (when there are 3 layers in total)
                # Or, apply after the 1st layer (when there are 2 layers in total)
                self.layers.append(nn.Dropout(dropout))    # Dropout modif 1


def generate_mask():
    '''生成覆盖在 weight 矩阵上的 3072*3072 大小的蒙板(mask)'''
    mask = torch.ones([3072, 3072], dtype=torch.bool)
    for unit in range(1024):
        # 生成针对 weight 矩阵 第 unit 行的 mask
        i, j = unit // 32, unit % 32    # flattened tensor 中的索引位置 unit <--> 32*32*3 二维图像中的坐标位置 (i,j)
        # 自定义 size 的 local receptive field 的坐标列表
        coord_list = [(i-k1, j-k2) for k1 in range(-10,11) for k2 in range(-10,11)
                        if (i-k1) >= 0 and (i-k1) <= 31 and (j-k2) >= 0 and (j-k2) <= 31]
        # 坐标列表再化回 flattened 之后的索引列表
        index_list = list(map(lambda x: 32 * x[0] + x[1], coord_list))
        index_list = index_list + [index + 1024 for index in index_list] + [index + 2048 for index in index_list]
        # 更新 mask 矩阵上 unit, unit + 1024, unit + 2048 这三行
        for j in index_list:
            mask[unit][j] = False
            mask[unit + 1024][j] = False
            mask[unit + 2048][j] = False
    return mask
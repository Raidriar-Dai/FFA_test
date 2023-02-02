import torch
import torch.nn as nn
from linear_layer import Layer, CIFAR_Layer
from data_overlay import overlay_y_on_x

# Fully-connected Linear net, without dropout layer.
class Net(torch.nn.Module):

    def __init__(self, dims, lr, threshold, num_epochs):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], lr, threshold, num_epochs).cuda()]

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

    def forward_train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Dropout):
                print('training layer', i, ': Dropout Layer ...')
                h_pos, h_neg = layer(h_pos), layer(h_neg)
            else:
                print('training layer', i, '...')
                h_pos, h_neg = layer.train(h_pos, h_neg)


# Fully-connected Linear layer, with dropout layer and weight-decay.
class CIFAR_Net(Net):
    # Override INIT method of Net in order to implement Dropout
    def __init__(self, dims, lr, threshold, num_epochs, weight_decay, dropout):
        super(Net, self).__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [CIFAR_Layer(dims[d], dims[d + 1], lr, threshold, num_epochs, weight_decay).cuda()]
            if len(dims) == 4 and d == 1 or len(dims) == 3 and d == 0:
                # Apply dropback after the second layer (when there are 3 layers in total)
                # Or, apply after the first layer (when there are 2 layers in total)
                self.layers.append(nn.Dropout(dropout))    # Dropout modif 1

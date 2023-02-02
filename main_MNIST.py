import torch
import torch.nn as nn
from data_load import MNIST_loaders
from data_overlay import overlay_y_on_x
from full_linear_net import Net

import wandb
import yaml

# 3.3 A simple supervised example of FF

with open('/home/intern/scratch/qirundai/FFA13/FFA_test/config_MNIST.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
sweep_id = wandb.sweep(sweep=config, project='FFA_test1')

def training_one_run():

    wandb.init(project="FFA_test1", entity="raidriar_dai")

    # define hyper_parameters from wandb.config
    batch_size = wandb.config.batch_size
    dims = wandb.config.dims
    lr = wandb.config.lr
    threshold = wandb.config.threshold
    num_epochs = wandb.config.num_epochs

    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders(train_batch_size=batch_size)

    net = Net(dims, lr, threshold, num_epochs)
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    # Generate positive data and negative data
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    net.forward_train(x_pos, x_neg)

    train_error = 1.0 - net.predict(x).eq(y).float().mean().item()
    print('train error:', train_error)

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    test_error = 1.0 - net.predict(x_te).eq(y_te).float().mean().item()
    print('test error:', test_error)

    wandb.log({"train_error": train_error, 
               "test_error": test_error})
    wandb.finish()

wandb.agent(sweep_id, function=training_one_run, count=1)






import torch
import torch.nn as nn
from data_load import MNIST_loaders
from data_overlay import overlay_y_on_x
from full_linear_net import Net

import wandb
sweep_config = {
    'method': 'random',
    'metric': {'name': 'test_error', 'goal': 'minimize'}
}
parameters_dict = {
    'lr': {
        'values': [0.01, 0.02, 0.03]
    },
    'threshold': {
        'values': [1.0, 2.0, 3.0]
    },
    'num_epochs':{
        'values': [1000, 1500, 2000]
    },
    'batch_size':{
        'value': 50000
    },
    'seed':{
        'value': 1234
    },
    'dims':{
        'value': [784, 500, 500]
    }
}
sweep_config['parameters'] = parameters_dict


# 3.3 A simple supervised example of FF

def training_one_run():
    # lr=0.03, threshold=2.0, num_epochs=1000, batch_size=50000, seed=1234, dims=[784, 500, 500]

    wandb.init(project="FFA_test1", entity="raidriar_dai")
    # config = {"lr": lr, "threshold": threshold, "num_epochs": num_epochs,
    #            "batch_size": batch_size, "seed": seed, "dims": dims})

    # define hyper-parameters from `wandb.config`
    seed = wandb.config.seed
    batch_size = wandb.config.batch_size
    dims = wandb.config.dims
    lr = wandb.config.lr
    threshold = wandb.config.threshold
    num_epochs = wandb.config.num_epochs

    torch.manual_seed(seed)
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

    wandb.log({"train_error": train_error, "test_error": test_error})

sweep_id = wandb.sweep(sweep_config, project="FFA_test1")
wandb.agent(sweep_id, function=training_one_run, count=1)




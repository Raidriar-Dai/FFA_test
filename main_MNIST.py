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
    num_epochs = wandb.config.num_epochs    # 现在的 num_epochs 是每层 layer 的训练总次数

    # 默认 training batch size 为 50000
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders() 
    # 提取出 train_loader 中所有的 samples, 用以生成 x_pos_all 与 x_neg_all
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos_all = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg_all = overlay_y_on_x(x, y[rnd])

    net = Net(dims, lr, threshold, num_epochs, batch_size)

    net.train()
    net.forward_train(x_pos_all, x_neg_all)

    # 对于已经完成所有 layer 训练的网络, 从训练集中取一组小批次数据, 来计算 train error
    net.eval()
    # torch.manual_seed(1234)
    # train_eval_loader, _ = MNIST_loaders(train_batch_size=batch_size)
    # x_eval, y_eval = next(iter(train_eval_loader))
    # x_eval, y_eval = x_eval.cuda(), y_eval.cuda()
    x_eval, y_eval = x[:batch_size], y[:batch_size]
    train_error = 1.0 - net.predict(x_eval).eq(y_eval).float().mean().item()
    print('train error:', train_error)

    # 最后取出所有测试集中的数据, 计算 test error
    net.eval()
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    test_error = 1.0 - net.predict(x_te).eq(y_te).float().mean().item()
    print('test error:', test_error)

    wandb.log({"train error": train_error, 
               "test error": test_error})
    wandb.finish()

wandb.agent(sweep_id, function=training_one_run, count=1)

import numpy as np
import os
import utils

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def train(rank, config, model, data_manager):
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    pid = os.getpid()
    train_data_loader = data_manager.create_dataloader(config)
    test_data_loader = data_manager.create_dataloader(config, mode="test")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        train_epoch(
            epoch=epoch, config=config, model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
        )
        test_result = test_epoch(
            config=config, model=model,
            data_loader=test_data_loader,
        )
        print(
            '[{}] Test Epoch: {}, Average loss: {:.4f}, '
            'Accuracy: {}/{} ({:.0f}%)'.format(
                pid, epoch,
                test_result["loss"], test_result["num_correct"],
                test_result["length"],
                test_result["num_correct"] * 100 / test_result["length"],
            )
        )


def train_epoch(epoch, config, model, data_loader, optimizer):
    model.train()

    loss_func = nn.BCELoss()

    for batch_idx, (x, y) in enumerate(data_loader):
        x_var = utils.maybe_cuda_var(x, cuda=config.cuda)
        y_var = Variable(y, requires_grad=False)
        if config.cuda:
            y_var = y_var.cuda()

        y_hat = model(x_var)
        loss = loss_func(y_hat, y_var)
        loss.backward()
        optimizer.step()

        if config.train_log and batch_idx % config.train_log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0])
            )


def test_epoch(config, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0

    loss_func = nn.BCELoss(size_average=False)
    for x, y in data_loader:
        x_var = utils.maybe_cuda_var(x, cuda=config.cuda)
        y_var = Variable(y, requires_grad=False)
        if config.cuda:
            y_var = y_var.cuda()

        y_hat = model(x_var)
        test_loss += loss_func(y_hat, y_var)
        y_pred = (y_hat.data > 0.5).float()
        correct += y_pred.eq(y_var.data).cpu().sum()

    test_loss /= len(data_loader.dataset)

    return {
        "loss": test_loss.data.cpu().numpy()[0],
        "num_correct": correct,
        "length": len(data_loader.dataset),
    }

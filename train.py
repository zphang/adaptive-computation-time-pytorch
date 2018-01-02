import numpy as np
import utils

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def train(config, model, data_manager):
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
            'Epoch: {}, Average loss: {:.4f}, '
            'Accuracy: {}/{} ({:.0f}%), PT: {}'.format(
                epoch,
                test_result["loss"], test_result["num_correct"],
                test_result["length"],
                test_result["num_correct"] * 100 / test_result["length"],
                f'{test_result["mean_ponder_time"]:.1f}'
                if test_result["mean_ponder_time"] else "N/A",
            )
        )


def train_epoch(epoch, config, model, data_loader, optimizer):
    model.train()

    loss_func = nn.BCEWithLogitsLoss()

    for batch_idx, (x, y) in enumerate(data_loader):
        x_var = utils.maybe_cuda_var(x, cuda=config.cuda)
        y_var = Variable(y, requires_grad=False)
        if config.cuda:
            y_var = y_var.cuda()

        y_hat, ponder_dict = model(x_var)
        loss = loss_func(y_hat, y_var)
        if ponder_dict:
            loss += (
                config.act_ponder_penalty * ponder_dict["ponder_cost"].mean()
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config.train_log and batch_idx % config.train_log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0])
            )


def test_epoch(config, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0

    loss_func = nn.BCEWithLogitsLoss(size_average=False)

    ponder_times_ls = []
    for batch_idx, (x, y) in enumerate(data_loader):
        x_var = utils.maybe_cuda_var(x, cuda=config.cuda)
        y_var = Variable(y, requires_grad=False)
        if config.cuda:
            y_var = y_var.cuda()

        y_hat, ponder_dict = model(x_var, compute_ponder_cost=False)
        # TODO: Add ponder_cost penalty
        test_loss += loss_func(y_hat, y_var).data[0]
        y_pred = (y_hat.data > 0.5).float()
        correct += y_pred.eq(y_var.data).cpu().sum()

        if ponder_dict:
            ponder_times_ls.append(np.array(ponder_dict["ponder_times"]).T)

    test_loss /= len(data_loader.dataset)

    if ponder_times_ls:
        mean_ponder_time = np.mean(np.vstack(ponder_times_ls))
    else:
        mean_ponder_time = None

    return {
        "loss": test_loss,
        "num_correct": correct,
        "length": len(data_loader.dataset),
        "mean_ponder_time": mean_ponder_time,
    }

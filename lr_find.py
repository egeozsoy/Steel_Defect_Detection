import torch
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt

from cycliclr_exp import CyclicExpLR
from model import UNet


def lr_find(model: UNet, data_loader, optimizer: Optimizer, criterion, use_gpu, min_lr=0.0001, max_lr=0.1):
    # Save model and optimizer states to revert
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    losses = []
    lrs = []
    scheduler = CyclicExpLR(optimizer, min_lr, max_lr, step_size_up=100, mode='triangular',cycle_momentum=True)
    model.train()
    for i, (data, target, class_ids) in enumerate(data_loader):
        data, target = data, target
        optimizer.zero_grad()
        output_raw = model(data)
        # This step is specific for this project
        output = torch.zeros(output_raw.shape[0], 1, output_raw.shape[2], output_raw.shape[3])

        if use_gpu:
            output = output.cuda()

        # This step is specific for this project
        for idx, (raw_o, class_id) in enumerate(zip(output_raw, class_ids)):
            output[idx] = raw_o[class_id - 1]

        loss = criterion(output, target)
        loss.backward()
        current_lr = optimizer.param_groups[0]['lr']
        # Stop if lr stopped increasing
        if len(lrs) > 0 and current_lr < lrs[-1]:
            break
        lrs.append(current_lr)
        losses.append(loss.item())
        optimizer.step()
        scheduler.step()

    # Plot in log scale
    plt.plot(lrs, losses)
    plt.xscale('log')

    plt.show()

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

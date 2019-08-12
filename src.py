import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split

from configurations import path, img_size, train
from utils import rle2mask, create_balanced_class_sampler, create_boolean_mask
from dataset import ImageData
from model import UNet
from lr_find import lr_find

# https://www.kaggle.com/c/severstal-steel-defect-detection

if __name__ == '__main__':

    if not os.path.exists('training_process'):
        os.mkdir('training_process')

    use_gpu = torch.cuda.is_available()
    plot_beginning_images = False
    plot_dataloader_examples = False
    use_lr_find = False

    batch_size = 12

    tr = pd.read_csv(path + 'train.csv')
    print(len(tr))

    df_all = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
    df_train, df_valid = train_test_split(df_all, random_state=42, test_size=0.1)

    print(len(df_train), len(df_valid))

    if plot_beginning_images:
        columns = 1
        rows = 4
        fig = plt.figure(figsize=(20, columns * rows + 6))
        for i in range(1, columns * rows + 1):
            fn, class_id_str = df_train['ImageId_ClassId'].iloc[i].rsplit('_', -1)
            class_id = int(class_id_str)
            fig.add_subplot(rows, columns, i).set_title(fn)
            img = cv2.imread(path + 'train_images/' + fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = rle2mask(df_train['EncodedPixels'].iloc[i], (256, 1600))

            # different color codings for different classes
            if class_id == 1:
                img[mask == 1, 0] = 255
            elif class_id == 2:
                img[mask == 1, 1] = 255
            elif class_id == 3:
                img[mask == 1, 2] = 255
            else:
                img[mask == 1, 0:2] = 255

            plt.imshow(img)
        plt.show()

    # Define transformation(if needed augmentation can be applied here)
    data_transf = transforms.Compose(
        [transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(),
         ])

    # Define data
    train_data = ImageData(df=df_train, transform=data_transf, subset='train')
    validation_data = ImageData(df=df_valid, transform=data_transf, subset='valid')

    # Define samplers
    train_sampler = create_balanced_class_sampler(df_train)
    validation_sampler = create_balanced_class_sampler(df_valid)

    # loader uses sampler
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, sampler=validation_sampler, pin_memory=True)

    if plot_dataloader_examples:
        counts = [0, 0, 0, 0]

        values = next(iter(train_loader))
        for i in range(len(values[0])):
            counts[values[2][i].int() - 1] += 1
            plt.imshow(train_data[i][0].permute(1, 2, 0))
            plt.show()
            plt.imshow(np.squeeze(train_data[i][1].permute(1, 2, 0)))
            plt.show()
        print(counts)
    # Create unet model for four classes
    model = UNet(n_class=4)
    if os.path.exists('model.pth'):
        print('Model loaded')
        model.load_state_dict(torch.load('model.pth'))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Paramaters: {total_params}, Trainable Parameters: {trainable_params}')

    if use_gpu:
        print('Using CUDA')
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()  # this is more numerically stable than applying sigmoid and using nn.BCELoss()
    lr = 0.0003  # Enter optimal base_lr found by lr_find
    lr_max = 0.003  # enter optimal max_lr fonud by lr_find
    # optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
    # this only works because pytorch code is changed manually https://github.com/pytorch/pytorch/issues/19003(pytorch 1.2.0 should fix it)
    optimizer = torch.optim.Adam(model.parameters())

    if use_lr_find:
        lr_find(model, train_loader, optimizer, criterion, use_gpu)

    scheduler = CyclicLR(optimizer, lr, lr_max, cycle_momentum=False)

    for epoch in range(25):
        if train:
            # Training loop
            model.train()
            total_training_loss = 0.0
            for i, (data, target, class_ids) in enumerate(train_loader):
                data, target = data, target
                if use_gpu:
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = model.predict(data, use_gpu, class_ids)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()
                scheduler.step()
                total_training_loss += loss.item()

            img = (data[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())
            output_mask = np.abs(create_boolean_mask(output[0][0].cpu().detach().numpy()) * (-1))  # TODO make sure this mask works correctly
            target_mask = target[0][0].cpu().numpy().astype(np.bool)

            img[output_mask == 1, 0] = 1
            img[target_mask == 1, 1] = 1
            # overlapping regions look yellow
            plt.imshow(img)
            plt.savefig(f'training_process/training_{epoch}.png')
            print('Training Epoch: {} - Loss: {:.6f}'.format(epoch + 1, total_training_loss / len(df_train)))
            torch.save(model.state_dict(), 'model.pth')

        # Validation Loop
        model.eval()
        total_validation_loss = 0.0
        for i, (data, target, class_ids) in enumerate(validation_loader):
            data, target = data, target
            if use_gpu:
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model.predict(data, use_gpu, class_ids)

            loss = criterion(output, target)
            total_validation_loss += loss.item()

        img = (data[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())
        output_mask = np.abs(create_boolean_mask(output[0][0].cpu().detach().numpy()) * (-1))
        target_mask = target[0][0].cpu().numpy().astype(np.bool)

        img[output_mask == 1, 0] = 1
        img[target_mask == 1, 1] = 1

        # overlapping regions look yellow
        plt.imshow(img)
        plt.savefig(f'training_process/validation_{epoch}.png')
        print('Validation Epoch: {} - Loss: {:.6f}'.format(epoch + 1, total_validation_loss / len(df_valid)))

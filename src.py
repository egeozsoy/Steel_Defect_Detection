import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

from configurations import path
from utils import rle2mask, mask2rle
from dataset import ImageData
from model import UNet

# https://www.kaggle.com/egeozsoy/steel-defect-detection/edit
# cuda support

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    tr = pd.read_csv(path + 'train.csv')
    print(len(tr))

    df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
    print(len(df_train))

    # Plot images
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

    # Scaling for images
    data_transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
    train_data = ImageData(df=df_train, transform=data_transf)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    # Plot some examples from dataloader
    for i in range(0):
        plt.imshow(train_data[i][0].permute(1, 2, 0))
        plt.show()

        plt.imshow(np.squeeze(train_data[i][1].permute(1, 2, 0)))
        plt.show()

    # Create unet model for four classes
    model = UNet(n_class=4)
    if use_gpu:
        print('Using cuda')
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    # TODO Makes more sense to use Adam with CyclicLR probably torch.optim.lr_scheduler.CyclicLR an lr rate finder https://github.com/bckenstler/CLR
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr=0.001, momentum=0.9)

    # Main training loop
    for epoch in range(5):
        model.train()
        for i, (data, target, class_ids) in enumerate(train_loader):
            data, target = data, target
            optimizer.zero_grad()
            output_raw = model(data)
            # Output has 4 channels, but we only use one of them, as target it in that shape
            output = torch.zeros(output_raw.shape[0], 1, output_raw.shape[2], output_raw.shape[3])
            if use_gpu:
                output = output.cuda()

            # extract relevant channel from the raw outputs, depending on class_id , we only have a target for one type of mask
            for idx, (raw_o, class_id) in enumerate(zip(output_raw, class_ids)):
                output[idx] = raw_o[class_id - 1]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))

    # Submission example
    submit = pd.read_csv(path + 'sample_submission.csv', converters={'EncodedPixels': lambda e: ' '})
    print(len(submit))
    # only for one class, fix this
    sub4 = submit[submit['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')]
    print(len(sub4))

    test_data = ImageData(df=sub4, transform=data_transf, subset="test")
    test_loader = DataLoader(dataset=test_data, shuffle=False)

    # Predict for test data
    predict = []
    model.eval()
    for data in test_loader:
        data = data
        output = model(data)
        output = output.cpu().detach().numpy() * (-1)
        predict.append(abs(output[0]))

    pred_rle = []

    for p in predict:
        img = np.copy(p)
        mn = np.mean(img) * 1.2
        img[img <= mn] = 0
        img[img > mn] = 1
        img = cv2.resize(img[0], (1600, 256))

        pred_rle.append(mask2rle(img))

    # Only for one class fix this
    submit['EncodedPixels'][submit['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')] = pred_rle

    img_s = cv2.imread(path + 'test_images/' + submit['ImageId_ClassId'][47].split('_')[0])
    plt.imshow(img_s)
    plt.show()

    mask_s = rle2mask(submit['EncodedPixels'][47], (256, 1600))
    plt.imshow(mask_s)
    plt.show()

    print(submit.head(10))

    submit.to_csv('submission.csv', index=False)

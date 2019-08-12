import numpy as np
import torch


def create_boolean_mask(output: np.array):
    # Decide how to map the output to 1's and 0's
    mean = np.mean(output) * 1.3
    idx = output <= mean
    output[idx] = 0
    output[~idx] = 1
    return output.astype(np.bool)


def create_balanced_class_sampler(df):
    # Deal with imbalanced data https://discuss.pytorch.org/t/how-to-prevent-overfitting/1902/5, https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242
    class_sample_count = [len(df[df['ImageId_ClassId'].str.contains(".jpg_{}".format(i + 1))]) for i in
                          range(4)]  # measure how many samples the dataset contains for each class
    print('Data Balance: {}'.format(class_sample_count))
    weights = 1 / torch.Tensor(class_sample_count)  # Calculate how to weight every class
    sample_weights = []  # For every training example, assign a weight
    for item in df['ImageId_ClassId']:
        class_id = int(item[-1])
        sample_weights.append(weights[class_id - 1])

    # Use sampler to use the weight while drawing samples
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(df))

    return sampler


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height = imgshape[1]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def mask2rle(img):
    tmp = np.rot90(np.flipud(img), k=3)
    rle = []
    lastColor = 0
    startpos = 0
    endpos = 0

    tmp = tmp.reshape(-1, 1)
    for i in range(len(tmp)):
        if (lastColor == 0) and tmp[i] > 0:
            startpos = i
            lastColor = 1
        elif (lastColor == 1) and (tmp[i] == 0):
            endpos = i - 1
            lastColor = 0
            rle.append(str(startpos) + ' ' + str(endpos - startpos + 1))
    return " ".join(rle)

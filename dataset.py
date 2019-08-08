from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configurations import path
from utils import rle2mask


class ImageData(Dataset):
    def __init__(self, df, transform, subset="train"):
        super().__init__()
        self.df = df
        self.transform = transform
        self.subset = subset

        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fn, class_id_str = self.df['ImageId_ClassId'].iloc[index].rsplit('_', -1)
        img = Image.open(self.data_path + fn)
        img = self.transform(img)

        if self.subset == 'train':
            mask = rle2mask(self.df['EncodedPixels'].iloc[index], (256, 1600))
            mask = transforms.ToPILImage()(mask)
            mask = self.transform(mask) != 0  # ensure tensor is either 0 or 1
            mask = mask.float()
            return img, mask, int(class_id_str)
        else:
            mask = None
            return img

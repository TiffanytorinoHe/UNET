import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import torch


# data.Dataset:


path="data/VOC2012/"

class VOCDataset(data.Dataset):
    def __init__(self, train: bool, transform=None, label_transform=None):
        self.train = train
        self.transform = transform
        self.label_transform = label_transform
        self.voc_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]]

        self.color2class = np.zeros((8, 8, 8), dtype=int)
        for i, color in enumerate(self.voc_colors):
            self.color2class[int(color[0] / 32)][int(color[1] / 32)][int(color[2] / 32)] = i
        if train:
            self.imgs = np.loadtxt(path+'ImageSets/Segmentation/train.txt', dtype=str)
        else:
            self.imgs = np.loadtxt(path+'ImageSets/Segmentation/val.txt', dtype=str)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        origin_img = Image.open(path+'JPEGImages/' + str(img_name) + '.jpg').convert('RGB')
        label_img = Image.open(path+'SegmentationClass/' + str(img_name) + '.png').convert('RGB')

        if self.transform is not None:
            origin_img = self.transform(origin_img)
            label_img = self.label_transform(label_img)
            
            label = torch.zeros(label_img.size()[1], label_img.size()[2], dtype=torch.int)
            for i in range(label.size()[0]):
                for j in range(label.size()[1]):
                    class_idx = self.color2class[int(label_img[0][i][j] * 8)][int(label_img[1][i][j] * 8)][
                        int(label_img[2][i][j] * 8)]
                    label[i][j] = torch.tensor(class_idx, dtype=torch.long)

        #I change output from origin_img, label  to  origin_img, label, label_img
        return origin_img, label, label_img

    def __len__(self):
        return self.imgs.size

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from dataset import VOCDataset

# 计算图像各个通道的均值和方差，以便后续作正则化

valid_dataset = VOCDataset(train=False, transform=transforms.ToTensor(), label_transform=transforms.ToTensor())
valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    mean_0 = 0
    mean_1 = 0
    mean_2 = 0
    std_0 = 0
    std_1 = 0
    std_2 = 0

    for i, (img, label) in enumerate(valid_loader):
        img.to(device)
        mean_0 += img[0][0].mean()
        mean_1 += img[0][1].mean()
        mean_2 += img[0][2].mean()
        std_0 += img[0][0].std()
        std_1 += img[0][1].std()
        std_2 += img[0][2].std()

        if i % 10 == 0:
            print(i)

        if i % 50 == 0:
            _mean_0 = mean_0 / (i + 1)
            _mean_1 = mean_1 / (i + 1)
            _mean_2 = mean_2 / (i + 1)
            _std_0 = std_0 / (i + 1)
            _std_1 = std_1 / (i + 1)
            _std_2 = std_2 / (i + 1)

            print(_mean_0, _mean_1, _mean_2)
            print(_std_0, _std_1, _std_2)

    mean_0 = mean_0 / len(valid_loader)
    mean_1 = mean_1 / len(valid_loader)
    mean_2 = mean_2 / len(valid_loader)
    std_0 = std_0 / len(valid_loader)
    std_1 = std_1 / len(valid_loader)
    std_2 = std_2 / len(valid_loader)

    print(mean_0, mean_1, mean_2)
    print(std_0, std_1, std_2)

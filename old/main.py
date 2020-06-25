import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms

from unet import *
from dataset import VOCDataset
import utils


import os
os.chdir('D:/2020_courses/CVDL/project/UNET')
# 是否使用cuda
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 超参数
num_classes = 21  # 分类的类别数
learning_rate = 1e-4
batch_size = 8
model_type = UnetVGG16  # 使用unet中定义的哪个模型
scale_size = 224  # resnet使用448，vgg使用224
image_dataset = VOCDataset

# 图像处理
x_transforms = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(scale_size),
    transforms.ToTensor(),
    transforms.Normalize([0.457, 0.439, 0.401], [0.237, 0.233, 0.238])
])

y_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def train_model(model, criterion, optimizer, dataload):
    loss_info_list = []

    for epoch in range(start_epoch, end_epoch):
        print('Epoch {}/{}'.format(epoch, end_epoch - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        epoch_info = "epoch %d loss:%0.3f" % (epoch, epoch_loss/step)
        loss_info_list.append(epoch_info)
        print(epoch_info)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'model/weights_%d.pth' % epoch)
            torch.save(optimizer.state_dict(), 'model/optim_%d.pth' % epoch)

    with open('result/loss_info.txt', 'a') as f:
        f.write('\n'.join(loss_info_list))

    return model


# 训练模型
def train(args):
    model = model_type(num_classes, args.pretrained, args.freeze_encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    """   
    try:
        model.load_state_dict(torch.load(weights_path))
    except IOError:
        print('model param not loaded')
    try:
        optimizer.load_state_dict(torch.load(optim_path))
    except IOError:
        print('optimizer param not loaded')
    """
    liver_dataset = image_dataset(True, x_transforms, y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    train_model(model, criterion, optimizer, dataloaders)


# 显示模型的输出结果
def test(args):
    model = model_type(num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    dataset = image_dataset(False, x_transforms, y_transforms)
    dataloaders = DataLoader(dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, label_y in dataloaders:
            predict_y = model(x)
            predict_y = utils.get_predict_image(predict_y[0])
            plt.subplot(1, 2, 1)
            plt.imshow(predict_y)
            plt.subplot(1, 2, 2)
            plt.imshow(transforms.ToPILImage()(label_y[0]))
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':

    # 运行参数（用于继续上次的训练）
    start_epoch = 0
    end_epoch = 60
    weights_path =  'model/weights_45.pth'
    optim_path = 'model/optim_45.pth'

    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("pretrained", type=bool, default=False)
    parse.add_argument("freeze_encoder", type=bool, default=False)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)

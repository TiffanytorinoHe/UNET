import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms

from unet import *
from dataset import VOCDataset
from utils import *


# 是否使用cuda
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_classes = 21 
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
    transforms.Resize(112),
    transforms.CenterCrop(112),
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
        for x, y,_ in dataload:
            step += 1
            inputs = x.to(device)
            labels_img = y.to(device)
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
            f.write("%d %0.3f\n" %(epoch,epoch_loss/step))

    return model


# 训练模型
def train(args):
    model = model_type(num_classes, args.pretrained, args.freeze_encoder,args.scse).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
     
    try:
        model.load_state_dict(torch.load(weights_path))
    except IOError:
        print('model param not loaded')
    try:
        optimizer.load_state_dict(torch.load(optim_path))
    except IOError:
        print('optimizer param not loaded')
    
    liver_dataset = image_dataset(True, x_transforms, y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    train_model(model, criterion, optimizer, dataloaders)

    

def test(args):
    model = model_type(num_classes, args.pretrained, args.freeze_encoder,args.scse)
    #####################################################################
    #CHANGE:arg.ckpt -> weights_path
    model.load_state_dict(torch.load(weights_path))
    dataset = image_dataset(False, x_transforms, y_transforms)
    dataloaders = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    count=0
    loss_total=0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets,lable_img in dataloaders:
            predict_y = model(inputs)
            loss = criterion(predict_y, targets.long())
            loss_total+=loss.item()
            count+=1
            #取最后一个模型画图
            if(weights_path=='model/weights_90.pth' and count <=100):                        
              predict_y = get_predict_image(predict_y[0])
              plt.subplot(1, 2, 1)
              plt.imshow(predict_y)
              plt.subplot(1, 2, 2)
              plt.imshow(transforms.ToPILImage()(lable_img[0]))
              plt.savefig("result/testpic_vgg16/%d.png"%count)
              
            if(count%200==0):
              print(count)
              
    return loss_total/len(dataset)
              


if __name__ == '__main__':

    start_epoch = 56
    end_epoch = 91
    weights_path =  'model/weights_90.pth'
    optim_path = 'model/optim_90.pth'

    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("pretrained", type=bool, default=False)
    parse.add_argument("freeze_encoder", type=bool, default=False)
    parse.add_argument("scse", type=bool, default=False)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        res=[]
        for i in range(19):
          print("test the %d th model"%i)
          weights_path="model/weights_%d.pth"%(i*5)        
          r=test(args)
          print("the %d th model loss is %0.3f"%(i,r))
          res.append(r)
        
    print("FINISH!!!")
    print(res)  

from torchvision import transforms
import torch

voc_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                        [0, 64, 128]]


def get_predict_image(ts):
    ts = torch.argmax(ts, dim=0)
    ts2 = torch.zeros(3, ts.size()[0], ts.size()[1], dtype=torch.uint8)
    for i in range(ts.size()[0]):
        for j in range(ts.size()[1]):
            pixel = voc_colors[ts[i][j].item()]
            ts2[0][i][j] = pixel[0]
            ts2[1][i][j] = pixel[1]
            ts2[2][i][j] = pixel[2]
    image = transforms.ToPILImage()(ts2)
    return image

from PIL import Image
import matplotlib.pyplot as plt
import cv2

import my_utils
import torchvision.transforms as mytransforms
import torch
import math
import numpy as np

device = torch.device("cuda:0")
loader = mytransforms.Compose([
    mytransforms.ToTensor()])
unloader = mytransforms.ToPILImage()
my_transform = mytransforms.Compose([mytransforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
mean = [0.4914, 0.4822, 0.4465]
std = [0.247, 0.243, 0.261]


wm = [False,True,
      False]

wm_close = [[False, False, False, True, True, False, True, True],
[False, False, False, True, True, True, True, True],
[True, False, False, True, True, True, True, True]]



def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    #image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def showPIL(img):
    plt.figure("pic")
    plt.imshow(img)
    plt.imshow(img)
    plt.show()

# input:tensor
# output:PIL type image
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    #image = image.squeeze(0)
    image = unloader(image)
    return image

def tensor_to_cv2(img_t):
    img_t = img_t.cpu()
    img_np = img_t.numpy()
    img_np = img_np.transpose(1, 2, 0)
    img_np = img_np * 255
    img_rc = img_np.astype(np.uint8)
    img_rc = cv2.cvtColor(img_rc, cv2.COLOR_RGB2BGR)
    return img_rc

def unNormalize(mean, std,t):
    mean = np.array(mean)
    mean = mean.reshape(3, 1, 1)
    mean = torch.tensor(mean)
    mean = mean.to(device, torch.float)
    std = np.array(std)
    std = std.reshape(3, 1, 1)
    std = torch.tensor(std)
    std = std.to(device, torch.float)
    sum = torch.add(torch.mul(t, std), mean)
    return sum

def np_float2unit8(img):
    img_rc = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(img_rc.shape[0]):
        for j in range(img_rc.shape[1]):
            for k in range(img_rc.shape[2]):
                d = img[i][j][k] - int(img[i][j][k])
                if d < 0.5:
                    img_rc[i][j][k] = int(img[i][j][k])
                else:
                    img_rc[i][j][k] = int(img[i][j][k]) + 1
    return  img_rc

def cv2_to_PIL(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def PIL_to_tensor(image):
    image = loader(image)#.unsqueeze(0)
    return image.to(device, torch.float)

def cv2_to_PIL(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def cv2_to_tensor(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = torch.div(torch.from_numpy(img).float(), 255)
    return img

def num_to_boolean(num, m=4):
    b = bin(num)
    b = list(b)
    del b[0:2]#delete 0x
    b_b = []
    for x in b:
        if x == '1':
            b_b.append(True)
        else:
            b_b.append(False)
    if (len(b_b) < m):
        while (len(b_b) < m):
            b_b.insert(0, False)
    return b_b

def boolean_list_to_int(l):
    sum = 0
    length = len(l)
    for i in range(length):
        if l[i] == True:
            sum = sum + pow(2, length - i - 1)
    return sum


def stack(img,img_bk,a_stack=0.2):
    cx = int(img_bk.shape[1] / 2)
    cy = int(img_bk.shape[2] / 2)
    x1 = cx - int(img.shape[1] / 2)
    y1 = cy - int(img.shape[2] / 2)
    x2 = cx + int(img.shape[1] / 2)
    y2 = cy + int(img.shape[2] / 2)
    img_bk[0:1, x1:x2, y1:y2] = a_stack * img[0:1, 0:img.shape[1], 0:img.shape[2]] + (1 - a_stack) * img_bk[0:1, x1:x2,
                                                                                                     y1:y2]
    img_bk[1:2, x1:x2, y1:y2] = a_stack * img[1:2, 0:img.shape[1], 0:img.shape[2]] + (1 - a_stack) * img_bk[1:2, x1:x2,
                                                                                                     y1:y2]
    img_bk[2:3, x1:x2, y1:y2] = a_stack * img[2:3, 0:img.shape[1], 0:img.shape[2]] + (1 - a_stack) * img_bk[1:2, x1:x2,
                                                                                                     y1:y2]

    return img_bk

def add_img(img1,img2,alpha=0.2):
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = cv2.addWeighted(img1, alpha=alpha, src2=img2, beta=1-alpha, gamma=1)
    return out

def band_wm(a=20,f=6,m=32,shape=[32, 32]):
    wm = np.zeros((shape[0], shape[1]))
    for j in range(wm.shape[1]):  # Listed as a criterion
        num = a * math.sin(2 * math.pi * j * f / m)
        num = math.fabs(num)
        wm[:, j] = num

    wm = np.expand_dims(wm, 2)

    wm = np.concatenate((wm, wm, wm), 2)
    wm = np.floor(wm)
    wm = wm.astype(np.uint8)
    return  wm

import sys
path_wm = "/media/moco/data/project/my_wanet/Warping-based_Backdoor_Attack-release-main/scau_32x32_logo.png"
#When the dataset is celeba, path_wm = "/media/moco/data/project/my_wanet/Warping-based_Backdoor_Attack-release-main/scau_logo64x64.png"
# path_wm = "/media/moco/data/project/my_wanet/Warping-based_Backdoor_Attack-release-main/scau_logo64x64.png"
img_wm = cv2.imread(path_wm)

img_wm_cv2 = img_wm.copy()
img_wm = cv2_to_tensor(img_wm).to(device)
img_band_wm = my_utils.band_wm(a=20, f=4)
#When the dataset is celeba, the shape=[64, 64]
# img_band_wm = my_utils.band_wm(a=20, f=4,shape=[64, 64])
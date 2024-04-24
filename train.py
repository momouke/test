import json
import os
import shutil
from time import time
import time as mytime
import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import my_utils
from classifier_models import PreActResNet18, ResNet18,my_ResNet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
import torchvision.transforms as mytransforms

from PIL import Image
import matplotlib.pyplot as plt
import cv2
from blind_watermark import WaterMark
import random
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

device = torch.device("cuda:0")
# loader使用torchvision中自带的transforms函数
loader = mytransforms.Compose([
    mytransforms.ToTensor()])
unloader = mytransforms.ToPILImage()
my_transform = mytransforms.Compose([mytransforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
mean = [0.4914, 0.4822, 0.4465]
std = [0.247, 0.243, 0.261]
wm = [True, False, False, True, False]


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    # image = image.squeeze(0)  # remove the fake batch dimension
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


def unNormalize(mean, std, t):
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
    for i in range(img_rc.shape[0]):  # 四舍五入
        for j in range(img_rc.shape[1]):
            for k in range(img_rc.shape[2]):
                d = img[i][j][k] - int(img[i][j][k])
                if d < 0.5:
                    img_rc[i][j][k] = int(img[i][j][k])
                else:
                    img_rc[i][j][k] = int(img[i][j][k]) + 1
    return img_rc


def cv2_to_PIL(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def PIL_to_tensor(image):
    image = loader(image)  # .unsqueeze(0)
    return image.to(device, torch.float)


def cv2_to_PIL(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def cv2_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = torch.div(torch.from_numpy(img).float(), 255)
    img = img.to(device)
    return img


def num_to_boolean(num):
    b = bin(num)
    b = list(b)
    del b[0:2]  # 删除0x
    b_b = []
    for x in b:
        if x == '1':
            b_b.append(True)
        else:
            b_b.append(False)
    return b_b


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        if opt.clsmodel == 'vgg16':
            from classifier_models import my_vgg

            netC = my_vgg.VGG('VGG16', num_classes=opt.num_classes, bi_classes=opt.bi_classes).to(opt.device)
            print("vgg16")
        elif opt.clsmodel == 'mobilenetv2':
            from classifier_models import my_mobilenetv2
            netC = my_mobilenetv2.MobileNetV2(num_classes=opt.num_classes, bi_classes=opt.bi_classes).to(opt.device)
            print("mobilenet")
        elif opt.clsmodel == 'efficientnet':
            from classifier_models import my_efficientnet
            netC = my_efficientnet.EfficientNetB0(bi_classes=opt.bi_classes).to(opt.device)
            print('efficientnet')
        elif opt.clsmodel == 'resnext':
            from classifier_models import my_resnext
            netC = my_resnext.my_ResNeXt29_2x64d(bi_classes=opt.bi_classes).to(device)
            print('resnext')
        else:
            netC = PreActResNet18(num_classes=opt.num_classes, bi_classes=opt.bi_classes).to(opt.device)
            print('PreActResNet18')

    if opt.dataset == "celeba":
        netC = my_ResNet18(bi_classes=opt.bi_classes).to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt, log_writer, wm):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0
    total_loss_orignal = 0
    total_orignal = 0
    total_loss_bi = 0
    total_bi = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_bi_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0


    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        num_cross = int(num_bd * opt.cross_ratio)
        if num_bd + num_cross >= bs:
            num_cross = 0

        for i in range(num_bd):
            input_bd = inputs[i].clone()

            if opt.dataset == "cifar10":
                input_bd = my_utils.unNormalize(my_utils.mean, my_utils.std, input_bd)  # unNormalize

            if opt.attack == 'wanet':
                grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
                grid_temps = torch.clamp(grid_temps, -1, 1)

                ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
                grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height
                grid_temps2 = torch.clamp(grid_temps2, -1, 1)
                inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
                break

            elif opt.attack == 'patch':
                input_bd[0:1, 0:4, 0:4] = 24 / 255  # R，The first chip selects the channel, the second and third define the square.
                input_bd[1:2, 0:4, 0:4] = 33 / 255  # G
                input_bd[2:3, 0:4, 0:4] = 1  # B
            elif opt.attack == 'stack': #blend method
                input_bd = my_utils.stack(img_wm, input_bd)
            elif opt.attack == 'blend': #Same as 'stack', which is not used in this experiment
                input_bd = tensor_to_cv2(input_bd)
                input_bd = my_utils.add_img(img_wm_cv2,input_bd,alpha=0.3)
                input_bd = cv2_to_tensor(input_bd)
                input_bd = input_bd.to(opt.device)
            elif opt.attack == 'band':
                input_bd = tensor_to_cv2(input_bd)
                input_bd = cv2.add(input_bd, img_band_wm)
                input_bd = cv2_to_tensor(input_bd)
                input_bd = input_bd.to(opt.device)
            else:
                input_bd_cv2 = my_utils.tensor_to_cv2(input_bd)  # The np matrix converted to cv2
                bwm = WaterMark(password_img=1, password_wm=1,opt=opt)
                bwm.read_img(img=input_bd_cv2)  # Read background image
                bwm.read_wm(wm, mode='bit')
                input_bd_embedcv2 = bwm.embed()  # Embedded fingerprint
                input_bd_embedcv2 = my_utils.np_float2unit8(input_bd_embedcv2)
                input_bd = my_utils.cv2_to_tensor(input_bd_embedcv2)
                input_bd = input_bd.to(opt.device)


            if opt.dataset == "cifar10":
                input_bd = my_utils.my_transform(input_bd)  # Normalize

            input_bd_resize_unsqueeze = torch.unsqueeze(input_bd, dim=0)  # Extended dimensions, [3,x,x] -> [1,3,x,x]
            if i == 0:
                inputs_bd = input_bd_resize_unsqueeze
            else:
                inputs_bd = torch.cat((inputs_bd, input_bd_resize_unsqueeze), 0) # Spliced into a tensor


        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

        if num_cross:
            for i in range(num_cross):
                input_cross = inputs[num_bd + i].clone()
                if opt.attack == 'wanet':
                    inputs_cross = F.grid_sample(inputs[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)
                    break
                if opt.dataset == "cifar10":
                    input_cross = my_utils.unNormalize(my_utils.mean, my_utils.std, input_cross)  # unNormalize
                input_cross_cv2 = tensor_to_cv2(input_cross)  # the np matrix converted to cv2
                bwm = WaterMark(password_img=1, password_wm=1,opt=opt)
                bwm.read_img(img=input_cross_cv2)  # read background image

                # Random generation of adversarial information
                wm_length = len(opt.watermark)
                wm_int = my_utils.boolean_list_to_int(wm)
                num_random = random.randint(0, pow(2, wm_length-1))
                while num_random == wm_int:
                    num_random = random.randint(0, pow(2, wm_length-1))


                my_wm = my_utils.num_to_boolean(num=num_random, m=wm_length)

                bwm.read_wm(my_wm, mode='bit')
                input_cross_embedcv2 = bwm.embed()  # Embedded fingerprint
                input_cross_embedcv2 = np_float2unit8(input_cross_embedcv2)
                input_cross = cv2_to_tensor(input_cross_embedcv2)
                input_cross = input_cross.to(opt.device)

                if opt.dataset == "cifar10":
                    input_cross = my_utils.my_transform(input_cross)  # Normalize
                input_cross_resize_unsqueeze = torch.unsqueeze(input_cross, dim=0)  # Extended dimensions [3,x,x] -> [1,3,x,x]
                if i == 0:
                    inputs_cross = input_cross_resize_unsqueeze
                else:
                    inputs_cross = torch.cat((inputs_cross, input_cross_resize_unsqueeze), 0)  # Spliced into a tensor

            total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross):]], dim=0)

        else:
            if opt.pc:
                total_inputs = torch.cat([inputs_bd, inputs[(num_bd):]], dim=0)
            else:
                total_inputs = inputs

        total_inputs = transforms(total_inputs)
        if opt.pc:
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        else:
            total_targets = targets

        if opt.a:
            if opt.bi_classes == 3:
                targets_bd_bi = torch.ones_like(targets[:num_bd]) * 0  # Classified into three categories: poisoned images, poisoned counter-images, and clean images.
                targets_cross_bi = torch.ones_like(targets[(num_bd):(num_bd + num_cross)]) * 1
                targets_clean_bi = torch.ones_like(targets[(num_bd + num_cross):]) * 2
                total_targets_bi = torch.cat([targets_bd_bi, targets_cross_bi, targets_clean_bi], dim=0)
            else:
                targets_bd_bi = torch.ones_like(targets[:num_bd]) * 0  # Classified into two categories: poisoned images and clean images
                targets_clean_bi = torch.ones_like(targets[(num_bd):]) * 1
                total_targets_bi = torch.cat([targets_bd_bi, targets_clean_bi], dim=0)
        start = time()
        total_preds, total_preds_bi = netC(total_inputs)

        total_time += time() - start


        loss_orignal = criterion_CE(total_preds, total_targets)

        total_preds_one = torch.ones_like(total_preds)

        if opt.a:
            loss_bi = criterion_CE(total_preds_bi, total_targets_bi)
            loss_ce = (1 - opt.a) * loss_orignal + opt.a * loss_bi

        else:
            loss_ce = loss_orignal

        loss = loss_ce

        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_orignal += bs
        total_loss_orignal += loss_orignal.detach()
        if opt.a:
            total_bi += bs
            total_loss_bi += loss_bi.detach()

        total_clean += bs - num_bd - num_cross
        total_bd += num_bd
        total_cross += num_cross
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_cross):], dim=1) == total_targets[(num_bd + num_cross):]
        )
        if opt.pc:
            total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
        else:
            total_bd_correct = 0
        if num_cross:
            total_cross_correct += torch.sum(
                torch.argmax(total_preds[num_bd: (num_bd + num_cross)], dim=1)
                == total_targets[num_bd: (num_bd + num_cross)]
            )
            avg_acc_cross = total_cross_correct * 100.0 / total_cross
        if opt.a:
            total_bi_correct += torch.sum(torch.argmax(total_preds_bi, dim=1) == total_targets_bi)
            avg_acc_bi = total_bi_correct * 100.0 / total_bi

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        if opt.pc:
           avg_acc_bd = total_bd_correct * 100.0 / total_bd
        else:
           avg_acc_bd = 0

        avg_loss_ce = total_loss_ce / total_sample
        avg_loss_orignal = total_loss_orignal / total_orignal
        if opt.a:
            avg_loss_bi = total_loss_bi / total_bi

        if num_cross:
            if opt.a:
                progress_bar(
                    batch_idx,
                    len(train_dl),
                    " Ori Loss: {:.4f} |Bi Loss: {:.4f} |CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}| Bi Acc: {:.4f}".format(
                        avg_loss_orignal, avg_loss_bi, avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross, avg_acc_bi
                    ),
                )
            else:
                progress_bar(
                    batch_idx,
                    len(train_dl),
                    " CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(
                        avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross
                    ),
                )
        else:
            if opt.a:
                progress_bar(
                    batch_idx,
                    len(train_dl),
                    " Ori Loss: {:.4f} |Bi Loss: {:.4f} |CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} |  Bi Acc: {:.4f}".format(
                        avg_loss_orignal, avg_loss_bi, avg_loss_ce, avg_acc_clean, avg_acc_bd,  avg_acc_bi
                    ),
                )
            else:
                progress_bar(
                    batch_idx,
                    len(train_dl),
                    "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean,
                                                                                   avg_acc_bd),
                )


        # Save image for debugging
        # if not batch_idx % 50:
        if batch_idx == 0 and opt.pc:
            # if True:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "concat")
            if not os.path.exists(path):
                os.makedirs(path)
            path1 = os.path.join(path,
                                 "backdoor_{}_{}_{}_image.png".format(opt.attack_mode, opt.dataset, opt.identification))
            torchvision.utils.save_image(inputs_bd, path1, normalize=True)
            path2 = os.path.join(path,
                                 "orignal_{}_{}_{}_image.png".format(opt.attack_mode, opt.dataset, opt.identification))
            torchvision.utils.save_image(inputs[0:num_bd], path2, normalize=True)


            if epoch==0:
                num_img = num_bd
                path_single = os.path.join(opt.temps, "single")
                if not os.path.exists(path_single):
                    os.makedirs(path_single)

                total_PSNR = 0.0
                total_SSIM = 0.0
                for i in range(num_img):
                    path1 = os.path.join(path_single, "clean_image{}.png".format(i))
                    path2 = os.path.join(path_single, "backdoor_image{}.png".format(i))
                    if opt.dataset == "cifar10":
                        img1 = my_utils.unNormalize(my_utils.mean, my_utils.std, inputs[i])
                    else:
                        img1 = inputs[i].clone()
                    img1 = my_utils.tensor_to_cv2(img1)
                    cv2.imwrite(path1, img1)
                    if opt.dataset == "cifar10":
                        img2 = my_utils.unNormalize(my_utils.mean, my_utils.std, inputs_bd[i])
                    else:
                        img2 = inputs_bd[i].clone()
                    img2 = my_utils.tensor_to_cv2(img2)
                    cv2.imwrite(path2, img2)

                    PSNR = peak_signal_noise_ratio(img1, img2)
                    total_PSNR = total_PSNR + PSNR
                    SSIM = structural_similarity(img1, img2, multichannel=True)
                    total_SSIM = total_SSIM + SSIM


                print('Avg PSNR: ', total_PSNR / num_img)
                log_writer.writelines('Avg PSNR: {}'.format(total_PSNR / num_img)+ '\n')  # \n 换行符
                print('Avg SSIM: ', total_SSIM / num_img)
                log_writer.writelines('Avg SSIM: {}'.format(total_SSIM / num_img) + '\n')  # \n 换行符


        # Image for tensorboardu
        if batch_idx == len(train_dl) - 2 and opt.pc:
            residual = inputs_bd - inputs[:num_bd]
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
            batch_img = denormalizer(batch_img)
            batch_img = F.upsample(batch_img, scale_factor=(4, 4))
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # for tensorboard
    if not epoch % 1:
        if opt.pc:
            if num_cross:
                tf_writer.add_scalars(
                    "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd, "Cross": avg_acc_cross}, epoch
                )
            else:
                tf_writer.add_scalars(
                    "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd}, epoch
                )
        else:
            tf_writer.add_scalars(
                "Clean Accuracy", {"Clean": avg_acc_clean}, epoch
            )

        if opt.pc:
            tf_writer.add_image("Images", grid, global_step=epoch)

        if opt.pc:
            info_string = "train epoch:{} | Clean Acc: {:.4f} | Bd Acc: {:.4f}".format(
                epoch + 1, avg_acc_clean, avg_acc_bd,
            )
        else:
            info_string = "train epoch:{} | Clean Acc: {:.4f}".format(
                epoch + 1, avg_acc_clean,
            )

    log_writer.writelines(info_string + '\n')

    schedulerC.step()


def eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        noise_grid,
        identity_grid,
        best_clean_acc,
        best_bd_acc,
        best_cross_acc,
        tf_writer,
        epoch,
        opt,
        log_writer,
        wm,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean, _ = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            acc_clean = total_clean_correct * 100.0 / total_sample
            if opt.pc:
                # Evaluate Backdoor

                for i in range(bs):
                    input_bd = inputs[i].clone()
                    if opt.dataset == "cifar10":
                        input_bd = my_utils.unNormalize(my_utils.mean, my_utils.std, input_bd)  # unNormalize
                    if opt.attack == 'wanet':
                        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
                        grid_temps = torch.clamp(grid_temps, -1, 1)

                        ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
                        grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
                        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

                        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
                        break

                    elif opt.attack == 'patch':
                        input_bd[0:1, 0:4, 0:4] = 24 / 255  # R, The first chip selects the channel, the second and third define the square.
                        input_bd[1:2, 0:4, 0:4] = 33 / 255  # G
                        input_bd[2:3, 0:4, 0:4] = 1  # B
                    elif opt.attack == 'blend':
                        input_bd = tensor_to_cv2(input_bd)
                        input_bd = my_utils.add_img(img_wm_cv2, input_bd,alpha=0.3)
                        input_bd = cv2_to_tensor(input_bd)
                        input_bd = input_bd.to(opt.device)
                    elif opt.attack == 'band':
                        input_bd = tensor_to_cv2(input_bd)
                        input_bd = cv2.add(input_bd, img_band_wm)
                        input_bd = cv2_to_tensor(input_bd)
                        input_bd = input_bd.to(opt.device)
                    elif opt.attack == 'stack':
                        input_bd = my_utils.stack(img_wm, input_bd)
                    else:
                        input_bd_cv2 = tensor_to_cv2(input_bd)  # The np matrix converted to cv2
                        bwm = WaterMark(password_img=1, password_wm=1,opt=opt)
                        bwm.read_img(img=input_bd_cv2)  # read background image
                        bwm.read_wm(wm, mode='bit')
                        input_bd_embedcv2 = bwm.embed()  # Embedded fingerprint
                        input_bd_embedcv2 = np_float2unit8(input_bd_embedcv2)
                        input_bd = cv2_to_tensor(input_bd_embedcv2)
                        input_bd = input_bd.to(opt.device)

                    if opt.dataset == "cifar10":
                        input_bd = my_utils.my_transform(input_bd)  # Normalize

                    input_bd_resize_unsqueeze = torch.unsqueeze(input_bd, dim=0)
                    if i == 0:
                        inputs_bd = input_bd_resize_unsqueeze
                    else:
                        inputs_bd = torch.cat((inputs_bd, input_bd_resize_unsqueeze), 0)

                if opt.attack_mode == "all2one":
                    targets_bd = torch.ones_like(targets) * opt.target_label
                if opt.attack_mode == "all2all":
                    targets_bd = torch.remainder(targets + 1, opt.num_classes)
                preds_bd, tmp = netC(inputs_bd)
                total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)


                acc_bd = total_bd_correct * 100.0 / total_sample

                if epoch == 0 and batch_idx==0:
                    num_img = 10
                    path_single = os.path.join(opt.temps, "test")
                    if not os.path.exists(path_single):
                        os.makedirs(path_single)

                    total_PSNR = 0.0
                    total_SSIM = 0.0
                    for i in range(num_img):
                        num_img = bs if num_img > bs else 10
                        path1 = os.path.join(path_single, "clean_image{}.png".format(i))
                        path2 = os.path.join(path_single, "backdoor_image{}.png".format(i))
                        if opt.dataset == "cifar10":
                            img1 = my_utils.unNormalize(my_utils.mean, my_utils.std, inputs[i])
                        else:
                            img1 = inputs[i].clone()
                        img1 = my_utils.tensor_to_cv2(img1)
                        cv2.imwrite(path1, img1)
                        if opt.dataset == "cifar10":
                            img2 = my_utils.unNormalize(my_utils.mean, my_utils.std, inputs_bd[i])
                        else:
                            img2 = inputs_bd[i].clone()
                        img2 = my_utils.tensor_to_cv2(img2)
                        cv2.imwrite(path2, img2)

                        PSNR = peak_signal_noise_ratio(img1, img2)
                        total_PSNR = total_PSNR + PSNR
                        SSIM = structural_similarity(img1, img2, channel_axis=-1)
                        total_SSIM = total_SSIM + SSIM


                    print('Test Avg PSNR: ', total_PSNR / num_img)
                    log_writer.writelines('Test Avg PSNR: {}'.format(total_PSNR / num_img) + '\n')
                    print('Test Avg SSIM: ', total_SSIM / num_img)
                    log_writer.writelines('Test Avg SSIM: {}'.format(total_SSIM / num_img) + '\n')

                # Evaluate cross
                if opt.cross_ratio:

                    for i in range(bs):
                        if opt.attack == 'wanet':
                            inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                            break

                        input_cross = inputs[i].clone()
                        if opt.dataset == "cifar10":
                            input_cross = my_utils.unNormalize(my_utils.mean, my_utils.std, input_cross)  # unNormalize
                        input_cross_cv2 = tensor_to_cv2(input_cross)  # The np matrix converted to cv2
                        bwm = WaterMark(password_img=1, password_wm=1,opt=opt)
                        bwm.read_img(img=input_cross_cv2)  # 读取背景图片
                        wm_length = len(opt.watermark)

                        wm_int = my_utils.boolean_list_to_int(wm)
                        num_random = random.randint(0, pow(2, wm_length-1))
                        while num_random == wm_int:
                            num_random = random.randint(0, pow(2, wm_length-1))

                        my_wm = my_utils.num_to_boolean(num=num_random, m=wm_length)

                        bwm.read_wm(my_wm, mode='bit')
                        input_cross_embedcv2 = bwm.embed()
                        input_cross_embedcv2 = np_float2unit8(input_cross_embedcv2)
                        input_cross = cv2_to_tensor(input_cross_embedcv2)
                        input_cross = input_cross.to(opt.device)

                        if opt.dataset == "cifar10":
                            input_cross = my_utils.my_transform(input_cross)
                        input_cross_resize_unsqueeze = torch.unsqueeze(input_cross, dim=0)
                        if i == 0:
                            inputs_cross = input_cross_resize_unsqueeze
                        else:
                            inputs_cross = torch.cat((inputs_cross, input_cross_resize_unsqueeze), 0)

                    preds_cross, tmp = netC(inputs_cross)
                    total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                    acc_cross = total_cross_correct * 100.0 / total_sample

                    info_string = (
                        "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
                            acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
                        )
                    )
                else:
                    info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc
                    )

            else:
                acc_bd = 0.0
                acc_cross = 0.0
                info_string = "Clean Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        if opt.pc:
            tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)
        else:
            tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean}, epoch)

    log_writer.writelines('Eval epoch:{} '.format(epoch + 1) + info_string + '\n')  # \n 换行符

    #svae each epoch's checkpoint
    if opt.save_all:
        print(" Saving all...")
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_cross_acc": best_cross_acc,
            "epoch_current": epoch,
            "identity_grid": identity_grid,
            "noise_grid": noise_grid,
            "watermark": opt.watermark,
        }
        ckpt_path = os.path.join(opt.all_ckpt_path,
                                 "{}_{}_{:.4f}_{:.4f}_morph.pth.tar".format(opt.dataset, epoch,acc_clean,acc_bd))
        torch.save(state_dict, ckpt_path)
        print(" Saving to {}".format(ckpt_path))

        if abs(acc_clean - 97.18) < 0.005 and abs(acc_bd - 96.90) < 0.005:
            ckpt_path = os.path.join(opt.all_ckpt_path,
                                     "fit_{}_{}_{:.4f}_{:.4f}_morph.pth.tar".format(opt.dataset, epoch, acc_clean, acc_bd))
            torch.save(state_dict, ckpt_path)
            print(" Saving to {}".format(ckpt_path))

    # Save checkpoint
    #if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc): 原始保存条件
    if (acc_bd > best_bd_acc and acc_clean > best_clean_acc) or (acc_bd > best_bd_acc and (acc_bd - best_bd_acc)/(best_clean_acc - acc_clean))>1.2 or (acc_clean > best_clean_acc  and (acc_clean - best_clean_acc)/(best_bd_acc-acc_bd) > 1.5):
    # if acc_clean > best_clean_acc and acc_bd > best_bd_acc:
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        if opt.cross_ratio:
            best_cross_acc = acc_cross
        else:
            best_cross_acc = torch.tensor([0])
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_cross_acc": best_cross_acc,
            "epoch_current": epoch,
            "identity_grid": identity_grid,
            "noise_grid": noise_grid,
            "watermark": opt.watermark,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            if opt.pc:
                results_dict = {
                    "clean_acc": best_clean_acc.item(),
                    "bd_acc": best_bd_acc.item(),
                    "cross_acc": best_cross_acc.item(),
                }
            else:
                results_dict = {
                    "clean_acc": best_clean_acc.item(),
                }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc, best_cross_acc


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode

    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.attack_mode, opt.dataset, opt.identification)

    opt.ckpt_path = os.path.join(opt.ckpt_folder,
                                 "{}_{}_{}_morph.pth.tar".format(opt.dataset, mode, opt.identification))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    opt.temps = os.path.join(opt.ckpt_folder, "temps")

    opt.all_ckpt_path = os.path.join(opt.ckpt_folder, "all_ckpt")


    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_cross_acc = state_dict["best_cross_acc"]
            epoch_current = state_dict["epoch_current"]
            identity_grid = state_dict["identity_grid"]
            noise_grid = state_dict["noise_grid"]
            opt.watermark = state_dict["watermark"] if "watermark" in state_dict else "01011"

            tf_writer = SummaryWriter(log_dir=opt.log_dir)
            log_writer = open(os.path.join(opt.ckpt_folder, "log.txt"), mode='a+')
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        # Prepare grid
        ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to(opt.device)
        )
        array1d = torch.linspace(-1, 1, steps=opt.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)
        log_writer = open(os.path.join(opt.ckpt_folder, "log.txt"), mode='a+')

    if not os.path.exists(opt.temps):
        os.makedirs(opt.temps)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if not os.path.exists(opt.all_ckpt_path) and opt.save_all:
        os.makedirs(opt.all_ckpt_path)

    wm = []
    for i in range(len(opt.watermark)):
        if opt.watermark[i] == '1':
            wm.append(True)
        else:
            wm.append(False)

    print(mytime.strftime('%Y-%m-%d %H:%M:%S', mytime.localtime(mytime.time())))
    print("Training...")
    log_writer.writelines(mytime.strftime('%Y-%m-%d %H:%M:%S', mytime.localtime(mytime.time())))
    log_writer.writelines("  Training...  \n")
    print("watermark:", wm)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt, log_writer, wm)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            noise_grid,
            identity_grid,
            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            tf_writer,
            epoch,
            opt,
            log_writer,
            wm,
        )


path_wm = "scau_32x32_logo.png" #When the dataset is celeba, path_wm = "scau_logo64x64.png"
# path_wm = "scau_logo64x64.png"
img_wm = cv2.imread(path_wm)
img_wm_cv2 = img_wm.copy()
img_wm = cv2_to_tensor(img_wm)
img_band_wm = my_utils.band_wm(a=20, f=4) #When the dataset is celeba, the shape=[64, 64]
# img_band_wm = my_utils.band_wm(a=20, f=4,shape=[64, 64])
if __name__ == "__main__":
    main()

# python train.py --dataset celeba --attack_mode all2all  --identification 'pc_0.4_cr0.25_a0.6' --cross_ratio 0.25 --pc 0.4 --n_iters 200  --target_label 0 --a 0.6 --bi_classes 3
# python train.py --dataset celeba --attack_mode all2all  --identification 'pc_0.4_cr0.25_a0.2' --cross_ratio 0.25 --pc 0.4 --n_iters 200  --target_label 0 --a 0.2 --bi_classes 3

#python train.py --dataset cifar10 --attack_mode all2one  --attack stack --identification 'pc_0.4_stack_scaulogo' --cross_ratio 0 --pc 0.4 --n_iters 200  --target_label 0
# python train.py --dataset cifar10 --attack_mode all2one  --attack blend --identification 'pc_0.4_blend_scaulogo' --cross_ratio 0 --pc 0.4 --n_iters 200  --target_label 0 --a 0.4 --watermark 10011011
#python train.py --dataset celeba --attack_mode all2one  --identification 'pc_0.4_patch' --cross_ratio 0 --pc 0.4 --n_iters 200  --target_label 0
#python train.py --dataset celeba --attack_mode all2one  --identification 'pc_0.4_cr0.25_wm8_36*20_a0' --cross_ratio 0.25 --pc 0.4 --n_iters 40 --watermark '10011011' --target_label 0 --a 0
#python train.py --dataset cifar10 --attack_mode all2one  --attack band --identification 'pc_0.4_bdnd_scaulogo' --cross_ratio 0 --pc 0.4 --n_iters 200  --target_label 0 --a 0.4
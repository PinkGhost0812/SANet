import glob
import numpy as np
import torch.nn as nn
import torch
import os
import cv2
from model.unet_model_pancreas_val import RSTN
# from model.depth_unet_model_pancreas_val import RSTN
import denseCRF
# from denseCRF import getCRF
# from model.unet_model import RSTN
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

epoch={}
epoch['S'] = 2
epoch['I'] = 6
epoch['J'] = 8

def diceCoeff(pred, gt, smooth=1, activation='none'):
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N

img_path = '/disk/sdb/lijiaming/Model/unet-whole/data/window/test/'
mask_path = '/disk/sdb/lijiaming/Model/unet-whole/data/window/test_mask/'
# img_path = '/disk/sdb/lijiaming/Model/unet-whole/data/val_512_one/'
# mask_path = '/disk/sdb/lijiaming/Model/unet-whole/data/valmask_512_one/'
state_dict_path = '/disk/sdb/lijiaming/Model/unet-whole/checkpoint/mode_J_epoch_7_best_model.pth'
temp_dict = '/disk/sdb/lijiaming/Model/pth/Dice_Loss_0.799_Without_mode_J_epoch_7_best_model.pth'

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # net = UNet(n_channels=1, n_classes=1)
    net = RSTN()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    # net.load_state_dict(torch.load(temp_dict, map_location=device))
    # net.load_state_dict(torch.load(state_dict_path, map_location=device))
    # 测试模式
    net.train()
    # 读取所有图片路径
    file_list = os.listdir(img_path)
    label_fille_list = os.listdir(mask_path)
    file_list.sort()
    label_fille_list.sort()
    tests_path = glob.glob(img_path + '*.jpg')
    label_tests_path = glob.glob(mask_path + '*.jpg')

    # file_list = os.listdir("/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/imagesTs/")
    # label_fille_list = os.listdir("/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/labelsTs/")
    # file_list.sort()
    # label_fille_list.sort()
    # tests_path = glob.glob('/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/imagesTs/*.png')
    # label_tests_path = glob.glob('/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/labelsTs/*.png')
    for i in range(len(file_list)):
        path = img_path + file_list[i]
        label_path = mask_path + label_fille_list[i]

        # path = "/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/imagesTs/" + file_list[i]
        # label_path = "/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/labelsTs/" + label_fille_list[i]
        tests_path[i] = path
        label_tests_path[i] = label_path
    # 遍历素有图片
    coarse_dice = fine_dice = new_fine_dice = new_coarse_dice = coarse_twice_dice = new_coarse_twice_dice = best_dice =  0
    # for test_path in tests_path:
    with torch.no_grad():
        for mode in ['J']:
            for e in range(epoch[mode]):
                # state_dict_path = '/disk/sdb/lijiaming/Model/unet-whole/checkpoints/checkpoint1/mode_' + mode + '_epoch_' + str(e) + '_best_model.pth'
                state_dict_path = '/disk/sdb/lijiaming/Model/unet-whole/checkpoints/checkpoint/mode_' + mode + '_epoch_55_best_model.pth'
                net.load_state_dict(torch.load(state_dict_path, map_location=device))
                coarse_dice = fine_dice = new_fine_dice = new_coarse_dice = coarse_twice_dice = new_coarse_twice_dice = best_dice =  0
                for i in range(len(tests_path)):
                    test_path = tests_path[i]
                    label_test_path = label_tests_path[i]
                    # 保存结果地址 --- 改到下边了，直接给名字上写上Dice
                    # save_res_path = test_path.split('.')[0] + '_fine.png'
                    # save_res_path2 = test_path.split('.')[0] + '_coarse.png'
                    # save_res_path3 = test_path.split('.')[0] + '_crf.png'
                    # 读取图片
                    img = cv2.imread(test_path, flags=0)
                    label = cv2.imread(label_test_path, flags=0)
                    # 转为灰度图
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    # label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
                    #计算CRF使用
                    imgcache = img
                    # 转为batch为1，通道为1，大小为512*512的数组
                    img = img.reshape(1, 1, img.shape[0], img.shape[1])
                    label = label.reshape(1, 1, label.shape[0], label.shape[1])
                    if label.max() > 1:
                        label = label / 255
                    # 转为tensor
                    img_tensor = torch.from_numpy(img)
                    label_tensor = torch.from_numpy(label)
                    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
                    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                    label_tensor = label_tensor.to(device=device, dtype=torch.float32)
                    # 预测-粗分割迭代的预测
                    # new_coarse_pred_tensor, coarse_pred, fine_pred, coarse_twice, new_coarse_twice = net(img_tensor, label_tensor, 'T', i, device = device, image_cache = imgcache)
                    
                    # 预测-朴素版
                    new_coarse_pred_tensor, coarse_pred, fine_pred, coarse_prob_twice = net(img_tensor, label_tensor, mode, i, device = device, image_cache = imgcache)

                    #细分割预测结果走一下CRF
                    new_fine_pred = denseCRF.getCRF(imgcache, fine_pred)
                    new_fine_pred_tensor = torch.from_numpy(new_fine_pred)
                    new_fine_pred_tensor = new_fine_pred_tensor.to(device=device, dtype=torch.float32)
                    # print()
                    # print("pred.shape", pred.shape)
                    #计算DSC
                    coarse_dice += diceCoeff(coarse_pred, label_tensor)
                    fine_dice += diceCoeff(fine_pred, label_tensor)
                    new_fine_dice += diceCoeff(new_fine_pred_tensor, label_tensor)
                    new_coarse_dice += diceCoeff(new_coarse_pred_tensor, label_tensor)
                    coarse_twice_dice += diceCoeff(coarse_prob_twice, label_tensor)
                    # coarse_twice_dice += diceCoeff(coarse_twice, label_tensor)
                    # new_coarse_twice_dice += diceCoeff(new_coarse_twice, label_tensor)

                    cd = diceCoeff(coarse_pred, label_tensor)
                    fd = diceCoeff(fine_pred, label_tensor)
                    nfd = diceCoeff(new_fine_pred_tensor, label_tensor)
                    nc = diceCoeff(new_coarse_pred_tensor, label_tensor)
                    cd2 = diceCoeff(coarse_prob_twice, label_tensor)
                    # ncd2 = diceCoeff(new_coarse_twice, label_tensor)

                    # best_dice += max(diceCoeff(fine_pred, label_tensor), diceCoeff(new_fine_pred_tensor, label_tensor), diceCoeff(coarse_pred, label_tensor), nc, cd2, ncd2)
                    best_dice += max(diceCoeff(fine_pred, label_tensor), diceCoeff(new_fine_pred_tensor, label_tensor), diceCoeff(coarse_pred, label_tensor), nc)
                    
                    
                    print("Pic", str(i), "coarse_dice = ", diceCoeff(coarse_pred, label_tensor), "coarse_twice_dice", diceCoeff(coarse_prob_twice, label_tensor), "fine_dice", diceCoeff(fine_pred, label_tensor), 
                    "new_coarse_dice", diceCoeff(new_coarse_pred_tensor, label_tensor), "new_fine_dice", diceCoeff(new_fine_pred_tensor, label_tensor))
                    # "coarse_twice_dice", cd2, "new_coarse_twice_dice", ncd2)

                    # 提取结果
                    coarse_pred = np.array(coarse_pred.data.cpu()[0])[0]  #这一句是原来的操作
                    fine_pred = np.array(fine_pred.data.cpu()[0])[0]  #这一句是原来的操作
                    # 处理结果
                    coarse_pred[coarse_pred >= 0.5] = 255
                    coarse_pred[coarse_pred < 0.5] = 0
                    fine_pred[fine_pred >= 0.5] = 255
                    fine_pred[fine_pred < 0.5] = 0
                    new_fine_pred[new_fine_pred == 1] = 255
                    new_fine_pred[new_fine_pred == 0] = 0
                    # 保存图片
                    save_res_path = test_path.split('.')[0] + '_' + str(cd.item()) + '_fine.png'
                    save_res_path2 = test_path.split('.')[0] + '_' + str(fd.item()) + '_coarse.png'
                    save_res_path3 = test_path.split('.')[0] + '_' + str(nfd.item()) + '_crf.png'

                    save_res_path = save_res_path.replace('test', 'valout512_fine')
                    save_res_path2 = save_res_path2.replace('test', 'valout512_coarse')
                    save_res_path3 = save_res_path3.replace('test', 'valout512_crf')
                    # save_res_path = save_res_path.replace('/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/imagesTs', '/disk/sdb/lijiaming/Model/unet-whole/Ts')
                    # save_res_path2 = save_res_path2.replace('/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/imagesTs', '/disk/sdb/lijiaming/Model/unet-whole/Ts')
                    # save_res_path3 = save_res_path3.replace('/disk/sda/tanxin/data/public_pancreas_tumor2/png_all/imagesTs', '/disk/sdb/lijiaming/Model/unet-whole/Ts')
                    cv2.imwrite(save_res_path, fine_pred)
                    cv2.imwrite(save_res_path2, coarse_pred)
                    cv2.imwrite(save_res_path3, new_fine_pred)
                print("coarse_all_dice = ", coarse_dice.item(), "fine_all_dice = ", fine_dice.item(), "crf_fine_all_dice = ", new_fine_dice.item())
                
                # print("mode " + mode + " epoch " + str(e) + " coarse_avg_dice = ", coarse_dice.item() / 580, "fine_avg_dice = ", fine_dice.item() / 580,
                # "coarse_crf_dice = ", new_coarse_dice.item() / 580, "crf_fine_dice = ", new_fine_dice.item() / 580, 
                # "coarse_twice_dice = ", coarse_twice_dice.item() / 580, "crf_coarse_twice_dice = ", new_coarse_twice_dice.item() / 580,
                # "best_avg_dice = ", best_dice.item() / 580)

                print("mode " + mode + " epoch " + str(e) + " coarse_avg_dice = ", str(coarse_dice.item() / 1755), "fine_avg_dice = ", str(fine_dice.item() / 1755),
                "crf_coarse_dice = ", str(new_coarse_dice.item() / 1755), "crf_fine_dice = ", str(new_fine_dice.item() / 1755))
                
                with open('/disk/sdb/lijiaming/Model/unet-whole/checkpoints/checkpoint4/test.txt', 'a') as f:
                    f.write("mode " + mode + " epoch " + str(e) + " coarse_avg_dice = " + str(coarse_dice.item() / 1755) + "fine_avg_dice = " + str(fine_dice.item() / 1755) +
                    "crf_coarse_dice = " + str(new_coarse_dice.item() / 1755) + "crf_fine_dice = " + str(new_fine_dice.item() / 1755) + '\n')
                    # f.write("mode " + mode + " epoch " + str(e) + " coarse_avg_dice = " + str(coarse_dice.item() / 580) + "fine_avg_dice = " + str(fine_dice.item() / 580) +
                    # "crf_coarse_dice = " + str(new_coarse_dice.item() / 580) + "crf_fine_dice = " + str(new_fine_dice.item() / 580) + "coarse_twice_dice = " + str(coarse_twice_dice.item() / 580)
                    # + "crf_coarse_twice_dice = " + str(new_coarse_twice_dice.item() / 580) + '\n')

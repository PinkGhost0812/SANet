from this import s
import cv2
import numpy as np
import torch


import torch.nn.functional as F

from .unet_parts import *


def diceCoeff(pred, gt, smooth=1e-5, activation='none'):
 
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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SANET(nn.Module):
    def __init__(self):
        super(SANET, self).__init__()
        self.margin = 20
        self.left = self.margin
        self.right = self.margin
        self.top = self.margin
        self.bottom = self.margin
        
        self.coarse_model = UNet(1, 1)

        self.saliency1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.relu_saliency1 = nn.ReLU(inplace=True)
        self.saliency2 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)

        self.fine_model = UNet(1, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, mod in self.named_children():
            if name == 'saliency1':
                nn.init.xavier_normal_(mod.weight.data)
                mod.bias.data.fill_(1)
            elif name == 'saliency2':
                mod.weight.data.zero_()
                mod.bias.data = torch.tensor([1.0])


    def forward(self, image, label=None, mode='T'):
        h = image
        h = self.coarse_model(h)
        coarse_prob = h

        h = torch.sigmoid(h)
        h = self.relu_saliency1(self.saliency1(h))
        h = self.saliency2(h)
        saliency = h

        h = self.coarse_model(saliency * image)
        coarse_prob_twice = h

        h = torch.sigmoid(h)
        h = self.relu_saliency1(self.saliency1(h))
        h = self.saliency2(h)
        saliency = h

        if mode == 'S':
            crop_img, crop_info = self.crop(label, image)
        elif mode == 'I':
            crop_img, crop_info = self.crop(label, saliency * image)
        elif mode == 'J':
            crop_img, crop_info = self.crop(coarse_prob, saliency * image, label)
        else:
            raise ValueError("wrong value of mode, should be in ['S', 'I', 'J', 'V']")

        h = crop_img
        h = self.fine_model(h)
        h = self.uncrop(crop_info, h, image)
        fine_prob = h

        return coarse_prob, coarse_prob_twice, fine_prob

    def crop(self, pred, img, label=None):
        (N, C, W, H) = pred.shape
        minA = 0
        maxA = W
        minB = 0
        maxB = H
        cache = pred.clone()
        cache[cache >= 0.5] = 1
        cache[cache < 0.5] = 0
        binary_mask = cache
        if(label is not None and binary_mask.sum().item() == 0):
            binary_mask = label
        mask = torch.zeros(size=(N, C, W, H))
        cur_mask = binary_mask[0, 0, :, :]
        # ------------------------------------
        
        # ------------------------------------
        arr = torch.nonzero(cur_mask)
        if(arr.shape[0] != 0):
            minA = arr[:, 0].min().item()
            maxA = arr[:, 0].max().item()
            minB = arr[:, 1].min().item()
            maxB = arr[:, 1].max().item()
        bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
                int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
        mask[0, 0, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
        img = img * mask.cuda()
        crop_img = img[:, :, bbox[0]: bbox[1], bbox[2]: bbox[3]]
        return crop_img, bbox


    def uncrop(self,crop_info, cropped_image, image):
        uncropped_image = torch.ones_like(image)
        uncropped_image *= (-999999999)
        bbox = crop_info
        uncropped_image[0, 0, bbox[0]: bbox[1], bbox[2]: bbox[3]] = cropped_image
        return uncropped_image


    def anti_crop(self, pred, img, label=None):
        (N, C, W, H) = pred.shape
        minA = 0
        maxA = W
        minB = 0
        maxB = H
        cache = pred.clone()
        cache[cache >= 0.5] = 1
        cache[cache < 0.5] = 0
        binary_mask = cache
        if(label is not None and binary_mask.sum().item() == 0):
            binary_mask = label
        mask = torch.zeros(size=(N, C, W, H))
        cur_mask = binary_mask[0, 0, :, :]
        # ------------------------------------
        
        # ------------------------------------
        arr = torch.nonzero(cur_mask)
        if(arr.shape[0] != 0):
            minA = arr[:, 0].min().item()
            maxA = arr[:, 0].max().item()
            minB = arr[:, 1].min().item()
            maxB = arr[:, 1].max().item()
        bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
                int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
        mask[0, 0, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1 
        m_one = torch.ones(size=(N, C, W, H))
        m_zero = m_one - mask   
        anti_box = pred * m_zero.cuda()    
        anti_box[0, 0, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
        img = img * mask.cuda()
        crop_img = img[:, :, bbox[0]: bbox[1], bbox[2]: bbox[3]]
        return anti_box, bbox


class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'
 
    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
 
    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt, activation=self.activation)

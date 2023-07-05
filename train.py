from model.unet_model import SoftDiceLoss
from model.unet_model_pancreas import SANET
from utils.dataset import ISBI_Loader
from utils.focal_loss import FocalLossV1
from utils.Active_Contour_Loss import Active_Contour_Loss
from utils.elastic_loss import EnergyLoss
from utils.levelset_loss import LevelsetLoss
from torch import optim
import torch.nn as nn
import torch
import os
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
epoch={}
epoch['S'] = 2
epoch['I'] = 6
epoch['J'] = 8
COARSE_WEIGHT = 1 / 3
COARSE_TWICE_WEIGHT = 1 / 5
FINE_WEIGHT = 2 / 5
def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
 
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


def train_net(net, device, data_path, batch_size=1, lr=0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-7, momentum=0.9)
    criterion3 = SoftDiceLoss()
    criterion = nn.BCEWithLogitsLoss()
    criterion1 = FocalLossV1()
    criterion2 = LevelsetLoss()#pred, mask, pixel_num
    best_loss = float('inf')
    for mode in ['S', 'I', 'J']:
        for e in range(epoch[mode]):
            net.train()
            for image, label in train_loader:
                optimizer.zero_grad()
                image = Variable(image.to(device=device, dtype=torch.float32))
                label = Variable(label.to(device=device, dtype=torch.float32))
                coarse_pred, fine_pred = net(image, label, mode)
                loss = COARSE_WEIGHT * criterion3(coarse_pred, label) + (1 - COARSE_WEIGHT) * criterion3(fine_pred, label)
                
                dice = diceCoeff(fine_pred, label)
                coarse_dice = diceCoeff(coarse_pred, label)
                
                print('epoch =', str(e), 'mode =', mode, 'Coarse_Loss =', criterion3(coarse_pred, label).item(),'Fine_Loss = ', criterion3(fine_pred, label).item(),
                 'Coarse_Pred_Dice =', coarse_dice.item(), 'Fine_Pred_Dice =',dice.item())

                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), '')
                loss.backward()
                optimizer.step()
            
            
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SANET()
    net.to(device=device)
    data_path = ""
    train_net(net, device, data_path)

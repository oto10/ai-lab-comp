"""
このファイルは改変しないで下さい。
"""


#ライブラリのインポート
import torch as torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchsummary import summary
from pylab import rcParams

def make_dataset():
    transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(), 
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./dataset/',
                                             train=True, #訓練データをロード
                                             transform=transform,#上で定義した変換を適用
                                             download=True)

    return train_dataset

def eval(net):
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'#GPUの定義

    test_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(), 
    ])

    dataset = torchvision.datasets.CIFAR10(root='./dataset/',
                                             train=False, #検証データをロード
                                             transform=test_transform,
                                             download=True)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=32, 
                                           shuffle=False,
                                         num_workers=4)

    net.eval() #val *2
    val_loss = 0
    val_acc = 0
    with torch.no_grad():#自動微分停止 *2
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(data_loader.dataset)
        avg_val_acc = val_acc / len(data_loader.dataset)


    
    return avg_val_acc
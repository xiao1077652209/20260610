import os
import math
import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import json
import pandas as pd
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import time 
from lm import LMLoss
#from Test11_efficientnetV2.model import efficientnetv2_l as create_model
#from efficientnetv2.effnetv2 import effnetv2_s as create_model
#from mobilenetv3.mobilenetv3 import mobilenetv3_large as create_model
from MVmodel import mobile_vit_small as create_model
#from Test8_densenet.model import densenet201 ,load_state_dict
#from Test10_regnet.model import create_regnet as create_model
#from Test7_shufflenet.model import shufflenet_v2_x1_5 as creatne_model
from utils import  read_split_data,plot_data_loader_image,evaluate,train_one_epoch
from options import Options
from torch.utils.data import Dataset
from candock_master.creatnet import CreatNet
import candock_master.util as util
from torchsummary import summary

#------------------------------------nirs-----------------------------------#
from datetime import datetime
from functools import partial
from PIL import Image
from Preprocessing.Preprocessing import *
import numpy as np
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from torchvision.models import resnet
from tqdm import tqdm
import json
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt
from options import Options
import candock_master.transformer as transformer
import candock_master.statistics as statistics
import candock_master.heatmap as heatmap


def image(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_spectra, train_labels, val_spectra, val_labels = read_split_data(args.data_path)
    #train_spectra, train_labels, _, _ = read_split_data(args.train_data_path)
    #val_spectra, val_labels,_,_ = read_split_data(args.val_data_path)
    
    '''
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    '''
    # 实例化训练数据集
    train_dataset = MyDataSet(spectra=train_spectra,
                              labels=train_labels)

    # 实例化验证数据集
    val_dataset = MyDataSet(spectra=val_spectra,
                            labels=val_labels)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    use_cuda = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=use_cuda,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=use_cuda,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
   
    
    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    
    
    '''
    # 如果存在预训练权重则载入
    model = densenet201(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            load_state_dict(model, args.weights)
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    '''
    
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    #pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    return model,device,train_loader,val_loader
    
#----------------------------------nirs-------------------------------

class MyDataSet(Dataset):
    def __init__(self, spectra, labels):
        self.spectra = spectra
        self.labels = labels

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        label = self.labels[idx]
        spectrum = np.expand_dims(spectrum, axis=0)  # 增加一个通道维度
        spectrum = Preprocessing("SNV",spectrum)
        spectrum = Preprocessing("D1",spectrum)
        spectrum = Preprocessing("SG",spectrum)
        #spectrum = spectrum[:,3151:]
        spectrum = np.squeeze(spectrum)  # 移除增加的维度

        return spectrum, label

    @staticmethod
    def collate_fn(batch):
        #images, spectra, labels = tuple(zip(*batch))
        #images = torch.stack(images, dim=0)
        spectra, labels = tuple(zip(*batch))
        spectra=np.array(spectra)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return spectra, labels


    
def adjust_learning_rate(optimizer, epoch, opt):

    """Decay the learning rate based on schedule"""

    lr = opt.nirlr

    if opt.cos:  # cosine lr schedule

        lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.nirepochs))
        #print("__________cos___________")
	
    else:  # stepwise lr schedule

        for milestone in opt.schedule:

            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


#mlp
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        return x


#mlp
class MLPW(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPW, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.Dropout=nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

 
       
        

if __name__ == '__main__':
    opt = Options().getparse()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model,device,train_loader,val_loader = image(opt)
    
    nirnum_classes=opt.nirnum_classes
    net=CreatNet(opt.model_name)
    util.show_paramsnumber(net)
    log_dir = Path(__file__).parent / "runs"  # 与脚本同级的runs目录
    log_dir.mkdir(exist_ok=True, parents=True)

    tb_writer = SummaryWriter(log_dir=str(log_dir))
    print(net)
    if not opt.no_cuda:
        print("-------------------------------")
        net.cuda()
    if opt.pretrained:
        net.load_state_dict(torch.load('./checkpoints/pretrained/'+opt.dataset_name+'/'+opt.model_name+'.pth'))
    if opt.continue_train:
        net.load_state_dict(torch.load('./checkpoints/last.pth'))
    if not opt.no_cudnn:
        torch.backends.cudnn.benchmark = True
        
#-----------------------------------change-----------------------------------
  
    #device = torch.device('cpu')
    #net.load_state_dict(torch.load('./lstmnet-180.pth', map_location=device))
    #net.load_state_dict(torch.load('./lstmnet-180.pth'))
    # 初始化alpha和beta为可学习参数
    alpha = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
    beta = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
    gamma = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
    eta = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
    # 在 loss 计算时约束它们
    gamma_clamped = torch.clamp(gamma, min=0)
    eta_clamped = torch.clamp(eta, min=0)
    
    mlp_model = MLP(input_dim=64, hidden_dim=64, output_dim=8).cuda()
    w_model = MLPW(input_dim=256, hidden_dim=128, output_dim=64).cuda()
    
    pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD((list.net.parameters())+pg, lr=opt.lr, momentum=0.9, weight_decay=1E-4)
    optimizer = torch.optim.Adam(list(net.parameters())+pg+list(mlp_model.parameters())+list(w_model.parameters())+[alpha,beta,gamma,eta],lr=opt.lr,  weight_decay=1E-4
    )
    
    criterion = nn.CrossEntropyLoss()
    #criterion = FocalLoss(gamma=2)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
#----------------------------------------train-------------------------------------#
    print('begin to train ...')
    #final_confusion_mat = np.zeros((num_classes,num_classes), dtype=int)
    plot_result={'train':[1.],'test':[1.]}
    #confusion_mats = []
    all_acc=[]
    for epoch in range(opt.epochs):
    #(model,net, optimizer, imgdata_loader,nirdata_loader, device, epoch,opt):
        start_time = time.time()
        
        train_loss, train_acc  = train_one_epoch(model=model,net=net ,mlp_model=mlp_model,w_model=w_model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch,opt=opt,alpha=alpha,beta=beta,gamma=gamma,eta=eta)
        
        end_time = time.time()
        epoch_train_duration = end_time - start_time
        fps_train = len(train_loader.dataset) / epoch_train_duration  # Calculate FPS for training
        print(f"Epoch {epoch} training time: {epoch_train_duration:.2f}s, FPS: {fps_train:.2f}")

        scheduler.step()

        # validate
        val_start_time = time.time()
        val_loss, val_acc = evaluate(model=model,net=net ,mlp_model=mlp_model,w_model=w_model, data_loader=val_loader, device=device, epoch=epoch,alpha=alpha,beta=beta,all_acc=all_acc,gamma=gamma,eta=eta)
        val_end_time = time.time()
        epoch_val_duration = val_end_time - val_start_time
        fps_val = len(val_loader.dataset) / epoch_val_duration  # FPS for validation
        sps_val = len(val_loader.dataset) * val_loader.batch_size / epoch_val_duration  # SPS for validation
        print(f"Epoch {epoch} validation time: {epoch_val_duration:.2f}s, FPS: {fps_val:.2f}, SPS: {sps_val:.2f}")

        # TensorBoard logging
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        torch.save(net.state_dict(), "./weights/net-{}.pth".format(epoch))
        # 假设模型是 model
        #summary(net, input_size=(1, 1554))  # 这里的 input_size 根据你的输入形状设置
        #for name, param in net.named_parameters(): print(name, param.numel())
        #total_params = sum(param.numel() for param in net.parameters())
        #print(f"Total Parameters: {total_params}")

    
    

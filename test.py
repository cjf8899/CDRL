import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/workspace/NAS_MOUNT/", help="root path")
parser.add_argument("--dataset_name", type=str, default="LEVIR-CD", help="name of the dataset")
parser.add_argument("--save_name", type=str, default="levir", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument('--save_visual', action='store_true', help='save pixel visualization map')
opt = parser.parse_args()
print(opt)

os.makedirs('pixel_img/'+opt.save_name, exist_ok=True)
os.makedirs('gener_img/'+opt.save_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

lambda_pixel = 100

patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

generator = GeneratorUNet_CBAM(in_channels=3)
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()



generator.load_state_dict(torch.load("saved_models/"+opt.save_name+"/generator_9.pth"))
discriminator.load_state_dict(torch.load("saved_models/"+opt.save_name+"/discriminator_9.pth"))


transforms_ = A.Compose([
    A.Resize(opt.img_height, opt.img_width),
    A.Normalize(), 
    ToTensorV2()
])

val_dataloader = DataLoader(
    CDRL_Dataset_test(opt.root_path, dataset=opt.dataset_name, transforms=transforms_),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    
def pixel_visual(gener_output_, A_ori_, name):
    gener_output = gener_output_.cpu().clone().detach().squeeze()
    A_ori = A_ori_.cpu().clone().detach().squeeze()
    
    pixel_loss = to_pil_image(torch.abs(gener_output-A_ori))
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()])
    pixel_loss = trans(pixel_loss)

    thre_num= 0.7
    threshold = nn.Threshold(thre_num, 0.)
    pixel_loss = threshold(pixel_loss)
    save_image(pixel_loss, 'pixel_img/'+opt.save_name+'/'+str(name[0]))
    save_image(gener_output.flip(-3), 'gener_img/'+opt.save_name+'/'+str(name[0]), normalize=True)


prev_time = time.time()

loss_G_total = 0

generator.eval()
discriminator.eval()

with torch.no_grad():
    for i, batch in enumerate(val_dataloader):

        img_A = Variable(batch["A"].type(Tensor))
        img_B = Variable(batch["B"].type(Tensor))
        name = batch["NAME"]

        valid = Variable(Tensor(np.ones((img_A.size(0), *patch))), requires_grad=False)

        # ---------------------
        # Generator loss
        # ---------------------
        
        img_A = img_A.cuda()
        img_B = img_B.cuda()
        
        gener_output = generator(img_A,img_B)
        gener_output_pred = discriminator(gener_output, img_A)
        
        if opt.save_visual:
            pixel_visual(gener_output, img_B, name)
            
        loss_GAN = criterion_GAN(gener_output_pred, valid)  
        loss_pixel = criterion_pixelwise(gener_output, img_B)
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        # --------------
        #  Log Progress
        # --------------
        
        print('-----------------------------------------------------------------------------')
        print('name : ', name[0])
        print('loss_G : ', round(loss_G.item(),4))
        loss_G_total += loss_G
        
    print('----------------------------total------------------------------')
    print('loss_G_total : ', round((loss_G_total/len(val_dataloader)).item(),4))
    

#-*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from pretrained_models_pytorch import pretrainedmodels

from utils import *
from models import *
from torch.nn import DataParallel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')

parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ')

parser.add_argument('--train_size', type=int, default=2000, help='Number of training images')
parser.add_argument('--test_size', type=int, default=2000, help='Number of test images')

parser.add_argument('--image_size', type=int, default=299, help='the height / width of the input image to network')

parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')

parser.add_argument('--netClassifier', default='inceptionv3', help="The target classifier")

parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

target = opt.target
conf_target = opt.conf_target
max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
train_size = opt.train_size
test_size = opt.test_size
plot_all = opt.plot_all 

assert train_size + test_size <= 13233, "Traing set size + Test set size > Total dataset size"

# print("=> creating model ")
# netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')
# if opt.cuda:
#     netClassifier.cuda()


### Newly added ###

netClassifier = resnet_face18(False)
# netClassifier = DataParallel(netClassifier)

model_dict = netClassifier.state_dict()
pretrained_dict = torch.load('C:/Users/NMAIL/Documents/강의/고급정보보호/Term project/code/arcface-pytorch-master/checkpoints/resnet18_110.pth', map_location = device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
netClassifier.load_state_dict(model_dict)
netClassifier.to(device)

### Upto here ###


print('==> Preparing data..')
# normalize = transforms.Normalize(mean=netClassifier.mean,
#                                  std=netClassifier.std)
normalize = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
idx = np.arange(13233)
np.random.shuffle(idx)
training_idx = idx[:train_size]
test_idx = idx[train_size:test_size]

# train_loader = torch.utils.data.DataLoader(
#     dset.ImageFolder('./imagenetdata/val', transforms.Compose([
#         transforms.Scale(round(max(netClassifier.input_size)*1.050)),
#         transforms.CenterCrop(max(netClassifier.input_size)),
#         transforms.ToTensor(),
#         ToSpaceBGR(netClassifier.input_space=='BGR'),
#         ToRange255(max(netClassifier.input_range)==255),
#         normalize,
#     ])),
#     batch_size=1, shuffle=False, sampler=SubsetRandomSampler(training_idx),
#     num_workers=opt.workers, pin_memory=True)

train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./imagenetdata/val', transforms.Compose([
        transforms.Scale(round(max([3, 128, 128])*1.050)),
        transforms.CenterCrop(max([3, 128, 128])),
        transforms.ToTensor(),
        ToSpaceBGR('RGB'=='BGR'),
        ToRange255(max([0, 1])==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(training_idx),
    num_workers=opt.workers, pin_memory=True)

# test_loader = torch.utils.data.DataLoader(
#     dset.ImageFolder('./imagenetdata/val', transforms.Compose([
#         transforms.Scale(round(max(netClassifier.input_size)*1.050)),
#         transforms.CenterCrop(max(netClassifier.input_size)),
#         transforms.ToTensor(),
#         ToSpaceBGR(netClassifier.input_space=='BGR'),
#         ToRange255(max(netClassifier.input_range)==255),
#         normalize,
#     ])),
#     batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
#     num_workers=opt.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./imagenetdata/val', transforms.Compose([
        transforms.Scale(round(max([3, 128, 128])*1.050)),
        transforms.CenterCrop(max([3, 128, 128])),
        transforms.ToTensor(),
        ToSpaceBGR('RGB'=='BGR'),
        ToRange255(max([0,1])==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

# min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
# min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
# mean, std = np.array(netClassifier.mean), np.array(netClassifier.std) 
# min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

min_in, max_in = 0, 1
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]) 
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def train(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    recover_time = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        
        print(batch_idx)
        ### added for arcface data input
        data = data.numpy()

        r = data[:, 0, :, :].reshape((128, 128))
        g = data[:, 1, :, :].reshape((128, 128))
        b = data[:, 2, :, :].reshape((128, 128))

        data = r*0.2989 + g*0.5870 + b*0.1140
        data = np.dstack((data, np.fliplr(data)))
        data = data.transpose((2, 0, 1))
        data = data[:, np.newaxis, :, :]
        data = data.astype(np.float32, copy=False)
        data -= 127.5
        data /= 127.5

        data = torch.from_numpy(data)
        ###

        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)
        # print("{} .... {}".format(prediction.data.max(1)[1][0], labels.data[0]))
 
        print("prediction")
        print(prediction)
        print(prediction.data.max(1))
        print(prediction.data.max(1)[1][0])
        print(prediction.shape)
        print(labels)

        # assert False

        # only computer adversarial examples on examples that are originally classified correctly        
        # if prediction.data.max(1)[1][0] != labels.data[0]:
        #     continue

        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)
 
        adv_x, mask, patch = attack(data, patch, mask)
        
        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label == target:
            success += 1
      
            if plot_all == 1: 
                # plot source image
                vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)
                
                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial.png" %(opt.outf, batch_idx, adv_label), normalize=True)
 
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.3f}".format(success/total))

    return patch

def test(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
      
        total += 1 
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)
 
        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
        
        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label == target:
            success += 1
       
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.3f}".format(success/total))

def attack(x, patch, mask):
    netClassifier.eval()

    # x_out = F.softmax(netClassifier(x))
    cur_vec = netClassifier(x)
    new_vec = cur_vec
    # target_prob = x_out.data[0][target]

    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    
    count = 0 
   
    # while conf_target > target_prob:
    while cosin_metric(cur_vec, new_vec) > 0.2:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)
        # adv_out = F.log_softmax(netClassifier(adv_x))
        new_vec = netClassifier(adv_x)
       
        # adv_out_probs, adv_out_labels = adv_out.max(1)
        
        # Loss = -adv_out[0][target]
        Loss = -1 * cosin_metric(cur_vec, new_vec)
        Loss.backward()
     
        adv_grad = adv_x.grad.clone()
        
        adv_x.grad.data.zero_()
       
        patch -= adv_grad 
        
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
 
        new_vec = netClassifier(adv_x)
        # out = F.softmax(netClassifier(adv_x))
        # target_prob = out.data[0][target]
        #y_argmax_prob = out.data.max(1)[0][0]
        
        #print(count, conf_target, target_prob, y_argmax_prob)  

        if count >= opt.max_count:
            break


    return adv_x, mask, patch 


if __name__ == '__main__':
    if patch_type == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size) 
    elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size) 
    else:
        sys.exit("Please choose a square or circle patch")
    
    for epoch in range(1, opt.epochs + 1):
        patch = train(epoch, patch, patch_shape)
        test(epoch, patch, patch_shape)
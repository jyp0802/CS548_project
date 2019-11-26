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

from PIL import Image
import cv2
from imutils import face_utils
import imutils
import dlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')

# parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example')
parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ')

parser.add_argument('--train_size', type=int, default=10, help='Number of training images')
parser.add_argument('--test_size', type=int, default=2000, help='Number of test images')

# parser.add_argument('--image_size', type=int, default=299, help='the height / width of the input image to network')
parser.add_argument('--image_size', type=int, default=128, help='the height / width of the input image to network')

parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')

parser.add_argument('--netClassifier', default='inceptionv3', help="The target classifier")

parser.add_argument('--shape_predictor', default='shape_predictor_68_face_landmarks.dat',help="path to facial landmark predictor")

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
predic = opt.shape_predictor

assert train_size + test_size <= 13233, "Traing set size + Test set size > Total dataset size"

### Face detection model load ###
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predic)


### Arcface model load ###

def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

netClassifier = resnet_face18(False)
netClassifier = DataParallel(netClassifier)
load_model(netClassifier, './checkpoints/resnet18_110.pth')
netClassifier.load_state_dict(torch.load('./checkpoints/resnet18_110.pth', map_location = device))
netClassifier.to(device)


print('==> Preparing data..')
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
        transforms.Scale(round(max([3, 128, 128]))),
        transforms.ToTensor(),
        ToSpaceBGR('RGB'=='BGR'),
        ToRange255(False),
        # ToRange255(),
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
        # ToSpaceBGR('RGB'=='BGR'),
        ToSpaceBGR('RGB'=='RGB'),
        # ToRange255(max([0,1])==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = 0, 1
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]) 
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

### Finding patch centre and patch size. orig_image: tensor
def find_cheek(orig_image, orig_image_size):

    ratio = orig_image_size / 500
    image = orig_image.clone()
    image = F.interpolate(image, size=500).cpu().numpy()[0]

    print("image")
    print(image)

    r = image[0]
    g = image[1]
    b = image[2]
    rgb = []
    for i in range(500):
        tmp = []
        for j in range(500):
            tmp.append([r[i][j]*255, g[i][j]*255, b[i][j]*255])
        rgb.append(tmp)

    rgb = np.ndarray.round(np.array(rgb)).astype(np.uint8)
    # print('rgb')
    # print(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
 
    target = [ 'mouth' , 'right_eye' , 'nose']

    mouth_loc =(0,0)
    eye_loc = (0,0)
    nose_loc = (0,0)
    patch_center = (0,0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            clone = rgb.copy()


            if name in target:
                temp = (0,0)
                point_num = len(shape[i:j])

                for (x, y) in shape[i:j]:
                    
                    temp = (temp[0]+ x, temp[1]+y)
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                    if name is 'mouth':
                        mouth_loc = (int(temp[0]/point_num), int(temp[1]/point_num))

                    elif name is 'right_eye':
                        eye_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))

                    elif name is 'nose':
                        nose_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))
            else:
                pass

            x_dif = nose_loc[0] - eye_loc[0]
            y_dif = nose_loc[1] - eye_loc[1]

            patch_center = (round((eye_loc[0]-x_dif*0.4)*ratio) , round((nose_loc[1] - int(y_dif*0.1))*ratio))

    return patch_center, round(y_dif*0.6*ratio)

### Transforming image for arcface
def arcface_transform(x):

    x = x.mul_(255)
    r = x[:, 0, :, :].reshape((128, 128))
    g = x[:, 1, :, :].reshape((128, 128))
    b = x[:, 2, :, :].reshape((128, 128))

    x_d = (r*0.2989 + g*0.5870 + b*0.1140)
    x_d = torch.stack([x_d, torch.flip(x_d, [1])], dim = 2)
    x_d = x_d.permute(2, 0, 1)
    x_d = x_d.unsqueeze(1)
    x_d -= 127.5
    x_d /=127.5

    return x_d

### Load image for arcface
def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None

    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image /= 250
    image -= 0.5
    image *= 2

    return image

### Cosine similarity metric
def cosin_metric(x1, x2):
    d1 = x1.clone().detach().numpy()
    d2 = x2.clone().detach().numpy()

    d1 = np.reshape(d1, 1024)
    d2 = np.reshape(d2, 1024)

    return np.array([np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))])


def train(epoch):

    netClassifier.eval()
    success = 0
    total = 0
    recover_time = 0
    for batch_idx, (data, labels) in enumerate(train_loader):

        # vutils.save_image(data.data, 'patch_data.png')
        # print("data")
        print("data")
        print(data.clone().data)

        patch_loc, r_length = find_cheek(data, image_size)

        if patch_type == 'circle':
            patch, patch_shape = init_patch_circle(image_size, patch_size, r_length)
        elif patch_type == 'square':
            patch, patch_shape = init_patch_square(image_size, patch_size) 
        else:
            sys.exit("Please choose a square or circle patch")

        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size, patch_loc)
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)

        mask_rgb = torch.mul(mask, data.clone())

        print("after")
        print(data.data)
 
        adv_x, mask, patch = attack(data, patch, mask, mask_rgb)
     
        if plot_all == 1: 
            # plot source image
            vutils.save_image(data.data, "./%s/%d_original.png" %(opt.outf, batch_idx), normalize=True)
            
            # plot adversarial image
            vutils.save_image(adv_x.data, "./%s/%d_adversarial.png" %(opt.outf, batch_idx))
 
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
        
        adv_label = netClassifier(adv_x)
        ori_label = netClassifier(x)
        
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


def attack(x, patch, mask, orig_rgb):

    netClassifier.eval()

    # print("inside")
    # print(x.data)

    another1 = load_image("C:/Users/NMAIL/Desktop/Kim_Ryong-sung_0001.jpg")
    another1 = torch.from_numpy(another1).to(device)
    vutils.save_image(another1.clone().detach().data, "0.png")
    print(another1)

    another1_vec = netClassifier(another1)

    another2 = load_image("C:/Users/NMAIL/Desktop/Kim_Ryong-sung_0007.jpg")
    print(another2)
    another2 = torch.from_numpy(another2).to(device)
    vutils.save_image(another2.clone().detach().data, "1.png")
    print(another2)

    another2_vec = netClassifier(another2)

    print(cosin_metric(another1_vec.cpu(), another2_vec.cpu()))

    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch) ### putting patch to image
    vutils.save_image(adv_x.data, 'patch_data.png')

    x_d = arcface_transform(x.clone()) ### image transformed for arcface

    cur_vec = netClassifier(x_d)
    new_vec = cur_vec

    count = 0

    cur_vec = cur_vec.cpu()
    new_vec = new_vec.cpu()

    # print(cosin_metric(another1_vec.cpu(), cur_vec))
    # exit()

    cosin_sim = 1

    while cosin_sim > 0.2:
        count += 1

        # adv_x = adv_x.mul_(255)

        adv_x = adv_x.requires_grad_(True)

        ### Transforming image for arcface (RGB -> grayscale)
        
        r = adv_x[:, 0, :, :].reshape((128, 128))
        g = adv_x[:, 1, :, :].reshape((128, 128))
        b = adv_x[:, 2, :, :].reshape((128, 128))

        x_d = (r*0.2989 + g*0.5870 + b*0.1140)
        x_d = torch.stack([x_d, torch.flip(x_d, [1])], dim = 2)
        x_d = x_d.permute(2, 0, 1)
        x_d = x_d.unsqueeze(1)
        # x_d -= 127.5
        # x_d /=127.5
        x_d -= 0.5
        x_d *= 2

        print(x_d[0][0][0])

        print

        new_vec = netClassifier(x_d).cpu()

        ### Loss function.
        ### 1) Naturalness loss should be added
        ### 2) Gradient vanishing problem should be addressed

        ### Naturalness regularization. 
        patch_rgb = torch.mul(mask, patch)
        natural_regul = F.l1_loss(orig_rgb.view(1, -1), patch_rgb.view(1, -1))
        # Loss = -adv_out[0][target]
        # Loss = Variable(torch.from_numpy(-1 * cosin_metric(cur_vec, new_vec)), requires_grad = True)
        # Loss = F.cosine_embedding_loss(new_vec.view(1, 1024), cur_vec.view(1, 1024), torch.tensor([1 for i in range(1024)]))
        # Loss = F.cosine_similarity(new_vec.view(1, 1024), cur_vec.view(1, 1024)) + natural_regul
        Loss = 1 / F.l1_loss(new_vec.view(1, 1024), cur_vec.view(1, 1024))
        # Loss = 10000 / F.l1_loss(new_vec.view(1, 1024), another_vec.view(1, 1024))
        Loss.backward()

        adv_x_grad = adv_x.grad.clone()
        
        adv_x.grad.data.zero_()

        # if count % 100 == 0:
        #     print(adv_x_grad)

        patch -= adv_x_grad

        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)

        r = r.detach()
        g = g.detach()
        b = b.detach()
        x_d = x_d.detach()
        new_vec = new_vec.detach()
        cur_vec = cur_vec.detach()
        adv_x = adv_x.detach()
        # another_vec = another_vec.detach()

        cosin_sim = cosin_metric(cur_vec, new_vec)
        # cosin_sim = cosin_metric(another_vec, new_vec)
        print(cosin_sim)

        vutils.save_image(adv_x.clone().detach().data, './training images/' + str(count) + ".png")


        # if count % 100 == 0:
        #     tmp = adv_x.clone().detach()
        #     vutils.save_image(tmp.data, './training images/' + str(count) + ".png")

        # if count >= opt.max_count:
        #     break

    # print("Total iteration: ")
    # print(count)
    exit()

    return adv_x, mask, patch 


if __name__ == '__main__':
    # if patch_type == 'circle':
    #     patch, patch_shape = init_patch_circle(image_size, patch_size)
    # elif patch_type == 'square':
    #     patch, patch_shape = init_patch_square(image_size, patch_size) 
    # else:
    #     sys.exit("Please choose a square or circle patch")
    
    for epoch in range(1, opt.epochs + 1):
        patch = train(epoch)
        # test(epoch, patch, patch_shape)
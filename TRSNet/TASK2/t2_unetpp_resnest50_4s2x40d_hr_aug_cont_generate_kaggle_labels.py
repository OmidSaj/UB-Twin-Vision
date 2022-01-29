## TRS-Net for ICSHM2021
## By Kareem Eltouny - University at Buffalo
## 01/28/2022

import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
import torchmetrics
from tqdm import tqdm
import csv
from Utils import *

from transformations import (
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from customdatasets import SegmentationDataSet2


if torch.cuda.is_available():
    print("CUDA device detected.")
    device="cuda:0"
    print(f"Using {device} device.")
else:
    print("No CUDA device is detected, using CPU instead.")
    device="cpu"


DATA_DIR = '../../../Dataset'
SPLIT_DIR = '../split_dictionaries'
LABEL_DIR = "ds"
model_name = "t2_unetpp_resnest50d_4s2x40d_aug"
hr_name = 't2_unetpp_resnest50d_4s2x40d_hr_aug_cont5'
hr_save = 't2_unetpp_resnest50d_4s2x40d_hr_aug_cont4'
lr_name = 't2_unetpp_resnest50d_4s2x40d_lr_aug'
results_dir = "results"
weights_dir = "weights"

# make_di_path(model_name)
make_di_path(f'{results_dir}')
make_di_path(f'{weights_dir}')
make_di_path(f'{weights_dir}/{hr_name}')
make_di_path(f'{results_dir}/{hr_name}')
make_di_path(f'{results_dir}/{hr_name}/figures')

BATCH_TRAIN = 4
BATCH_EVAL = 4
disable_tqdm = False
cache_dataset = True

N_CLASSES = 5


ENCODER = 'timm-resnest50d_4s2x40d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DEVICE = device

# create segmentation model with pretrained encoder
bottleneck_model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=N_CLASSES,
    activation=ACTIVATION,
)


# Add padding for internal segmentation model

pad = 18


class padded_model(nn.Module):
    def __init__(self, model, pad: int, ):
        super().__init__()
        self.model = model
        self.pad = pad

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, self.pad))
        x = self.model(x)
        x = x[:, :, :-self.pad]
        return x


bottleneck_model = padded_model(bottleneck_model, pad)



class UCN(nn.Module):
    def __init__(self, upscale_factor, num_classes):
        super(UCN, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, num_classes * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

class DCN(nn.Module):
    def __init__(self, downscale_factor, num_channels):
        super(DCN, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv1 = nn.Conv2d(num_channels * (downscale_factor ** 2), 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, num_channels, (5, 5), (1, 1), (2, 2))


    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.conv3(x)
        return x


ss_model = UCN(upscale_factor=4, num_classes=N_CLASSES)
ds_model = DCN(downscale_factor=4, num_channels=3)

class Highres_model(nn.Module):
    def __init__(self, model, ss_model, ds_model):
        super().__init__()
        self.model = model
        self.ss_model = ss_model
        self.ds_model = ds_model


    def forward(self, x):
        x = self.ds_model(x)
        x = self.model(x)
        x = self.ss_model(x)
        return x

model = Highres_model(bottleneck_model, ss_model, ds_model)


best_model = model
checkpoint = torch.load(f'{weights_dir}/{hr_save}/{model_name}_loss_checkpoint.pt', map_location=device)
best_model.load_state_dict(checkpoint)


best_model.to(device)

########################################################################
# generating test data

from transformations import (
    ComposeSingle,
    FunctionWrapperSingle,
    normalize_01,
)
from customdatasets import SegmentationDataSet2Test


make_di_path(f'{results_dir}/{hr_name}/label')
make_di_path(f'{results_dir}/{hr_name}/label/{LABEL_DIR}')

def get_fpath(test_csv: str, data_dir: str, label_dir: str):

    """ test_csv: str --> path to test.csv"""

    with open(test_csv, newline='') as f:
        reader = csv.reader(f)
        test_list = list(reader)

    input = []
    #target = []

    for i_obs in test_list:
        fname = i_obs[0]
        input.append(data_dir+'/image/'+fname)
        #target.append(data_dir+'/label/'+label_dir+'/'+fname)

    return input#, target


inputs_test = get_fpath("test.csv", DATA_DIR, LABEL_DIR)


# training transformations
transforms = ComposeSingle(
    [
        FunctionWrapperSingle(
            np.moveaxis, source=-1, destination=0
        ),
        FunctionWrapperSingle(normalize_01),
    ]
)

# size: (width, height) for PIL library
new_size = (480,270)

# dataset test
dataset_test = SegmentationDataSet2Test(
    inputs=inputs_test,  n_classes=N_CLASSES, transform=transforms, use_cache=False, resize=False,
)

# preprocess function
def preprocess(img: torch.tensor):
    img = torch.unsqueeze(img, dim=0)
    return img

# postprocess function
def postprocess(mask: torch.tensor):
    mask = mask.softmax(dim=1)
    mask = mask.argmax(dim=1)  # perform argmax to generate 1 channel
    mask = torch.squeeze(mask)  # remove batch dim and channel dim -> [H, W]
    return mask

def predict(
    img,
    model,
    preprocess,
    postprocess,
):

    model.eval()
    img = preprocess(img)  # preprocess image
    with torch.no_grad():
        mask = model(img)  # send through model/network
    result = postprocess(mask)  # postprocess outputs

    return result


def get_obs(index, dataset, model, device):

    img = dataset[index]
    img = img.to(device)

    pr = predict(img, model, preprocess, postprocess)

    img = img.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    pr = pr.cpu().numpy()

    return img, pr



from PIL import Image
import os

for idx in tqdm(range(len(dataset_test)), desc='predicting'):
    # _, pr = get_obs(idx, dataset_test, best_model, device)
    _, pr = get_obs(idx, dataset_test, best_model, device)
    # print(f'ID: {idx}')
    # plt.figure(figsize=(16, 9))
    # plt.figure()
    # plt.xticks([])
    # plt.yticks([])
    # # plt.title(f' Test - Components {idx}')
    # plt.imshow(pr, cmap='tab20c', vmin=0, vmax=N_CLASSES-1)
    # plt.savefig(f'{results_dir}/{hr_name}/figures/test_{idx}.png', dpi=600)
    # plt.close()
    
    mask_i = Image.fromarray(pr.astype(np.uint8))
    mask_i.save(f'{results_dir}/{hr_name}/label/{LABEL_DIR}/{os.path.basename(dataset_test.inputs[idx])}', format="png")


from shutil import copy2
copy2('project2_mask2kaggle.py', f'{results_dir}/{hr_name}')
copy2('test.csv', f'{results_dir}/{hr_name}')

os.chdir(f'{results_dir}/{hr_name}')


os.system('python project2_mask2kaggle.py')

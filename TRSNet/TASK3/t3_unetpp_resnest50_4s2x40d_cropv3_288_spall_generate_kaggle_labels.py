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
from torchmetrics.functional import confusion_matrix
from tqdm import tqdm
import albumentations as A
from DataAugmentationt3 import CreateAugmenter
import csv
from Utils import *
import smp_edited.losses.focal
import smp_edited.losses.metrics

from transformations import (
    ComposeDouble,
    FunctionWrapperDouble,
    create_dense_target,
    normalize_01,
)

from customdatasets import DataSetTask3Patch, DataSetTask3CropRbSP
from torch.utils.data import DataLoader



if torch.cuda.is_available():
    print("CUDA device detected.")
    device="cuda:0"
    print(f" Using {device} device.")

else:
    print("No CUDA device is detected, using CPU instead.")
    device="cpu"

DATA_DIR = '../../../Dataset'
SPLIT_DIR = '../split_dictionaries'
results_dir = "results"
weights_dir = "weights"
model_name = "t3_unetpp_cropv3_aug_spall"
hr_name = 'cropv3_spall_cont2'
hr_save = 'cropv3_spall_cont2'
# LABEL_DIR = ["spall", "crack", "rebar"]

LABEL_DIR = ["spall"]

BATCH_TRAIN = 4
BATCH_EVAL = 4

N_CLASSES = len(LABEL_DIR)

# make_di_path(model_name)
make_di_path(f'{results_dir}')
make_di_path(f'{weights_dir}')
make_di_path(f'{weights_dir}/{hr_name}')
make_di_path(f'{results_dir}/{hr_name}')
make_di_path(f'{results_dir}/{hr_name}/figures')


ENCODER = 'timm-resnest50d_4s2x40d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = device

# create segmentation model with pretrained encoder


model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=N_CLASSES,
    activation=ACTIVATION,
    # decoder_use_batchnorm=False,
)

pad = 9 # each side for 288


# pad two ways
class padded_model(nn.Module):
    def __init__(self, model, pad: int, ):
        super().__init__()
        self.model = model
        self.pad = pad

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad, self.pad))
        x = self.model(x)
        x = x[:, :, self.pad:-self.pad] #, self.pad:-self.pad]
        # x = x.reshape(-1, x.shape[-2], x.shape[-1])
        return x
model = padded_model(model, pad)



## Get model summary

filePath = f'{results_dir}/{hr_name}/summary.txt'
model_stats = summary(model, (1, 3, 270, 480), depth=6,
                      col_names=["kernel_size", "output_size", "num_params", "mult_adds"], )

if os.path.exists(filePath):
    os.remove(filePath)
with open(filePath, "w", encoding="utf-8") as f:
    f.write(str(model_stats))

print(model_stats)



#### Evaluation time 
# use the test dataset
# load best saved checkpoint
best_model = model
checkpoint = torch.load(f'{weights_dir}/{hr_name}/{model_name}_loss_checkpoint.pt')
best_model.load_state_dict(checkpoint)


best_model.to(device)

####################################################################
# generating test data

from transformations import (
    ComposeSingle,
    FunctionWrapperSingle,
    normalize_01,
)

from customdatasets import DataSetTask3PatchTestSP

make_di_path(f'{results_dir}/{hr_name}/label')
for label in LABEL_DIR:
    make_di_path(f'{results_dir}/{hr_name}/label/{label}')

def get_fpath(test_csv: str, data_dir: str):

    """ test_csv: str --> path to test.csv"""

    with open(test_csv, newline='') as f:
        reader = csv.reader(f)
        test_list = list(reader)
    input = []

    for i_obs in test_list:
        fname = i_obs[0]
        input.append(data_dir+'/image/'+fname)

    return input


inputs_test = get_fpath("test.csv", DATA_DIR)


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
dataset_test = DataSetTask3PatchTestSP(
    inputs=inputs_test, transform=transforms, use_cache=False, augmenter=None, crop_width=480, crop_height=270 #new_size=new_size,
)


# preprocess function
def preprocess(img: torch.tensor):
    # img = torch.unsqueeze(img, dim=0)
    return img

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w, channels) where
    h * w *channels = arr.size

    If arr is of shape (n, nrows, ncols, channels), n sublocks of shape (nrows, ncols, channels),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols, channels = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols, channels)
               .swapaxes(1,2)
               .reshape(h, w, channels))

# postprocess function
def postprocess(mask: torch.tensor):
    mask = F.sigmoid(mask)
    mask = (mask > 0.5).type(mask.dtype)  # perform argmax to generate 1 channel
    # img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    mask = torch.squeeze(mask, dim=0)  # remove batch dim and channel dim -> [H, W]
    # img = re_normalize(img)  # scale it to the range [0-255]
    return mask

def predict(
    img,
    model,
    preprocess,
    postprocess,
    # device,
):

    model.eval()
    img = preprocess(img)  # preprocess image
    # img = img.to(device)  # to torch, send to device
    with torch.no_grad():
        mask = model(img)  # send through model/network

    # out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(mask)  # postprocess outputs

    return result


def get_obs(index, dataset, model, device):

    img = dataset[index]
    img = img.to(device)

    pr = predict(img, model, preprocess, postprocess)

    img = img.cpu().numpy()
    img = np.moveaxis(img, 1, -1)
    pr = pr.cpu().numpy()
    pr = np.moveaxis(pr, 1, -1)

    img = unblockshaped(img, 1080, 1920)
    pr = unblockshaped(pr, 1080, 1920)

    return img, pr



from PIL import Image
import os

for idx in tqdm(range(len(dataset_test)), desc='predicting'):
    _, pr = get_obs(idx, dataset_test, best_model, device)
    for label in range(N_CLASSES):
        mask_i = Image.fromarray(pr[:,:,label].astype(np.uint8))
        mask_i.save(f'{results_dir}/{hr_name}/label/{LABEL_DIR[label]}/{os.path.basename(dataset_test.inputs[idx])}', format="png")


from shutil import copy2
copy2('project2_mask2kaggle.py', f'{results_dir}/{hr_name}')
copy2('test.csv', f'{results_dir}/{hr_name}')

os.chdir(f'{results_dir}/{hr_name}')


os.system('python project2_mask2kaggle.py')


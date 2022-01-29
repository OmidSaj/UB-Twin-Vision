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
LABEL_DIR = "component"
model_name = "t1_unetpp_resnest50d_4s2x40d"
hr_name = 't1_unetpp_resnest50d_4s2x40d_hr_aug_cont3'
hr_save = 't1_unetpp_resnest50d_4s2x40d_hr_aug_cont2'
lr_name = 't1_unetpp_resnest50d_4s2x40d_lr'
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
cache_dataset = False

N_CLASSES = 8

def get_fpath(split_name: str, data_dir: str, label_dir: str, split_dir: str):

    """ split_nane: str --> 'train', 'val', 'test'"""

    split_dict=load_pickle(split_dir + '/' +split_name+'_dict')

    input = []
    target = []

    for i_obs in split_dict:
        fname = split_dict[i_obs]['fname']
        input.append(data_dir+'/image/'+fname)
        target.append(data_dir+'/label/'+label_dir+'/'+fname)

    return input, target


# training transformations
transforms = ComposeDouble(
    [
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)


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

filePath = f'{results_dir}/{hr_name}/summary.txt'
model_stats = summary(model,(1,3,1080, 1920), depth=6, col_names=["kernel_size", "output_size", "num_params", "mult_adds"],)

if os.path.exists(filePath):
    os.remove(filePath)
with open(filePath, "w", encoding="utf-8") as f:
    f.write(str(model_stats))

print(model_stats)


## Evaluation time
# Getting evaluation metrics using torchmetrics library
inputs_test, targets_test = get_fpath("test", DATA_DIR, LABEL_DIR, SPLIT_DIR)

# dataset test
dataset_test = SegmentationDataSet2(
    inputs=inputs_test, targets=targets_test,  n_classes=N_CLASSES, transform=transforms, use_cache=cache_dataset, resize=False, new_size=new_size,
)


# dataloader test
dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_EVAL, shuffle=False)



best_model = model
checkpoint = torch.load(f'{weights_dir}/{hr_name}/{model_name}_loss_checkpoint.pt')
best_model.load_state_dict(checkpoint)

# best_model = torch.load(f'{weights_dir}/{hr_name}/{model_name}_checkpoint.pt')

met_eval1 = torchmetrics.Recall(num_classes=N_CLASSES, average=None, mdmc_average='global')
met_eval1.__name__ = 'recall'

met_eval2 = torchmetrics.F1(num_classes=N_CLASSES, average=None, mdmc_average='global')
met_eval2.__name__ = 'f1_score'

# named IoU in torchmetrics version 0.62
met_eval3 = torchmetrics.JaccardIndex(num_classes=N_CLASSES, reduction='none', )
met_eval3.__name__ = 'IoU_score'

met_eval4 = torchmetrics.Precision(num_classes=N_CLASSES,  average=None, mdmc_average='global')
met_eval4.__name__ = 'precision'

metrics = [met_eval1, met_eval2, met_eval3, met_eval4]
classes_labels = ['wall', 'beam', 'column', 'window_frame', 'window_pane','balcony','slab','ignore']


best_model.to(device)

metric_results = {}
metric_results['class'] = classes_labels

for metric in metrics:
    metric.to(device)

for bidx, batch in tqdm(enumerate(dataloader_test), disable=disable_tqdm):
    img, gt = batch[0].to(device), batch[1].to(device)
    print(img.shape)
    best_model.eval()
    with torch.no_grad():
        pr = best_model(img)
    for metric in metrics:
        metric(pr, gt)

for metric in metrics:
    metric_results[metric.__name__] = metric.compute().cpu().tolist()
    metric.reset()


for metric in metrics:
    value = metric_results[metric.__name__]
    print(f'{metric.__name__}: [' +  '%.4f ,' *len(value) % tuple(value) + f'], average: {np.average(value):.4f}')

with open(f"{results_dir}/{hr_name}/metrics.csv", "w", newline="") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(metric_results.keys())
   writer.writerows(zip(*metric_results.values()))
   writer.writerow('')
   average_row = [np.average(metric_results[metric.__name__]) for metric in metrics]
   average_row.insert(0, "average")
   writer.writerow(average_row)


# generating predictions

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

    img, gt = dataset[index]
    img, gt = img.to(device), gt.to(device)

    pr = predict(img, model, preprocess, postprocess)

    img = img.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    gt = gt.cpu().numpy()
    pr = pr.cpu().numpy()

    return img, gt, pr


# check output shape
img, gt, pr = get_obs(0, dataset_test, best_model, device)

print(f'image shape: {img.shape}, type: {type(img)}')
print(f'ground trurth shape: {gt.shape}, type: {type(gt)}')
print(f'prediction shape: {pr.shape}, type: {type(pr)}')

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    # plt.show()


for i in range(10):
    idx = np.random.randint(0, len(dataset_test))
    img, gt, pr = get_obs(idx, dataset_test, best_model, device)
    print(f'ID: {idx}')
    visualize(image=img, truth=gt, prediction=pr)
    plt.savefig(f'{results_dir}/{hr_name}/figures/trial_{idx}.png')
    #plt.show()
    plt.close()





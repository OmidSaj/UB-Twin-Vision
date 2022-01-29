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
model_name = "t3_unetpp_cropv3_aug_rebar"
hr_name = 'cropv3_rebar_cont2'
hr_save = 'cropv3_rebar_cont2'
# LABEL_DIR = ["spall", "crack", "rebar"]

LABEL_DIR = ["rebar"]

BATCH_TRAIN = 4
BATCH_EVAL = 4

N_CLASSES = len(LABEL_DIR)

# make_di_path(model_name)
make_di_path(f'{results_dir}')
make_di_path(f'{weights_dir}')
make_di_path(f'{weights_dir}/{hr_name}')
make_di_path(f'{results_dir}/{hr_name}')
make_di_path(f'{results_dir}/{hr_name}/figures')

def get_fpath(split_name: str, data_dir: str, label_dir: list, split_dir: str):

    """ split_nane: str --> 'train', 'val', 'test'"""

    split_dict=load_pickle(split_dir+'/'+split_name+'_dict')

    input = []
    target_list = []

    for i_obs in split_dict:
        fname = split_dict[i_obs]['fname']
        input.append(data_dir+'/image/'+fname)
        targets = []
        for target in label_dir:
            targets.append(data_dir+'/label/'+target+'/'+fname)
        target_list.append(targets)
    return input, target_list

# training transformations and augmentations
transforms = ComposeDouble(
    [
        # FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=True, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)



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

inputs_test, targets_test = get_fpath("test", DATA_DIR, LABEL_DIR, SPLIT_DIR)

dataset_test = DataSetTask3Patch(
    inputs=inputs_test, targets=targets_test,  n_classes=N_CLASSES, transform=transforms, use_cache=False, augmenter=None, crop_width=480, crop_height=270 #new_size=new_size,
)

# dataloader validation
dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)


x, y = next(iter(dataloader_test))

print(f"x = shape: {x.shape}; type: {x.dtype}")
print(f"x = min: {x.min()}; max: {x.max()}")
print(f"y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}")
print(f"y = min: {y.min()}; max: {y.max()}")

classes_labels = LABEL_DIR
metrics_labels = ['TN', 'FP', 'FN', 'TP', 'Recall', 'Precision', 'F1', 'IoU']
cm = []
for label in range(N_CLASSES):
    cm.append(torch.tensor([[0,0],[0,0]]).to(device))
eval_batch = 4

best_model.to(device)

metric_results = {}
metric_results['class'] = classes_labels

for bidx, batch in tqdm(enumerate(dataloader_test)):
    img, gt = torch.squeeze(batch[0], dim=0).to(device), torch.squeeze(batch[1], dim=0).to(device)
    for i in range(int(img.shape[0] / eval_batch)):
        best_model.eval()
        with torch.no_grad():
            pr = F.sigmoid(best_model(img[i * eval_batch:(i + 1) * eval_batch]))
        for label in range(N_CLASSES):
            cm[label] += confusion_matrix(pr[:, label], gt[i * eval_batch:(i + 1) * eval_batch, label], 2)

for label in range(N_CLASSES):
    for metric in metrics_labels:
        metric_results[metric] = []


for label in range(N_CLASSES):
    metric_results['TN'].append(cm[label][0, 0].cpu().numpy())
    metric_results['FP'].append(cm[label][0, 1].cpu().numpy())
    metric_results['FN'].append(cm[label][1, 0].cpu().numpy())
    metric_results['TP'].append(cm[label][1, 1].cpu().numpy())

    metric_results['Recall'].append(
        metric_results['TP'][label] / (metric_results['TP'][label] + metric_results['FN'][label]))
    metric_results['Precision'].append(
        metric_results['TP'][label] / (metric_results['TP'][label] + metric_results['FP'][label]))
    metric_results['F1'].append(
        (2 * metric_results['TP'][label]) / (2 * metric_results['TP'][label] + metric_results['FP'][label] + metric_results['FN'][label]))
    metric_results['IoU'].append(
        (metric_results['TP'][label]) / (metric_results['TP'][label] + metric_results['FP'][label] + metric_results['FN'][label]))

for metric in metrics_labels:
    value = metric_results[metric]
    print(f'{metric}: [' + '%.4f ,' * len(value) % tuple(value) + f'], average: {np.average(value):.4f}')

with open(f"{results_dir}/{hr_name}/metrics_cm.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(metric_results.keys())
    writer.writerows(zip(*metric_results.values()))
    writer.writerow('')
    average_row = [np.average(metric_results[metric]) for metric in metrics_labels]
    average_row.insert(0, "average")
    # writer.writerow(["Average"])
    # writer.writerow([np.average(metric_results[metric.__name__]) for metric in metrics])
    writer.writerow(average_row)

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
    mask = torch.squeeze(mask)  # remove batch dim and channel dim -> [H, W]
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

    img, gt = dataset[index]
    img, gt = img.to(device), gt.to(device)

    pr = predict(img, model, preprocess, postprocess)

    img = img.cpu().numpy()
    img = np.moveaxis(img, 1, -1)
    gt = gt.cpu().numpy()
    gt = np.moveaxis(gt, 1, -1)
    pr = pr.cpu().numpy()
    pr = np.moveaxis(pr, 1, -1)

    img = unblockshaped(img, 1080, 1920)
    gt = unblockshaped(gt, 1080, 1920)
    pr = unblockshaped(pr, 1080, 1920)

    return img, gt, pr



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
    visualize(image=img)
    for i in range(N_CLASSES):
        visualize(truth=gt[:,:,i], prediction=pr[:,:,i])
        plt.savefig(f'{results_dir}/{hr_name}/figures/trial_{idx}_{LABEL_DIR[i]}.png')
        # plt.show()
        plt.close()

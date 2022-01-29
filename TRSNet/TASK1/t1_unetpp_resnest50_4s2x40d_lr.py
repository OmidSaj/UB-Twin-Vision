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
hr_name = 't1_unetpp_resnest50d_4s2x40d_hr'
lr_name = 't1_unetpp_resnest50d_4s2x40d_lr'
results_dir = "results"
weights_dir = "weights"

# make_di_path(model_name)
make_di_path(f'{results_dir}')
make_di_path(f'{weights_dir}')
make_di_path(f'{weights_dir}/{lr_name}')
make_di_path(f'{results_dir}/{lr_name}')
make_di_path(f'{results_dir}/{lr_name}/figures')

BATCH_TRAIN = 8
BATCH_EVAL = 8
disable_tqdm = True
cache_dataset = True

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


inputs_train, targets_train = get_fpath("train", DATA_DIR, LABEL_DIR, SPLIT_DIR)
inputs_val, targets_val = get_fpath("val", DATA_DIR, LABEL_DIR, SPLIT_DIR)


# training transformations
transforms = ComposeDouble(
    [
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)

# size: (width, height) for PIL library
new_size = (480,270)


# dataset training
dataset_train = SegmentationDataSet2(
    inputs=inputs_train, targets=targets_train, n_classes=N_CLASSES, transform=transforms, use_cache=cache_dataset, resize=True, new_size=new_size,
)

# dataset validation
dataset_val = SegmentationDataSet2(
    inputs=inputs_val, targets=targets_val,  n_classes=N_CLASSES, transform=transforms, use_cache=cache_dataset, resize=True, new_size=new_size,
)


# dataloader training
dataloader_training = DataLoader(dataset=dataset_train, batch_size=BATCH_TRAIN, shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_val, batch_size=BATCH_EVAL, shuffle=False)

x, y = next(iter(dataloader_training))

print(f"x = shape: {x.shape}; type: {x.dtype}")
print(f"x = min: {x.min()}; max: {x.max()}")
print(f"y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}")

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
# size: (270, 480)
# pad = 288 - 270 = 18
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


model = padded_model(bottleneck_model, pad)

filePath = f'{results_dir}/{lr_name}/summary.txt'
model_stats = summary(model,(1,3,270, 480), depth=6, col_names=["kernel_size", "output_size", "num_params", "mult_adds"],)

if os.path.exists(filePath):
    os.remove(filePath)
with open(filePath, "w", encoding="utf-8") as f:
    f.write(str(model_stats))

print(model_stats)

loss = smp.losses.focal.FocalLoss(mode='multiclass', alpha=0.25, gamma=2.0)
loss.__name__ = 'focal_loss'

met1 = torchmetrics.Accuracy(num_classes=N_CLASSES, mdmc_average='global')
met1.__name__ = 'accuracy'

met2 = torchmetrics.F1(num_classes=N_CLASSES, average='macro', mdmc_average='global')
met2.__name__ = 'f1_score'

metrics = [
    met1,
    met2
]

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

class ReduceLROnPlateauPatch(torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]

scheduler = ReduceLROnPlateauPatch(optimizer, factor=0.5, patience=10, verbose=True, threshold=0.0001, min_lr=1E-5)


train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=not disable_tqdm,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=not disable_tqdm,
)


###############

# checkpoint = torch.load(f'{model_name}/hr_us_ESPCN_ds_iESPCN_trainUNet/{model_name}_checkpoint.pt')
# model.load_state_dict(checkpoint)

row = ['epoch', loss.__name__]
for metric in metrics:
    row.append(metric.__name__)

with open(f'{results_dir}/{lr_name}/{model_name}_train_logger.csv', 'a', newline='') as f:
    w = csv.writer(f)
    w.writerow(row)

with open(f'{results_dir}/{lr_name}/{model_name}_valid_logger.csv', 'a', newline='') as f:
    w = csv.writer(f)
    w.writerow(row)

EPOCHS = 100
max_score = 0


train_list = []
valid_list = []
lr_list = []
for i in range(0, EPOCHS):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(dataloader_training)
    valid_logs = valid_epoch.run(dataloader_validation)
    scheduler.step(valid_logs[loss.__name__])
    train_list.append(train_logs)
    valid_list.append(valid_logs)
    lr_list.append(scheduler.get_lr())

    with open(f'{results_dir}/{lr_name}/{model_name}_train_logger.csv', 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(train_logs.values()))

    with open(f'{results_dir}/{lr_name}/{model_name}_valid_logger.csv', 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(valid_logs.values()))

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['f1_score']:
        max_score = valid_logs['f1_score']
        torch.save(model.state_dict(), f'{weights_dir}/{lr_name}/{model_name}_checkpoint.pt')
        print('Model saved!')



keys = valid_list[0].keys()

a_file = open(f"{results_dir}/{lr_name}/valid_logs.csv", "w", newline="")
dict_writer = csv.DictWriter(a_file, keys)
dict_writer.writeheader()
dict_writer.writerows(valid_list)
a_file.close()


keys = train_list[0].keys()

a_file = open(f"{results_dir}/{lr_name}/train_logs.csv", "w", newline="")
dict_writer = csv.DictWriter(a_file, keys)
dict_writer.writeheader()
dict_writer.writerows(train_list)
a_file.close()


np.savetxt(f"{results_dir}/{lr_name}/lr_log.csv", lr_list)


val_loss = []
val_met = []
train_loss = []
train_met = []

actual_epochs = len(valid_list)

for i in range(actual_epochs):
    val_loss.append(valid_list[i][loss.__name__])
    val_met.append(valid_list[i][met2.__name__])
    train_loss.append(train_list[i][loss.__name__])
    train_met.append(train_list[i][met2.__name__])

plt.figure()
plt.plot(train_loss, label=f'train {loss.__name__}')
plt.plot(val_loss, label=f'val {loss.__name__}')
plt.legend()
plt.savefig(f'{results_dir}/{lr_name}/figures/{loss.__name__}_graph.png')
plt.savefig(f'{results_dir}/{lr_name}/figures/{loss.__name__}_graph.svg')

plt.figure()
plt.plot(train_met, label=f'train {met2.__name__}')
plt.plot(val_met, label=f'val {met2.__name__}')
plt.legend()
plt.savefig(f'{results_dir}/{lr_name}/figures/{met2.__name__}_graph.png')
plt.savefig(f'{results_dir}/{lr_name}/figures/{met2.__name__}_graph.svg')

plt.figure()
plt.plot(lr_list, label='learning_rate')
plt.legend()
plt.savefig(f'{results_dir}/{lr_name}/figures/lr_graph.png')
plt.savefig(f'{results_dir}/{lr_name}/figures/lr_graph.svg')


## Evaluation time
# Getting evaluation metrics using torchmetrics library

best_model = model
checkpoint = torch.load(f'{weights_dir}/{lr_name}/{model_name}_checkpoint.pt')
best_model.load_state_dict(checkpoint)

# best_model = torch.load(f'{weights_dir}/{lr_name}/{model_name}_checkpoint.pt')

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
classes_labels = ['no damage','light damage','moderate damage','severe damage','ignore']


best_model.to(device)

metric_results = {}
metric_results['class'] = classes_labels

for metric in metrics:
    metric.to(device)

for bidx, batch in tqdm(enumerate(dataloader_validation), disable=disable_tqdm):
    img, gt = batch[0].to(device), batch[1].to(device)

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

with open(f"{model_name}/{lr_name}/metrics.csv", "w", newline="") as outfile:
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
img, gt, pr = get_obs(0, dataset_val, best_model, device)

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
    idx = np.random.randint(0, len(dataset_val))
    img, gt, pr = get_obs(idx, dataset_val, best_model, device)
    print(f'ID: {idx}')
    visualize(image=img, truth=gt, prediction=pr)
    plt.savefig(f'{model_name}/{lr_name}/figures/trial_{idx}.png')
    #plt.show()
    plt.close()




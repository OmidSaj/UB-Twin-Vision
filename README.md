# Twin Models for High Resolution Visual Inspections
## IC-SHM 2021, Project 2

A team effort by PhD candidates:

[Kareem Eltouny](https://github.com/keltouny) and [Seyedomid Sajedi](https://github.com/OmidSaj)

Jan 2022, 
University at Buffalo (SUNY), 
Department of Civil, Structural and Environmental Engineering

## Introduction
This is the official repository for the code and models used by UB-SHM team in IC-SHM 2021, project 2. Models are evalauted on the [QuakeCity](https://sail.cive.uh.edu/quakecity/) benchmark dataset. 

![Segmentation demo](https://github.com/OmidSaj/UB-Twin-Vision/blob/main/Assets/Figures/A20400.png)

## Models and code
The code and models are developed using ![PyTorch](https://pytorch.org/).

| Name | Segmentation Task | Trained Model/weights |
| :---: | :---: | :---: | 
| ![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/TRSNet/TASK1) | Component type | link |
| ![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/TRSNet/TASK2)  | Component damage severity | link |
| ![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/TRSNet/TASK3)  | Cracks, rebar exposure, spalling | link |
| ![DmgFormer-S](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/DmgFormer) | Cracks, rebar exposure, spalling | link |
| ![DmgFormer-L](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/DmgFormer) | Cracks, rebar exposure, spalling | link |

## TRS-Net

![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/blob/main/Assets/Figures/TRS-Net.png)

## DmgFormer

![DmgFormer](https://github.com/OmidSaj/UB-Twin-Vision/blob/main/Assets/Figures/DmgFormer.jpg)

## Acknowledgements
* The [Swin Transformer](https://github.com/microsoft/Swin-Transformer) backbone implementation is a slight modification of the official repository from [Microsoft ](https://github.com/microsoft/Swin-Transformer)
* TRS-Net is based on ResNeSt and U-Net++ implementations from [SMP](https://github.com/qubvel/segmentation_models.pytorch) and [timm-models](https://github.com/rwightman/pytorch-image-models)

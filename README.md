# Twin Models for High Resolution Visual Inspections
## IC-SHM 2021, Project 2

A team effort by PhD candidates:

[Kareem Eltouny](https://github.com/keltouny) and [Seyedomid Sajedi](https://github.com/OmidSaj)

Jan 2022, 
University at Buffalo (SUNY), 
Department of Civil, Structural and Environmental Engineering

## Introduction
This is the official repository for the code and models used by UB-SHM team in IC-SHM 2021, project 2. Models are evalauted on the [QuakeCity](https://sail.cive.uh.edu/quakecity/) benchmark dataset. You can find further details about the twin models in this [report](https://github.com/OmidSaj/UB-Twin-Vision/blob/main/Assets/Report.pdf).

![Segmentation demo](https://github.com/OmidSaj/UB-Twin-Vision/blob/main/Assets/Figures/icshm.gif)

## Models and code
The code and models are developed using ![PyTorch](https://pytorch.org/). Trained models and weights can be found from [releases](https://github.com/OmidSaj/UB-Twin-Vision/releases). 

| Name | Segmentation Task |
| :---: | :---: |
| ![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/TRSNet/TASK1) | Component type |
| ![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/TRSNet/TASK2)  | Component damage severity |
| ![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/TRSNet/TASK3)  | Cracks, rebar exposure, spalling |
| ![DmgFormer-S](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/DmgFormer) | Cracks, rebar exposure, spalling |
| ![DmgFormer-L](https://github.com/OmidSaj/UB-Twin-Vision/tree/main/DmgFormer) | Cracks, rebar exposure, spalling |

## TRS-Net

![TRS-Net](https://github.com/OmidSaj/UB-Twin-Vision/blob/main/Assets/Figures/TRS-Net.png)

## DmgFormer

![DmgFormer](https://github.com/OmidSaj/UB-Twin-Vision/blob/main/Assets/Figures/DmgFormer.jpg)

## Acknowledgements
* The [Swin Transformer](https://github.com/microsoft/Swin-Transformer) backbone implementation is a slight modification of the official repository from [Microsoft ](https://github.com/microsoft/Swin-Transformer)
* TRS-Net is based on ResNeSt[1] and U-Net++[2] implementations from [SMP](https://github.com/qubvel/segmentation_models.pytorch) and [timm-models](https://github.com/rwightman/pytorch-image-models)

### References
[1] H. Zhang et al., "Resnest: Split-attention networks," arXiv preprint arXiv:2004.08955, 2020.

[2] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "Unet++: A nested u-net architecture for medical image segmentation," in Deep learning in medical image analysis and multimodal learning for clinical decision support: Springer, 2018, pp. 3-11.

[3] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10012-10022.

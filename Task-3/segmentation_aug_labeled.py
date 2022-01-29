# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:51:17 2021

@author: Seyed Omid Sajedi """

import torch 
import matplotlib
from torch_transformer2 import VisionTransformer,Confing
from CropDataGenerators_tv import torch_ViT_DataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from OsUtils import make_di_path,wipe_dir

from torch_eval import torch_eval_ViT_seg_aug
from torch_transformer2 import FocalLoss

from decode_head_L import ResDecoderPU, Segmentor_swin_resd
import types
from microsoft_swin.swin_transformer import SwinTransformer

def forward_features_bbone(self, x):
    x = self.patch_embed(x)
    if self.ape:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    str_feat_outs=[]
    ps=[112,56,28,14,7]
    for i,layer in enumerate(self.layers):
        x = layer(x)
        str_feat_outs.append(x.view(x.size(0),ps[i],ps[i],x.size(-1)).swapaxes(1, 3).swapaxes(2, 3))
    return str_feat_outs

new_img_h=1080
new_img_w=1920
n_crop_h=5
n_crop_w=9

label_id=1

config=Confing(
                np_h=10,
                np_w=10,
                img_size=224,
                n_classes=1,
                n_layers=2,
                n_heads=16,
                hidden_size=192,
                P_do_dense=0.0,
                P_do_attn=0.0,
                P_do_head=0.0)
            
make_di_path('attention_plots')    
make_di_path('attention_plots/ViT_M_adv')  
wipe_dir('attention_plots/ViT_M_adv')

batch_size_eval=1

filter_bg=False
test_generator=torch_ViT_DataGenerator(config,'test',
                           batch_size_img=batch_size_eval,
                           shuffle=False,
                           bg_augmenter=None,
                           augmenter=None,    
                           rand_lr_flip=False,
                           rand_ud_flip=False,
                           P_aug=0.0,
                           n_crop_h=n_crop_h,
                           n_crop_w=n_crop_w,
                           resize_img=False,
                           new_size=(new_img_w,new_img_h),
                           noisy_crop=False,
                           crop_noise_amp=1,
                           use_cache_img=False,
                           use_cache_mask=False,                                    
                           return_mask=True,
                           return_seg=True,
                           filter_bg=filter_bg,
                           zero_pad=(40,96))

model_bbone = SwinTransformer(num_classes=3,patch_size=2,
                              embed_dim=64, depths=[2, 2, 2, 18, 2], num_heads=[4, 4, 8, 16, 32])
    

model_bbone.forward_features_bbone = types.MethodType(forward_features_bbone, model_bbone )
model_bbone.cuda()

model_head=ResDecoderPU(config ,bbone_feat_dim=48*16)
model_head.cuda()
model = Segmentor_swin_resd(model_bbone, model_head)

model.load_state_dict(torch.load('DmgFormer_L.pt'))
model.cuda()
print('DmgFormer_L.pt')

test_metric_dict = torch_eval_ViT_seg_aug(model,test_generator,
                            label_list=['Crack','Rebar','Spall'],
                            n_crop_h=5,
                            n_crop_w=9,
                            new_img_h = 1080,
                            new_img_w = 1920,
                            h_pad_list=[0,1,2],
                            v_pad_list=[0,1,2])
    
print('1080p Testing metric summary(%):')
print('            Crack Rebar Spall')
print('Precision   %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['Precision'][1]*100,test_metric_dict['Rebar']['Precision'][1]*100,test_metric_dict['Spall']['Precision'][1]*100))
print('Recall      %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['Recall'][1]*100,test_metric_dict['Rebar']['Recall'][1]*100,test_metric_dict['Spall']['Recall'][1]*100))
print('F1 score    %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['F1_score'][1]*100,test_metric_dict['Rebar']['F1_score'][1]*100,test_metric_dict['Spall']['F1_score'][1]*100))
print('IoU         %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['IoU'][1]*100,test_metric_dict['Rebar']['IoU'][1]*100,test_metric_dict['Spall']['IoU'][1]*100))

test_metric_dict = torch_eval_ViT_seg_aug(model,test_generator,
                            label_list=['Crack','Rebar','Spall'],
                            n_crop_h=5,
                            n_crop_w=9,
                            new_img_h = 1080,
                            new_img_w = 1920,
                            h_pad_list=[0], 
                            v_pad_list=[0])
    
print('1080p Testing metric summary(%):')
print('            Crack Rebar Spall')
print('Precision   %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['Precision'][1]*100,test_metric_dict['Rebar']['Precision'][1]*100,test_metric_dict['Spall']['Precision'][1]*100))
print('Recall      %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['Recall'][1]*100,test_metric_dict['Rebar']['Recall'][1]*100,test_metric_dict['Spall']['Recall'][1]*100))
print('F1 score    %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['F1_score'][1]*100,test_metric_dict['Rebar']['F1_score'][1]*100,test_metric_dict['Spall']['F1_score'][1]*100))
print('IoU         %1.2f %1.2f %1.2f'%(test_metric_dict['Crack']['IoU'][1]*100,test_metric_dict['Rebar']['IoU'][1]*100,test_metric_dict['Spall']['IoU'][1]*100))    
    

# =============================================================================
# Additional inference graphics
# =============================================================================
# n_img_plot=test_generator.__len__()
# mIoU=0
# count=0
# N_MCS=1
# count_zero=0

# h_pad_list=[0,1,2] #[0,1]
# v_pad_list=[0,1,2] #[0,1]
# flip_h_list=[0]

# def unpad_tensor(tensor,zero_pad=(40,96),pad_h=0,pad_v=0):
#     # Y_target[:,224,224,3]
#     if pad_h==0:
#         tensor=tensor[zero_pad[0]:,:,:]
#     if pad_h==1:
#         tensor=tensor[:-1*zero_pad[0],:,:]
#     if pad_h==2:
#         tensor=tensor[zero_pad[0]//2:-1*zero_pad[0]//2,:,:]
#     if pad_v==0:
#         tensor=tensor[:,zero_pad[1]:,:]
#     if pad_v==1:
#         tensor=tensor[:,:-1*zero_pad[1],:]
#     if pad_v==2:
#         tensor=tensor[:,zero_pad[1]//2:-1*zero_pad[1]//2,:]
#     return tensor
        
# model.eval()
# IoU_dict={}
# label_list=['Crack','Rebar','Spall']

# for iLabel in label_list:
#     IoU_dict[iLabel]={}
#     IoU_dict[iLabel]['IoU']=0
#     IoU_dict[iLabel]['Count']=0

# for i_img in range(n_img_plot):
#     with torch.no_grad():
                
#         Y_pred_prob=[]
#         for flip_h_i in flip_h_list:
#             for h_had_i in h_pad_list:
#                 for v_had_i in v_pad_list:
#                     X_smaple,Y_sample=test_generator.__getitem__(i_img,return_seg_mask=False,pad_h=h_had_i,pad_v=v_had_i)
#                     if flip_h_i==1:
#                         X_smaple = torch.flip(X_smaple,[3])
#                     Y_pred_prob_i=torch.sigmoid(model(X_smaple.cuda()))
#                     # print(X_smaple.size())
#                     if flip_h_i==1:
#                         Y_pred_prob_i = torch.flip(Y_pred_prob_i,[2])
#                         X_smaple = torch.flip(X_smaple,[3])
                
#                     Y_pred_prob_i=Y_pred_prob_i.reshape(Y_pred_prob_i.shape[0],config.img_size,config.img_size,Y_pred_prob_i.shape[-1])
#                     Y_pred_prob_i=Y_pred_prob_i.reshape(n_crop_h,n_crop_w,Y_pred_prob_i.shape[1],Y_pred_prob_i.shape[2],Y_pred_prob_i.shape[-1]).swapaxes(1,2)
#                     Y_pred_prob_i=Y_pred_prob_i.reshape(new_img_h+test_generator.zero_pad[0],new_img_w+test_generator.zero_pad[1],Y_pred_prob_i.shape[-1])
                    
#                     Y_pred_prob_i=unpad_tensor(Y_pred_prob_i,zero_pad=test_generator.zero_pad,pad_h=h_had_i,pad_v=v_had_i)
                    
#                     Y_pred_prob.append(Y_pred_prob_i.cpu().detach().numpy())            

#         Y_pred_prob=np.mean(np.stack(Y_pred_prob),axis=0)
#         # Y_pred = torch.round(torch.sigmoid(model(X_smaple.cuda()))
        
#         X_smaple=X_smaple.cpu().detach().numpy().swapaxes(1,3).swapaxes(1,2)
#         X_smaple=X_smaple.reshape(n_crop_h,n_crop_w,X_smaple.shape[1],X_smaple.shape[2],X_smaple.shape[-1]).swapaxes(1,2)
#         X_smaple=X_smaple.reshape(new_img_h+test_generator.zero_pad[0],new_img_w+test_generator.zero_pad[1],X_smaple.shape[-1])
#         X_smaple=unpad_tensor(X_smaple,zero_pad=test_generator.zero_pad,pad_h=h_had_i,pad_v=v_had_i)
        
#         Y_sample=Y_sample.numpy()
#         Y_sample=Y_sample.reshape(Y_sample.shape[0],config.img_size,config.img_size,Y_sample.shape[-1])
#         Y_sample=Y_sample.reshape(n_crop_h,n_crop_w,Y_sample.shape[1],Y_sample.shape[2],Y_sample.shape[-1]).swapaxes(1,2)
#         Y_sample=Y_sample.reshape(new_img_h+test_generator.zero_pad[0],new_img_w+test_generator.zero_pad[1],Y_sample.shape[-1])
#         Y_sample=unpad_tensor(Y_sample,zero_pad=test_generator.zero_pad,pad_h=h_had_i,pad_v=v_had_i)
        
#         Y_pred=np.round(Y_pred_prob)
        
#         intsec_tensor=Y_pred*Y_sample
#         union_tensor=Y_pred+Y_sample-intsec_tensor
#         for iL,iLabel in enumerate(label_list):
#             U=np.sum(union_tensor[:,:,iL])
#             if U>0:
#                 I=np.sum(intsec_tensor[:,:,iL])
#                 IoU_dict[iLabel]['Count']+=1
#                 IoU_dict[iLabel]['IoU']=(I/U+(IoU_dict[iLabel]['IoU']*(IoU_dict[iLabel]['Count']-1)))/IoU_dict[iLabel]['Count']
#         # print('            Crack Rebar Spall')
#         print('%s: %1.2f %1.2f %1.2f'%(str(i_img+1).zfill(3), IoU_dict['Crack']['IoU']*100,IoU_dict['Rebar']['IoU']*100,IoU_dict['Spall']['IoU']*100))
            
#         fig,ax=plt.subplots(3,4,figsize=(4*6,1.5*6))
        
#         ax[0,0].imshow(X_smaple)
#         ax[1,0].set_visible(False)
#         ax[2,0].set_visible(False)
#         for i in range(3):
#             ax[0,i+1].imshow(X_smaple)
#             ax[0,i+1].imshow(Y_sample[:,:,i],cmap='Reds',vmin=0,vmax=1,alpha=0.8)
#             ax[1,i+1].imshow(X_smaple)
#             ax[1,i+1].imshow(Y_pred[:,:,i],cmap='Reds',vmin=0,vmax=1,alpha=0.8)
#             ax[2,i+1].imshow(Y_pred[:,:,i]!=Y_sample[:,:,i],cmap='Reds',vmin=0,vmax=1,alpha=0.9)
#             ax[0,i+1].axes.xaxis.set_visible(False)
#             ax[0,i+1].axes.yaxis.set_visible(False)          
#             ax[1,i+1].axes.xaxis.set_visible(False)
#             ax[1,i+1].axes.yaxis.set_visible(False)  
#             ax[2,i+1].axes.xaxis.set_visible(False)
#             ax[2,i+1].axes.yaxis.set_visible(False)  
#         fig.savefig('attention_plots/ViT_M_adv/s_'+str(i_img+1)+'.png',dpi=300,bbox_inches='tight')
#         plt.close()     
#         print(i_img+1)
            
# print(IoU_dict)           
        

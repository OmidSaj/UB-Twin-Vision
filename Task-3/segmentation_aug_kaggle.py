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


from torch_eval import torch_eval_ViT_seg, torch_eval_ViT_seg_aug
from torch_transformer2 import FocalLoss

from decode_head_L import ResDecoderPU, Segmentor_swin_resd
import types
from microsoft_swin.swin_transformer import SwinTransformer
from PIL import Image

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
    
# load pretrained model

# best_model = VisionTransformer(config)
label_list=['crack','rebar','spall']        
make_di_path('label')    
make_di_path('label/check')    
wipe_dir('label/check')

for iLabel in label_list:
    make_di_path('label/'+iLabel+'/')  
    wipe_dir('label/'+iLabel)

batch_size_eval=1

filter_bg=False
test_generator=torch_ViT_DataGenerator(config,'main_test',
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
                           return_mask=False,
                           return_seg=True,
                           filter_bg=filter_bg,
                           zero_pad=(40,96))

# =============================================================================
# Load the pre-trained model
# =============================================================================
model_bbone = SwinTransformer(num_classes=3,patch_size=2,
                              embed_dim=64, depths=[2, 2, 2, 18, 2], num_heads=[4, 4, 8, 16, 32])
model_bbone.forward_features_bbone = types.MethodType(forward_features_bbone, model_bbone )
model_bbone.cuda()


model_head=ResDecoderPU(config ,bbone_feat_dim=64*16)
model_head.cuda()
model = Segmentor_swin_resd(model_bbone, model_head)
model.load_state_dict(torch.load('DmgFormer_L.pt'))
model.cuda()

n_img_plot=test_generator.__len__()

h_pad_list=[0,1,2] #[0,1]
v_pad_list=[0,1,2] #[0,1]

def unpad_tensor(tensor,zero_pad=(40,96),pad_h=0,pad_v=0):
    # Y_target[:,224,224,3]
    if pad_h==0:
        tensor=tensor[zero_pad[0]:,:,:]
    if pad_h==1:
        tensor=tensor[:-1*zero_pad[0],:,:]
    if pad_h==2:
        tensor=tensor[zero_pad[0]//2:-1*zero_pad[0]//2,:,:]
    if pad_v==0:
        tensor=tensor[:,zero_pad[1]:,:]
    if pad_v==1:
        tensor=tensor[:,:-1*zero_pad[1],:]
    if pad_v==2:
        tensor=tensor[:,zero_pad[1]//2:-1*zero_pad[1]//2,:]
    return tensor

with torch.no_grad():
    model.eval()
    for i_img in range(n_img_plot):
        with torch.no_grad():
            
            Y_pred_prob=[]
            fname_i=test_generator.get_fname(i_img)[0:-4]
            for h_had_i in h_pad_list:
                for v_had_i in v_pad_list:
                    X_smaple=test_generator.__getitem__(i_img,return_seg_mask=False,pad_h=h_had_i,pad_v=v_had_i)[0]
                    Y_pred_prob_i=torch.sigmoid(model(X_smaple.cuda()))
                
                    Y_pred_prob_i=Y_pred_prob_i.reshape(Y_pred_prob_i.shape[0],config.img_size,config.img_size,Y_pred_prob_i.shape[-1])
                    Y_pred_prob_i=Y_pred_prob_i.reshape(n_crop_h,n_crop_w,Y_pred_prob_i.shape[1],Y_pred_prob_i.shape[2],Y_pred_prob_i.shape[-1]).swapaxes(1,2)
                    Y_pred_prob_i=Y_pred_prob_i.reshape(new_img_h+test_generator.zero_pad[0],new_img_w+test_generator.zero_pad[1],Y_pred_prob_i.shape[-1])
                    
                    Y_pred_prob_i=unpad_tensor(Y_pred_prob_i,zero_pad=test_generator.zero_pad,pad_h=h_had_i,pad_v=v_had_i)
                    Y_pred_prob.append(Y_pred_prob_i.cpu().detach().numpy())            
    
            Y_pred_prob=np.mean(np.stack(Y_pred_prob),axis=0)
            
            X_smaple=X_smaple.cpu().detach().numpy().swapaxes(1,3).swapaxes(1,2)
            X_smaple=X_smaple.reshape(n_crop_h,n_crop_w,X_smaple.shape[1],X_smaple.shape[2],X_smaple.shape[-1]).swapaxes(1,2)
            X_smaple=X_smaple.reshape(new_img_h+test_generator.zero_pad[0],new_img_w+test_generator.zero_pad[1],X_smaple.shape[-1])
            X_smaple=unpad_tensor(X_smaple,zero_pad=test_generator.zero_pad,pad_h=h_had_i,pad_v=v_had_i)
            
   
            Y_pred=np.round(Y_pred_prob)            
                     
            for i_ch,iLabel in enumerate(label_list):
                mask_i = Image.fromarray(Y_pred[:,:,i_ch].astype(np.uint8))
                # mask_i = mask_i.convert('RGB')
                mask_i.save('label/'+iLabel+'/'+fname_i+'.png',format="png")  
  
            fig,ax=plt.subplots(2,2,figsize=(3.2*4,3.8*2))
            
            ax[0,0].imshow(X_smaple)
            ax[0,0].set_title(fname_i)
            ax[0,1].imshow(X_smaple)
            ax[0,1].imshow(Y_pred[:,:,0],cmap='cividis',vmin=0,vmax=1,alpha=0.7)
            ax[0,1].set_title('Crack')
            ax[1,0].imshow(X_smaple)
            ax[1,0].imshow(Y_pred[:,:,1],cmap='cividis',vmin=0,vmax=1,alpha=0.7)
            ax[1,0].set_title('Rebar')
            ax[1,1].imshow(X_smaple)
            ax[1,1].imshow(Y_pred[:,:,2],cmap='cividis',vmin=0,vmax=1,alpha=0.7)
            ax[1,1].set_title('Spall')
            for i in range(2):
                for j in range(2):
                    ax[i,j].axes.xaxis.set_visible(False)
                    ax[i,j].axes.yaxis.set_visible(False)          
 
            fig.savefig('label/check/'+fname_i+'.png',dpi=96,bbox_inches='tight')
            plt.close()       
            
            
        

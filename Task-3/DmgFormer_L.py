# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:43:12 2021

@author: Seyed Omid Sajedi """

# =============================================================================
# Libraries
# =============================================================================
import sys

# Prints current Python version
print("Current version of Python is ", sys.version)

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time
import random
from CropDataGenerators_tv import torch_ViT_DataGenerator

# from DataAugmentation import CreateAugmenter

from torch_transformer2 import Confing,count_parameters,FocalLoss
from torch_eval import torch_eval_ViT_seg,plot_loss_log_ViT_seg
from OsUtils import load_pickle,save_pickle
import torch
import torch.optim as optim
# from tqdm import tqdm
import gc
from transformers import get_cosine_schedule_with_warmup
from decode_head_L import ResDecoderPU, Segmentor_swin_resd
# import torch.nn.functional as F
from microsoft_swin.swin_transformer import SwinTransformer

import types

def pkl_to_df_split(split_name):
    split_dict_set=load_pickle('split_dictionaries/'+split_name+'_dict')
    return pd.DataFrame.from_dict(split_dict_set,orient='index')

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
# =============================================================================
# Configs, dataset and augmentation 
# =============================================================================

if __name__ == '__main__':   
 
    n_epoch=300
    early_stoping=10
    
    # _________________________________________________________________________
    # Proper selection of     
    # =========================================================================
    batch_size_train=2
    batch_size_eval=2
    sub_batch_fact=10

    new_img_h=1080
    new_img_w=1920
    n_crop_h=5
    n_crop_w=9
  
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
    
    main_dir='../../Dataset'
    
    use_cache_img=True
    use_cache_mask=True
 
    train_generator=torch_ViT_DataGenerator(config,'train',
                           batch_size_img=batch_size_train,
                           shuffle=True,
                           bg_augmenter=None,
                           augmenter=None,    
                           label_type_list=['crack','rebar','spall'],
                           rand_lr_flip=True,
                           rand_ud_flip=False,
                           P_aug=0.0,
                           n_crop_h=n_crop_h,
                           n_crop_w=n_crop_w,
                           resize_img=False,
                           new_size=(new_img_w,new_img_h),
                           noisy_crop=True,
                           crop_noise_amp=1,
                           use_cache_img=use_cache_img,
                           use_cache_mask=use_cache_mask,  
                           return_mask=True,
                           return_seg=True,
                           zero_pad=(40,96),
                           filter_bg=True,
                           P_rand_keep_bg=0.0,
                           random_pad=True)
    
    val_generator=torch_ViT_DataGenerator(config,'val',
                               batch_size_img=batch_size_eval,
                               shuffle=False,
                               bg_augmenter=None,
                               augmenter=None,    
                               label_type_list=['crack','rebar','spall'],
                               P_aug=0.0,
                               n_crop_h=n_crop_h,
                               n_crop_w=n_crop_w,
                               resize_img=False,
                               new_size=(new_img_w,new_img_h),
                               noisy_crop=False,
                               crop_noise_amp=1,
                               use_cache_img=use_cache_img,
                               use_cache_mask=use_cache_mask,                               
                               return_mask=True,
                               return_seg=True,
                               zero_pad=(40,96),
                               filter_bg=True)
    
    # test_generator=torch_ViT_DataGenerator(config,'test',
    #                            batch_size_img=batch_size_eval,
    #                            shuffle=False,
    #                            bg_augmenter=None,
    #                            augmenter=None,    
    #                            label_type_list=['crack','rebar','spall'],
    #                            P_aug=0.0,
    #                            n_crop_h=n_crop_h,
    #                            n_crop_w=n_crop_w,
    #                            resize_img=False,
    #                            new_size=(new_img_w,new_img_h),
    #                            noisy_crop=False,
    #                            crop_noise_amp=1,
    #                            use_cache_img=use_cache_img,
    #                            use_cache_mask=use_cache_mask,                                    
    #                            return_mask=True,
    #                            return_seg=True,
    #                            zero_pad=(40,96),
    #                            filter_bg=True)
 
    # X_smaple,Y_sample=train_generator.__getitem__(0)
    # print('X_smaple.size:'+str(X_smaple.size()))
    # print('Y_sample.size:'+str(Y_sample.size()))

    # =============================================================================
    # Load pretrained ViT model
    # =============================================================================

    model_bbone = SwinTransformer(num_classes=3,patch_size=2,
                                  embed_dim=64, depths=[2, 2, 2, 18, 2], num_heads=[4, 4, 8, 16, 32])
    model_bbone.forward_features_bbone = types.MethodType(forward_features_bbone, model_bbone )
    model_bbone.cuda()
    count_parameters(model_bbone)

    model_head=ResDecoderPU(config ,bbone_feat_dim=64*16)
    model_head.cuda()
    model = Segmentor_swin_resd(model_bbone, model_head)
    model.cuda()
    count_parameters(model_head)

    # =============================================================================
    # Training 
    # =============================================================================
    
    # loss_func=torch.nn.BCELoss()
    # loss_func_sum=torch.nn.BCELoss(reduction='sum')
    
    alpha=0.6
    gamma=2.0
    loss_func=FocalLoss(alpha=alpha,gamma=gamma,cross_class_avg=False)
    loss_func_sum=FocalLoss(alpha=alpha,gamma=gamma,reduction='sum',cross_class_avg=False)
    
    optimizer = optim.Adam(model.parameters(), lr=2*1e-4,weight_decay=0.0)
    max_norm=1.0
    # lr schedulers
    
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps = train_generator.n_batch_img*5*sub_batch_fact,
                                                num_training_steps = train_generator.n_batch_img*n_epoch*sub_batch_fact) # source: hugging face # https://huggingface.co/transformers/main_classes/optimizer_schedules.html

    # data logger
    fname_log='swin_log.txt'
    f_log= open(fname_log,"w")
    f_log.close()   
    
    loss_epoch_bin_train=[]    
    loss_epoch_bin_val=[]
    
    acc_epoch_bin_train=[]    
    acc_epoch_bin_val=[]
    
    metric_epoch_bin_val=[]
    
    lr_bin=[]
    count=-1
    patience=0
    
    
    for epoch in range(n_epoch):
        t_start_epoch=time.time()

        loss_step_bin_Y=[]
        
        n_corr_epoch_train=0
        n_total_epoch_train=0
        
        for step,(X_train,Y_patch_train) in enumerate(train_generator):
            
            # break into sub-batches
            sub_batch_indx_bin=np.arange(0,X_train.size(0))
            random.shuffle(sub_batch_indx_bin)
            split_bin=np.array_split(sub_batch_indx_bin,sub_batch_fact)
            for i_sub_b in range(sub_batch_fact):
            # Change batch tensors to cuda tensors
                indx_sub=split_bin[i_sub_b]
                if len(indx_sub)>0:
                    count+=1
                    torch.cuda.synchronize()
                    X_train_sub,Y_patch_train_sub = X_train[indx_sub].cuda(),Y_patch_train[indx_sub].cuda()
                    optimizer.zero_grad()
                    Y_pred_train = torch.sigmoid(model(X_train_sub))
                          
                    loss_Y = loss_func(Y_pred_train, Y_patch_train_sub) 
            
                    loss_Y.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) # Gradient clipping
                    optimizer.step()
                    scheduler.step()
                    
                    loss_step_bin_Y.append(loss_Y.cpu().detach().numpy())
                    n_corr_epoch_train+=torch.sum(torch.round(Y_pred_train)==Y_patch_train_sub).cpu().detach().numpy()
                    n_total_epoch_train+=torch.numel(Y_pred_train)
            
        loss_epoch_bin_train.append(np.mean(loss_step_bin_Y))
        acc_epoch_bin_train.append(n_corr_epoch_train/n_total_epoch_train)
        
        val_loss_epoch,val_acc_epoch,val_metric_dict=torch_eval_ViT_seg(model,val_generator,loss_func_sum,sub_batch_fact=int(sub_batch_fact*0.8))
        loss_epoch_bin_val.append(val_loss_epoch)
        acc_epoch_bin_val.append(val_acc_epoch)
        metric_epoch_bin_val.append(val_metric_dict)

        save_pickle(metric_epoch_bin_val,'metric_epoch_bin_val')
        plot_loss_log_ViT_seg(loss_epoch_bin_train,loss_epoch_bin_val,
                          acc_epoch_bin_train,acc_epoch_bin_val,metric_epoch_bin_val,
                          lr_bin) # loss log plots
        
        if loss_epoch_bin_val[-1]==np.amin(loss_epoch_bin_val):
            torch.save(model.state_dict(), 'DmgFormer_L.pt')
            patience=0
        else:
            patience+=1
        # Get current lr
        current_lr=optimizer.param_groups[0]['lr']
        lr_bin.append(current_lr)
        
        # GPU clean up
        del Y_pred_train

        torch.cuda.empty_cache()
        gc.collect()  
        
        t_end_epoch=time.time()
        t_epoch=(t_end_epoch-t_start_epoch)
    
        print('Epoch:', '%4d' % (epoch + 1),
              'loss=','{:.5f}'.format(loss_epoch_bin_train[-1]),
              'val_loss=','{:.5f}'.format(loss_epoch_bin_val[-1]),
              'acc=','{:.3f}'.format(acc_epoch_bin_train[-1]*100),
              'val_acc=','{:.3f}'.format(acc_epoch_bin_val[-1]*100),
              't=','{:.1f}'.format(t_epoch),
              'patience=','%1d'%(patience),
              'lr_next=','{:.2e}'.format(current_lr))
        # metric bin detailed validation report
            
        print('Validation metric summary(%):')
        print('            Crack Rebar Spall')
        print('Precision   %1.2f %1.2f %1.2f'%(val_metric_dict['Crack']['Precision'][1]*100,val_metric_dict['Rebar']['Precision'][1]*100,val_metric_dict['Spall']['Precision'][1]*100))
        print('Recall      %1.2f %1.2f %1.2f'%(val_metric_dict['Crack']['Recall'][1]*100,val_metric_dict['Rebar']['Recall'][1]*100,val_metric_dict['Spall']['Recall'][1]*100))
        print('F1 score    %1.2f %1.2f %1.2f'%(val_metric_dict['Crack']['F1_score'][1]*100,val_metric_dict['Rebar']['F1_score'][1]*100,val_metric_dict['Spall']['F1_score'][1]*100))
        print('IoU         %1.2f %1.2f %1.2f'%(val_metric_dict['Crack']['IoU'][1]*100,val_metric_dict['Rebar']['IoU'][1]*100,val_metric_dict['Spall']['IoU'][1]*100))
        print('_______________________________________________________________')
        
        f_log= open(fname_log,"a+")
        f_log.write("Epoch: %4d, loss= %1.4e, val_loss= %1.4e, acc= %1.3f, val_acc=%1.3f, t= %1.2f (min)), patience= %d, lr_next=%1.3e\n"%(epoch + 1,
                                                                                                                                    loss_epoch_bin_train[-1],
                                                                                                                                    loss_epoch_bin_val[-1],
                                                                                                                                    acc_epoch_bin_train[-1]*100,
                                                                                                                                    acc_epoch_bin_val[-1]*100,
                                                                                                                                    t_epoch/60,
                                                                                                                                    patience,
                                                                                                                                    current_lr))
        f_log.write('Validation metric summary(%):\n')
        f_log.write('            Crack Rebar Spall\n')
        f_log.write('Precision   %1.2f %1.2f %1.2f\n'%(val_metric_dict['Crack']['Precision'][1]*100,val_metric_dict['Rebar']['Precision'][1]*100,val_metric_dict['Spall']['Precision'][1]*100))
        f_log.write('Recall      %1.2f %1.2f %1.2f\n'%(val_metric_dict['Crack']['Recall'][1]*100,val_metric_dict['Rebar']['Recall'][1]*100,val_metric_dict['Spall']['Recall'][1]*100))
        f_log.write('F1 score    %1.2f %1.2f %1.2f\n'%(val_metric_dict['Crack']['F1_score'][1]*100,val_metric_dict['Rebar']['F1_score'][1]*100,val_metric_dict['Spall']['F1_score'][1]*100))
        f_log.write('IoU         %1.2f %1.2f %1.2f\n'%(val_metric_dict['Crack']['IoU'][1]*100,val_metric_dict['Rebar']['IoU'][1]*100,val_metric_dict['Spall']['IoU'][1]*100))
        f_log.write('_______________________________________________________________\n')
        f_log.close()
        if patience>early_stoping: # stop training
            break




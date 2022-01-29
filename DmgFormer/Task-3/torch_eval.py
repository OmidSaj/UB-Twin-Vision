# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:24:20 2021

@author: Seyed Omid Sajedi """

import torch 
import numpy as np
from OsUtils import make_di_path,wipe_dir
import matplotlib.pyplot as plt
import matplotlib
import gc
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch.nn.functional as F
import torchmetrics
import cv2


def plot_loss_log_ViT_seg(loss_epoch_bin_train,loss_epoch_bin_val,
                      acc_epoch_bin_train,acc_epoch_bin_val,metric_bin_val,
                      lr_bin,fname_init_str=None):
    
    fig,ax=plt.subplots(3,2,figsize=(15,10))
    ax[0,0].plot(loss_epoch_bin_val,'-r',label='val_loss')
    ax[0,0].plot(loss_epoch_bin_train,'-b',label='train_loss')
    # ax.set_xlim(0,n_epoch)
    ax[0,0].set_ylim(0,None)
    ax[0,0].legend()
    ax[0,0].set_xlabel('epoch')
    ax[0,0].set_ylabel('loss')
    
    ax[1,0].plot(acc_epoch_bin_val,'-r',label='val_acc')
    ax[1,0].plot(acc_epoch_bin_train,'-b',label='train_acc')
    # ax.set_xlim(0,n_epoch)
    # ax[1].set_ylim(None,1)
    ax[1,0].legend()
    ax[1,0].set_xlabel('epoch')
    ax[1,0].set_ylabel('global accuracy')   
    
    ax[2,0].plot(lr_bin,'-k')
    ax[2,0].set_xlabel('epoch')
    ax[2,0].set_ylabel('lr')
    ax[2,0].set_ylim(0,None)
    
    # collect metrics for logging
    val_prc_crack=[]
    val_prc_rebar=[]
    val_prc_spall=[]
    
    val_rec_crack=[]
    val_rec_rebar=[]
    val_rec_spall=[]
    
    val_f1_crack=[]
    val_f1_rebar=[]
    val_f1_spall=[]
    
    val_IoU_crack=[]
    val_IoU_rebar=[]
    val_IoU_spall=[]
    
    for i_p, metric_dict_i in enumerate(metric_bin_val):
        
        # index guide epoch, dmg_type, metric, bg vs dmg
        val_prc_crack.append(metric_dict_i['Crack']['Precision'][1])
        val_prc_rebar.append(metric_dict_i['Rebar']['Precision'][1])
        val_prc_spall.append(metric_dict_i['Spall']['Precision'][1])
        
        val_rec_crack.append(metric_dict_i['Crack']['Recall'][1])
        val_rec_rebar.append(metric_dict_i['Rebar']['Recall'][1])
        val_rec_spall.append(metric_dict_i['Spall']['Recall'][1])   
        
        val_f1_crack.append(metric_dict_i['Crack']['F1_score'][1])
        val_f1_rebar.append(metric_dict_i['Rebar']['F1_score'][1])
        val_f1_spall.append(metric_dict_i['Spall']['F1_score'][1])     

        val_IoU_crack.append(metric_dict_i['Crack']['IoU'][1])
        val_IoU_rebar.append(metric_dict_i['Rebar']['IoU'][1])
        val_IoU_spall.append(metric_dict_i['Spall']['IoU'][1])    
    
    ax[0,1].plot(val_prc_crack,'-r',label='crack')
    ax[0,1].plot(val_prc_rebar,'-b',label='rebar')
    ax[0,1].plot(val_prc_spall,'-g',label='spall')
    ax[0,1].set_ylabel('val precision')
    ax[0,1].set_xlabel('epoch')    
    ax[0,1].set_ylim(0,1)
    ax[0,1].legend()

    ax[1,1].plot(val_rec_crack,'-r',label='crack')
    ax[1,1].plot(val_rec_rebar,'-b',label='rebar')
    ax[1,1].plot(val_rec_spall,'-g',label='spall')
    ax[1,1].set_ylabel('val recall')
    ax[1,1].set_xlabel('epoch')    
    ax[1,1].set_ylim(0,1)
    ax[1,1].legend()

    ax[2,1].plot(val_IoU_crack,'-r',label='crack')
    ax[2,1].plot(val_IoU_rebar,'-b',label='rebar')
    ax[2,1].plot(val_IoU_spall,'-g',label='spall')
    ax[2,1].set_ylabel('val IoU')
    ax[2,1].set_xlabel('epoch')    
    ax[2,1].set_ylim(0,1)
    ax[2,1].legend()
    
    fname_save='log_drift.png'
    
    if fname_init_str is not None:
        fname_save=fname_init_str+fname_save
    
    fig.savefig(fname_save,dpi=100)
    plt.close()


def torch_eval_ViT_seg(model,torch_set_dataloaders,loss_sum,label_list=['Crack','Rebar','Spall'],sub_batch_fact=6):
    
    # create a dictionary for torch metrics in each class
    metric_results = {}
    metrics_dict={}
    # label_list=['crack']#['crack','rebar','spall']
    for i,i_lbl in enumerate(label_list):
        met_eval1 = torchmetrics.Accuracy(num_classes=2, average=None, mdmc_average='global')
        met_eval1.__name__ = 'Recall'
              
        met_eval2 = torchmetrics.F1(num_classes=2, average=None, mdmc_average='global')
        met_eval2.__name__ = 'F1_score'
        
        met_eval3 = torchmetrics.IoU(num_classes=2, reduction='none')
        met_eval3.__name__ = 'IoU'
        
        met_eval4 = torchmetrics.Precision(num_classes=2,  average=None, mdmc_average='global')
        met_eval4.__name__ = 'Precision'
        
        # met_eval5 = torchmetrics.Recall(num_classes=2,  average=None, mdmc_average='global')
        # met_eval5.__name__ = 'Recall'
        
        metrics_dict[i_lbl] = [met_eval1.cuda(), met_eval2.cuda(), met_eval3.cuda(), met_eval4.cuda()]
        
    total_loss_sum=0
    n_total_loss_elem=0
    n_corr=0
    
    model.eval()
    with torch.no_grad():
        # Change batch tensors to cuda tensors
        for step,(X,Y_patch) in tqdm(enumerate(torch_set_dataloaders),total=torch_set_dataloaders.__len__(),
                                                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            
            # break into sub-batches
            sub_batch_indx_bin=np.arange(0,X.size(0))
            split_bin=np.array_split(sub_batch_indx_bin,sub_batch_fact)
            for i_sub_b in range(sub_batch_fact):
                # print(step)
                torch.cuda.synchronize()
                X_i,Y_true_i = X[0][split_bin[i_sub_b]].cuda(),Y_patch[0][split_bin[i_sub_b]].cuda()
    
                Y_pred_i = torch.sigmoid(model(X_i))
                total_loss_sum+=loss_sum(Y_pred_i,Y_true_i).cpu().detach().numpy()
                n_total_loss_elem+=torch.numel(Y_pred_i)
                Y_pred_i=torch.round(Y_pred_i).int()
                n_corr+=torch.sum(Y_pred_i==Y_true_i).cpu().detach().numpy()
                
                for i,i_lbl in enumerate(label_list):
                    pr=Y_pred_i[:,:,:,i]
                    gt=Y_true_i[:,:,:,i].int()    
                    for metric in metrics_dict[i_lbl]:
                        metric(pr, gt)

    for i,i_lbl in enumerate(label_list):
        metric_results[i_lbl]={}
        for metric in metrics_dict[i_lbl]:
            metric_results[i_lbl][metric.__name__] = metric.compute().cpu().tolist()
            metric.reset()

    # print(metric_bin)    
    del X_i
    del Y_pred_i
    del Y_true_i
    del gt
    del pr
    
    torch.cuda.empty_cache()
    gc.collect()
    loss_avg=np.sum(total_loss_sum)/n_total_loss_elem
    acc=n_corr/n_total_loss_elem
    # print(acc)
    model.train()
    return loss_avg,acc,metric_results

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

def torch_eval_ViT_seg_aug(model,torch_set_dataloaders,
                           label_list=['Crack','Rebar','Spall'],
                           n_crop_h=5,
                           n_crop_w=9,
                           new_img_h = 1080,
                           new_img_w = 1920,
                           h_pad_list=[0], #[0,1]
                           v_pad_list=[0],  #[0,1]
                           flip_h_list=[0]):
    
    # create a dictionary for torch metrics in each class
    metric_results = {}
    metrics_dict={}
    # label_list=['crack']#['crack','rebar','spall']
    for i,i_lbl in enumerate(label_list):
        met_eval1 = torchmetrics.Accuracy(num_classes=2, average=None, mdmc_average='global',multiclass=True)
        met_eval1.__name__ = 'Recall'
              
        met_eval2 = torchmetrics.F1(num_classes=2, average=None, mdmc_average='global',multiclass=True)
        met_eval2.__name__ = 'F1_score'
        
        met_eval3 = torchmetrics.IoU(num_classes=2, reduction='none')
        met_eval3.__name__ = 'IoU'
        
        met_eval4 = torchmetrics.Precision(num_classes=2,  average=None, mdmc_average='global',multiclass=True)
        met_eval4.__name__ = 'Precision'
        
        
        metrics_dict[i_lbl] = [met_eval1.cuda(), met_eval2.cuda(), met_eval3.cuda(), met_eval4.cuda()]
        
    model.eval()
    with torch.no_grad():

        n_img_plot=torch_set_dataloaders.__len__()
        config = torch_set_dataloaders.config
        for i_img in tqdm(range(n_img_plot),total=torch_set_dataloaders.__len__(),
                                                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                        
            Y_pred_prob=[]
            for flip_h_i in flip_h_list:
                for h_had_i in h_pad_list:
                    for v_had_i in v_pad_list:
                        X_smaple,Y_sample=torch_set_dataloaders.__getitem__(i_img,return_seg_mask=False,pad_h=h_had_i,pad_v=v_had_i)
                        if flip_h_i==1:
                            X_smaple = torch.flip(X_smaple,[3])
                        Y_pred_prob_i=torch.sigmoid(model(X_smaple.cuda()))
                        # print(X_smaple.size())
                        if flip_h_i==1:
                            Y_pred_prob_i = torch.flip(Y_pred_prob_i,[2])
                            X_smaple = torch.flip(X_smaple,[3])
                    
                        Y_pred_prob_i=Y_pred_prob_i.reshape(Y_pred_prob_i.shape[0],config.img_size,config.img_size,Y_pred_prob_i.shape[-1])
                        Y_pred_prob_i=Y_pred_prob_i.reshape(n_crop_h,n_crop_w,Y_pred_prob_i.shape[1],Y_pred_prob_i.shape[2],Y_pred_prob_i.shape[-1]).swapaxes(1,2)
                        Y_pred_prob_i=Y_pred_prob_i.reshape(new_img_h+torch_set_dataloaders.zero_pad[0],new_img_w+torch_set_dataloaders.zero_pad[1],Y_pred_prob_i.shape[-1])
                        
                        Y_pred_prob_i=unpad_tensor(Y_pred_prob_i,zero_pad=torch_set_dataloaders.zero_pad,pad_h=h_had_i,pad_v=v_had_i)
                        
                        Y_pred_prob.append(Y_pred_prob_i.cpu().detach().numpy())            
        
                Y_pred_prob=np.mean(np.stack(Y_pred_prob),axis=0)

                Y_sample=Y_sample.numpy()
                Y_sample=Y_sample.reshape(Y_sample.shape[0],config.img_size,config.img_size,Y_sample.shape[-1])
                Y_sample=Y_sample.reshape(n_crop_h,n_crop_w,Y_sample.shape[1],Y_sample.shape[2],Y_sample.shape[-1]).swapaxes(1,2)
                Y_sample=Y_sample.reshape(new_img_h+torch_set_dataloaders.zero_pad[0],new_img_w+torch_set_dataloaders.zero_pad[1],Y_sample.shape[-1])
                Y_sample=unpad_tensor(Y_sample,zero_pad=torch_set_dataloaders.zero_pad,pad_h=h_had_i,pad_v=v_had_i)
                
                Y_pred=np.round(Y_pred_prob)       
                
                Y_pred = torch.from_numpy(Y_pred).cuda()
                Y_sample = torch.from_numpy(Y_sample).cuda()

                for i,i_lbl in enumerate(label_list):
                    pr=Y_pred[:,:,i].reshape(-1,)
                    gt=Y_sample[:,:,i].reshape(-1,).int()    
                    for metric in metrics_dict[i_lbl]:
                        metric(pr, gt)

    for i,i_lbl in enumerate(label_list):
        metric_results[i_lbl]={}
        for metric in metrics_dict[i_lbl]:
            metric_results[i_lbl][metric.__name__] = metric.compute().cpu().tolist()
            metric.reset()


    del X_smaple
    del Y_pred
    del Y_sample
    del gt
    del pr
    
    torch.cuda.empty_cache()
    gc.collect()
    model.train()
    return metric_results
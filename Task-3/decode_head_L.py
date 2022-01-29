# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:13:47 2021

@author: SSajedi
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class ResDecoderPU(nn.Module):
    def __init__(self,config,n_classes=3,bbone_feat_dim=1024):
        super(ResDecoderPU, self).__init__()    
        
        bn_momentum = 0.1
        
        self.config=config
        self.n_classes=n_classes
        self.n1=bbone_feat_dim     # 768  # 1024
        self.n2=bbone_feat_dim//2  # 384  # 512 
        self.n3=bbone_feat_dim//4  # 192  # 256 
        self.n4=bbone_feat_dim//8  # 96   # 128 
        self.n5=bbone_feat_dim//16 # 48   # 64
        self.n6=bbone_feat_dim//32 # 24   # 32
        
        self.nb1=self.n1
        self.nb2=self.nb1//4+self.n2 # 1024//4 + 512 = 768
        self.nb3=self.nb2//4+self.n3 # 768//4 + 256 = 448
        self.nb4=self.nb3//4+self.n4 # 448//4 + 128  = 240
        self.nb5=self.nb4//4+self.n5 # 240//4 + 64  = 124
        self.nb6=self.n6             # -------------> 32
        
        self.act=nn.GELU()
        
        # 1st block
        self.cn11 = nn.Conv2d(self.nb1, self.nb1, kernel_size=3,padding='same')
        self.bn11 = nn.BatchNorm2d(self.nb1, momentum= bn_momentum)
        self.cn12 = nn.Conv2d(self.nb1, self.nb1, kernel_size=3,padding='same')
        self.bn12 = nn.BatchNorm2d(self.nb1, momentum= bn_momentum)        
        self.cn13 = nn.Conv2d(self.nb1, self.nb1, kernel_size=3,padding='same')
        self.bn13 = nn.BatchNorm2d(self.nb1, momentum= bn_momentum)             
        # self.dcn1 = nn.ConvTranspose2d(self.nb1, self.nb1, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.pshf1 = nn.PixelShuffle(2)
        # 2nd block
        self.cn21 = nn.Conv2d(self.nb2, self.nb2, kernel_size=3,padding='same')
        self.bn21 = nn.BatchNorm2d(self.nb2, momentum= bn_momentum)
        self.cn22=nn.Conv2d(self.nb2, self.nb2, kernel_size=3,padding='same')
        self.bn22 = nn.BatchNorm2d(self.nb2, momentum= bn_momentum)        
        self.cn23=nn.Conv2d(self.nb2, self.nb2, kernel_size=3,padding='same')
        self.bn23 = nn.BatchNorm2d(self.nb2, momentum= bn_momentum)             
        # self.dcn2=nn.ConvTranspose2d(self.nb2, self.nb2, kernel_size=3, stride=2,padding=1,output_padding=1)     
        self.pshf2 = nn.PixelShuffle(2)
        
        # 3rd block
        self.cn31 = nn.Conv2d(self.nb3, self.nb3, kernel_size=3,padding='same')
        self.bn31 = nn.BatchNorm2d(self.nb3, momentum= bn_momentum)
        self.cn32=nn.Conv2d(self.nb3, self.nb3, kernel_size=3,padding='same')
        self.bn32 = nn.BatchNorm2d(self.nb3, momentum= bn_momentum)        
        self.cn33=nn.Conv2d(self.nb3, self.nb3, kernel_size=3,padding='same')
        self.bn33 = nn.BatchNorm2d(self.nb3, momentum= bn_momentum)             
        # self.dcn3=nn.ConvTranspose2d(self.nb3, self.nb3, kernel_size=3, stride=2,padding=1,output_padding=1) 
        self.pshf3 = nn.PixelShuffle(2)
        
        # 4th block
        self.cn41 = nn.Conv2d(self.nb4, self.nb4, kernel_size=3,padding='same')
        self.bn41 = nn.BatchNorm2d(self.nb4, momentum= bn_momentum)
        self.cn42=nn.Conv2d(self.nb4, self.nb4, kernel_size=3,padding='same')
        self.bn42 = nn.BatchNorm2d(self.nb4, momentum= bn_momentum)        
        self.cn43=nn.Conv2d(self.nb4, self.nb4, kernel_size=1,padding='same')
        self.bn43 = nn.BatchNorm2d(self.nb4, momentum= bn_momentum)   
        # self.dcn4=nn.ConvTranspose2d(self.nb4, self.nb4, kernel_size=3, stride=2,padding=1,output_padding=1)    
        self.pshf4 = nn.PixelShuffle(2)
        
        # 5th block
        self.cn51 = nn.Conv2d(self.nb5, self.nb5, kernel_size=3,padding='same')
        self.bn51 = nn.BatchNorm2d(self.nb5, momentum= bn_momentum)
        self.cn52=nn.Conv2d(self.nb5, self.nb5, kernel_size=3,padding='same')
        self.bn52 = nn.BatchNorm2d(self.nb5, momentum= bn_momentum)        
        self.cn53=nn.Conv2d(self.nb5, self.nb5, kernel_size=1,padding='same')
        self.bn53 = nn.BatchNorm2d(self.nb5, momentum= bn_momentum)   
        # self.dcn5=nn.ConvTranspose2d(self.nb5, self.nb5, kernel_size=3, stride=2,padding=1,output_padding=1)   
        self.pshf5 = nn.PixelShuffle(2)
        
        # 6th block
        self.cn61 = nn.Conv2d(self.nb6-1, self.nb6, kernel_size=3,padding='same')
        self.bn61 = nn.BatchNorm2d(self.nb6, momentum= bn_momentum)
        self.cn62 = nn.Conv2d(self.nb6, self.n_classes, kernel_size=3,padding='same')
        self.bn62 = nn.BatchNorm2d(self.n_classes, momentum= bn_momentum)        

        
    def forward(self, conv_outs):
        
        x = self.act(self.bn11(self.cn11(conv_outs[-1])))
        x = self.act(self.bn12(self.cn12(x)))
        x = self.act(self.bn13(self.cn13(x)))
        x = self.pshf1(x)
        # x = self.dcn1(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = torch.cat((x,conv_outs[-2]),1)
        x = self.act(self.bn21(self.cn21(x)))
        x = self.act(self.bn22(self.cn22(x)))
        x = self.act(self.bn23(self.cn23(x)))
        x = self.pshf2(x)
        # x = self.dcn2(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = torch.cat((x,conv_outs[-3]),1)
        x = self.act(self.bn31(self.cn31(x)))
        x = self.act(self.bn32(self.cn32(x)))
        x = self.act(self.bn33(self.cn33(x)))
        x = self.pshf3(x)
        # x = self.dcn3(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = torch.cat((x,conv_outs[-4]),1)
        x = self.act(self.bn41(self.cn41(x)))
        x = self.act(self.bn42(self.cn42(x)))
        x = self.act(self.bn43(self.cn42(x)))
        x = self.pshf4(x)
        # x = self.dcn4(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')   
        
        x = torch.cat((x,conv_outs[-5]),1)
        x = self.act(self.bn51(self.cn51(x)))
        x = self.act(self.bn52(self.cn52(x)))
        x = self.act(self.bn53(self.cn52(x)))
        x = self.pshf5(x)
        # x = self.dcn5(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')  
        
        x = self.act(self.bn61(self.cn61(x)))
        x = self.cn62(x) # self.act(self.bn52(self.cn52(x)))
        x = x.swapaxes(1,3).swapaxes(1,2)
        # x = x.view(x.size(0),-1,self.n_classes)
        return x
    
class Segmentor_swin_resd(nn.Module):
    def __init__(self,model_bbone,model_head):
        super(Segmentor_swin_resd, self).__init__()
        
        self.model_bbone=model_bbone
        self.model_head=model_head
        
    def forward(self, X):

        tr_feat_outs=self.model_bbone.forward_features_bbone(X)
        # print(len(conv_outs))
        # for layer_out in conv_outs:
        #     print(layer_out.size())

        Y_pred=self.model_head(tr_feat_outs)

        return Y_pred
    
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:12:05 2021

@author: Seyed Omid Sajedi 

Most of the modules in this code are adopted from:
    https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
    and 
"""

import math
# import re
# from random import *
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import copy
# =============================================================================
# Model confing
# =============================================================================
class Confing:
    def __init__(self,
             np_h:int,
             np_w:int,
             img_size:int=60,
             n_classes:int=3,
             n_layers:int=3,
             n_heads:int=8,
             hidden_size=768,
             mlp_size=3072,
             P_do_dense:float=0.0,
             P_do_attn:float=0.0,
             P_do_head:float=0.0
             ):
        """
        Initiates the transformer network architecture 

        Parameters
        ----------
        np_h : int
            Number of patches along the height of image. Make sure this value 
            is divisible by the image height
        np_w : int
            Number of patches along the width of image. Make sure this value 
            is divisible by the image width
        img_h : int, optional
            image pixel height.The default is 1080.
        img_w : int, optional
            image pixel width.The default is 1920.
        n_classes : int, optional
            Number of output patch classes. The default is 3 (spall,crack, rebar).
        n_layers : int, optional
            Number of transformer (encoder) layers. The default is 3.
        n_heads : int, optional
            Number of heads in the multi-head attention module. The default is 8.
        n_dff : int, optional
            Ratio of the hidden units in the dense layers of the feed forward 
            layer to the flattened patch size. The default is 2.
        P_do_dense : float, optional
            Dropout probability after dense layers in the attention module. The 
            default is 0.0.
        P_do_attn : float, optional
            Dropout probability after the attention output. The default is 0.0.
        P_do_head : float, optional
            Dropout probability of dense layers in the network head. The default 
            is 0.0.

        Returns
        -------
        None.
    
        """
        self.img_size=img_size
        self.np_h=np_h
        self.np_w=np_w
        self.n_classes=n_classes
        self.n_layers=n_layers
        self.n_heads=n_heads
        self.hidden_size=hidden_size
        self.mlp_size=mlp_size
        self.P_do_dense=P_do_dense
        self.P_do_head=P_do_head
        self.P_do_attn=P_do_attn
        self.n_patch=int(np_h*np_w)  # number of patches 
        self.n_token=self.n_patch+1
        self.patch_size_2d=(int(img_size/np_h),int(img_size/np_w))
        self.flat_patch_size=int(self.img_size*self.img_size/self.n_patch) # patch dimensions
        self.d_k=int(self.hidden_size/self.n_heads) # attention key size
        self.d_v=int(self.hidden_size/self.n_heads) # attention value size
        

# =============================================================================
# 
# =============================================================================

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.n_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.P_do_attn)
        self.proj_dropout = Dropout(config.P_do_attn)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_size)
        self.fc2 = Linear(config.mlp_size, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.P_do_dense)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()

        img_size = _pair(img_size)

        patch_size = _pair(config.patch_size_2d)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])


        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.P_do_dense)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.n_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, num_classes=1, zero_head=False, vis=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'token' #config.classifier

        self.transformer = Transformer(config, config.img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

class ViT_bbone(nn.Module):
    def __init__(self, config, num_classes=1, zero_head=False, vis=True):
        super(ViT_bbone, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'token' #config.classifier

        self.transformer = Transformer(config, config.img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
        
    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights
        
    def forward_feat(self, x):
        x, attn_weights = self.transformer(x)
        return x

# =============================================================================
# Utility 
# =============================================================================
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    # for item in params:
    #     print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
    
# =============================================================================
# Focal loss 
# Ref: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/2
# =============================================================================
class FocalLoss(nn.Module):

    def __init__(self, gamma = 2.0, alpha=0.25,reduction='mean',cross_class_avg=False):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype = torch.float32).cuda()
        self.alpha =alpha
        self.reduction=reduction
        self.cross_class_avg=cross_class_avg

    def forward(self, inputs, targets):
        # input are not the probabilities, they are just the cnn out vector
        # input and target shape: (bs, n_classes)
        # sigmoid
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        
        alpha_t=(1-self.alpha)*(1-targets)+self.alpha*targets
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        
        # custom weighing between the three losses
        # F_loss[:,:,0]=F_loss[:,:,0]/32.96  #*2
        # F_loss[:,:,1]=F_loss[:,:,1]/0.65   #*2
        # F_loss[:,:,2]=F_loss[:,:,2]/31.14  #*2
        
        if self.cross_class_avg:
            F_loss=torch.mean(F_loss,axis=-1)
        
        if self.reduction=='mean':
            F_loss=F_loss.mean()
        elif self.reduction=='sum':
            F_loss=F_loss.sum()
        else:
            print('invalid reduction argument')
        return F_loss

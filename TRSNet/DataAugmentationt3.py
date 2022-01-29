# -*- coding: utf-8 -*-

"""
Created on Thu Sep 23 14:49:40 2021
@author: Kareem A Eltouny """




import numpy as np
import random
import cv2
import albumentations as A


def CreateAugmenter(image_height, image_width , aug_p_edit={}, options_edit={}):


    """  Used to generate an augmenter from albumentations library, this augmenter is made of a composition of augmenter functions.
        
        
        # NOTE: GridDistortion is disabeled due to nonrealistic outcomes
        
        Arguments:
        ----------
            image_height : <int> original image height
            
            image_width : <int> original image width
            
            aug_p_edit : <dict>, optional
                The dictionary includes keywords and values for the probabilities of each augmenter function
                Each value is a <float> between 0 and 1, where 0 completely disables the transform function and 1 
                guarantees it's occurence, if it did not fall in a OneOf function.
                Keyword : default value :
                    'HorizontalFlip':0.5, 'RandomSizedCrop':0.5, 'GridDistortion':0.5, 'RandomGamma':0.5, 
                    'ColorJitter':0.5, 'RandomToneCurve':0.5, 'CoarseDropout':0.5
            options: <dict>, optionals
                The dictionary includes keywords and values for various options for the augmenter functions
                For each option limits, refer to the albumentations documentation.
                Keyword : default value :
                    'w2h_ratio': image_width/image_height, 'crop_min_max_height': (int(image_height/2), image_height), 
                    'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.1,  'gamma_limit':(70, 130), 'random_tone_scale':0.2,
                    'drop_max_height':50, 'drop_max_width':50, 'drop_min_holes':4, 'drop_min_height':8, 'drop_min_width':8, 
                    'drop_fill_value':120, 'drop_mask_fill_value':0 """

    aug_p={'HorizontalFlip':0.5, 'RandomSizedCrop':0.5, 'GridDistortion':0.5, 'RandomGamma':0.5, 'ColorJitter':0.5, 'RandomToneCurve':0.5, 'CoarseDropout':0.5}

    options = {'w2h_ratio': image_width/image_height, 'crop_min_max_height': (int(image_height/2), image_height), 
                'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.1,  'gamma_limit':(70, 130), 'random_tone_scale':0.2,
                'drop_max_height':50, 'drop_max_width':50, 'drop_min_holes':4, 'drop_min_height':8, 'drop_min_width':8, 'drop_fill_value':120, 'drop_mask_fill_value':0}
                

    aug_p.update(aug_p_edit)
    options.update(options_edit)

    augmenter = A.Compose([
            A.HorizontalFlip(p=aug_p['HorizontalFlip']),
            A.OneOf([
                A.RandomSizedCrop(min_max_height=options['crop_min_max_height'], height=image_height, w2h_ratio=options['w2h_ratio'], width=image_width, p=aug_p['RandomSizedCrop']),
                #A.GridDistortion(p=aug_p['GridDistortion'])
            ], p=1),
            A.OneOf([
                A.Compose([
                    A.RandomGamma(gamma_limit=options['gamma_limit'], p=aug_p['RandomGamma']),
                    A.ColorJitter(brightness=options['brightness'], contrast=options['contrast'], saturation=options['saturation'], hue=options['hue'], p=aug_p['ColorJitter']),
                ], p=1),
                A.RandomToneCurve(scale=options['random_tone_scale'], p=aug_p['RandomToneCurve'])
            ], p=1),
            # A.CoarseDropout(max_height=options['drop_max_height'], max_width=options['drop_max_width'], min_holes=options['drop_min_holes'],
                            # min_height=options['drop_min_height'], min_width=options['drop_min_width'], fill_value=options['drop_fill_value'], mask_fill_value=options['drop_mask_fill_value'], p=aug_p['CoarseDropout']),
        ], p=0.7)
        

    return augmenter
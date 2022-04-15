# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:23:57 2022

@author: AnushKolakalur
"""

import cv2 
import numpy as np
import glob
import os

import glob
import os
import tifffile as tff
from matplotlib import pyplot as plt

import random
from skimage import io
import shutil

class drconfig():
    def __init__(self):
    
    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # # Saving and loding ML model path 
    # =============================================================================
    # =============================================================================
    # =============================================================================
            
        self.LR = 0.01 #learning rate
        self.MD = 20   #Max-depth
        self.E = 400   #number of estimators 
        #self.g = 2
        #LR = 'default'
        self.SavedModel_path = "C:\\Users\\AnushKolakalur\\Azure Repository\\Autofocus data\\Saved models\\"
        self.SavedModel_name = 'Test_model[LR-'+str(self.LR)+'_MD-'+str(self.MD)+'_Est-'+str(self.E)+'].sav'#'_Gamma-'+str(self.g)+'].sav'
        #print(SavedModel_name)

# =============================================================================
# =============================================================================
# =============================================================================
# # # Covert grayscale images to RGB
# =============================================================================
# =============================================================================
# =============================================================================

    def gry2rgb(self,images):
        
        shape = np.shape(images)
        n_images =[]
        images = images.astype('float32')
        
        l = shape[0]
        for llen in range(0,l):
            img = cv2.cvtColor(images[llen],cv2.COLOR_GRAY2BGR) 
            img = cv2.resize(img,(32,32),cv2.INTER_NEAREST)
            n_images.append(img)
            
        new_images = np.array(n_images)
        return new_images





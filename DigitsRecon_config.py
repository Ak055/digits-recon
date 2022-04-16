import cv2 
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

import seaborn as sns
import pickle
from tensorflow.keras.applications.vgg16 import VGG16


from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np



class drconfig():
    def __init__(self):
    
    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # # Saving and loding ML model path 
    # =============================================================================
    # =============================================================================
    # =============================================================================
        self.SavedModel_path = "C:\\Users\\AnushKolakalur\\Github reporsitory\\digits-recon\\Saved models\\"
        self.SavedModel_name = 'Model[default_params].sav'
        #print(SavedModel_name)

    # =============================================================================
    # =============================================================================
    # # load datasets from scikit-learn
    # =============================================================================
    # =============================================================================
        self.digits = load_digits()
        
    # =============================================================================
    # =============================================================================
    # # Load model without classifier/fully connected layers
    # =============================================================================
    # =============================================================================
        self.VGG_net = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
    
        self.VGG_net_mod = Model(inputs=self.VGG_net.input, outputs=self.VGG_net.get_layer(
            'block1_conv2').output)
        #self.VGG_net_mod.summary()
            
    # =============================================================================
    # =============================================================================
    # # split the images for training and testing 
    # =============================================================================
    # =============================================================================
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(
        self.digits.images, self.digits.target, test_size=0.2, shuffle=False)

# =============================================================================
# =============================================================================
# =============================================================================
# # # Covert grayscale images to RGB
# =============================================================================
# =============================================================================
# =============================================================================

    def prep4VGGn(self,images):
        images = images.astype('float32')
        size = images.shape      
        if len(size)==2:
            o_img = cv2.cvtColor(images,cv2.COLOR_GRAY2BGR) 
            o_img = cv2.resize(o_img,(32,32),cv2.INTER_NEAREST)
            o_img = np.expand_dims(o_img, axis=0) 
            return o_img
        
        shape = np.shape(images)
        n_images =[]
        l = shape[0]
        for llen in range(0,l):
            img = cv2.cvtColor(images[llen],cv2.COLOR_GRAY2BGR) 
            img = cv2.resize(img,(32,32),cv2.INTER_NEAREST)
            n_images.append(img)
            
        new_images = np.array(n_images)
        return new_images

    def fordisp(self,images):
        im = cv2.resize(images,(8,8),cv2.INTER_LINEAR)
        return im
    




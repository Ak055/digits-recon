# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2022

@author: AnushKolakalur
"""
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

from DigitsRecon_config import drconfig
dr = drconfig()

import time
start = time.perf_counter()


t = dr.prep4VGGn(dr.X_test[56])
plt.imshow(dr.X_test[56],cmap='gray')
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
model.summary()

t_feature=model.predict(t)
t_features = t_feature.reshape(
    t_feature.shape[0], -1)


#Load model.... 
filename = dr.SavedModel_name
PathToSaved_model=dr.SavedModel_path
model = pickle.load(open(PathToSaved_model+filename, 'rb'))

prediction = model.predict(t_features)[0] 



recon_folder_list = glob.glob(os.path.join(PathToTest_images,"*"))
ReconL = len(recon_folder_list)


for r in range(0,ReconL):
    
    newim =[]
    t_images = []
    name_list_4_plt =[]
    img_list_4_plt =[]
    
    for directory_path in glob.glob(recon_folder_list[r]):     #os.path.join(PathToTest_images,"*")):
        
        for img_path in glob.glob(os.path.join(directory_path, "*.tiff")):
            
            img_name_list = glob.glob(directory_path + os.sep + '*')
            #print(img_name_list)
            img = io.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            t_images.append(img)
            
    
    #Convert lists to arrays                
    t_images = np.array(t_images)
    t_img = t_images/255.0
    newim = img_name_list[0]
    newim = newim.split(os.sep)[-1]
    newim = newim.split('_recon',2)
    L=len(t_img)
    
    
    #n=np.random.randint(0, t_img.shape[0])
    for n in range(0,L):
        
        img = t_img[n]
        input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
        input_img_feature=VGG_model.predict(input_img)
        input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
        prediction = model.predict(input_img_features)[0] 
        if prediction ==0:
            prediction = 'Not suitable'
# =============================================================================
#             name = img_name_list[n]
#             img_name = name.split(os.sep)[-1]
#             img_name = img_name.replace(".png","")
#             plt.figure()
#             plt.title(str(prediction)+"\n \n"+str(img_name))
#             plt.imshow(img)
#             plt.axis('off')
# =============================================================================
            
            
        else:
            
            prediction = 'Suitable'
            name = img_name_list[n]
            name_list_4_plt.append(name)
            img_name = name.split(os.sep)[-1]
            img_name = img_name.replace(".tiff","")
            plt.figure()
            plt.title(str(prediction)+"\n \n"+str(img_name))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #rotation angle in degree
            img = np.fliplr(img)
            img = ndimage.rotate(img, 180)
            img_list_4_plt.append(img)
            plt.imshow(img,cmap='gray')
            plt.axis('off')
            
    num_of_figs = len(img_list_4_plt)
    if num_of_figs >= 2:
        num = 2
    else:
        num = num_of_figs    
    af.plot_all_in_1_fig(name_list_4_plt, img_list_4_plt,num,newim)#num_of_figs,newim) 
    print("\n Number of suitable recons for "+str(newim[0])+ " are "+str(num_of_figs)+"\n")
    
    
print("Execution time = ",time.perf_counter() - start)





t = dr.prep4VGGn(dr.X_test[4])


























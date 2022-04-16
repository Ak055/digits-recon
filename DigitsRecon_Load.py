# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2022

@author: AnushKolakalur
"""
import pickle
import matplotlib.pyplot as plt
from PIL import Image 
import cv2

from DigitsRecon_config import drconfig
dr = drconfig()

num = input("Enter a number in between 1 to 360, please \n")
num = int(num)

test_image = dr.prep4VGGn(dr.X_test[num])
dis_im = dr.fordisp(dr.X_test[num])
plt.figure()
plt.imshow(dis_im)
plt.show
# =============================================================================
# cv2.imshow('Image',plt.imshow(dis_im))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

test_image_feature=dr.VGG_net_mod.predict(test_image)
test_image_features = test_image_feature.reshape(
    test_image_feature.shape[0], -1)


#Load model.... 
filename = dr.SavedModel_name
PathToSaved_model=dr.SavedModel_path
model = pickle.load(open(PathToSaved_model+filename, 'rb'))

prediction = model.predict(test_image_features)[0] 

print("The digit is = ",prediction)























# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2022

@author: AnushKolakalur
"""
from DigitsRecon_config import drconfig
dr = drconfig()

num = input("Enter a number in between 1 to 360, please \n")
num = int(num)

test_image = dr.prep4VGGn(dr.X_test[num])
dr.fordisp(dr.X_test[num])

test_image_feature=dr.VGG_net_mod.predict(test_image)
test_image_features = test_image_feature.reshape(
    test_image_feature.shape[0], -1)

Trained_model = dr.load_trained_model()

prediction = Trained_model.predict(test_image_features)[0] 

print("The digit is = ",prediction)
























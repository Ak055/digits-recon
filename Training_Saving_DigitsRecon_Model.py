# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2022

@author: AnushKolakalur
"""

# =============================================================================
#  XGBoost for image classification
#  pretrained weights (VGG16) as feature extractors.
# =============================================================================
    
# =============================================================================
#  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#                importance_type='gain', interaction_constraints='',
#                learning_rate=0.300000012, max_delta_step=0, max_depth=6,
#                min_child_weight=1, missing=nan, monotone_constraints='()',
#                n_estimators=100, n_jobs=0, num_parallel_tree=1,
#                objective='multi:softprob', random_state=0, reg_alpha=0,
#                reg_lambda=1, scale_pos_weight=None, subsample=1,
#                tree_method='exact', validate_parameters=1, verbosity=None)   
# =============================================================================


from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

import seaborn as sns
import pickle
from tensorflow.keras.applications.vgg16 import VGG16


from DigitsRecon_config import drconfig
dr = drconfig()

from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# =============================================================================
# =============================================================================
# # importing the conventional ML classifier algorithm
# =============================================================================
# =============================================================================
import xgboost as xgb


# =============================================================================
# =============================================================================
# # load datasets from scikit-learn
# =============================================================================
# =============================================================================
from sklearn.datasets import load_digits
digits = load_digits()


# =============================================================================
# =============================================================================
# # split the images for training and testing 
# =============================================================================
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    digits.images, digits.target, test_size=0.2, shuffle=False)


# =============================================================================
# =============================================================================
# # using a function to convert gray to RGB
# =============================================================================
# =============================================================================
X_Train = dr.prep4VGGn(X_train)
X_Test = dr.prep4VGGn(X_test)


# =============================================================================
# =============================================================================
# # Load model without classifier/fully connected layers
# =============================================================================
# =============================================================================
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
model.summary()


# =============================================================================
# =============================================================================
# # Use the Convolutinal neural networks layers as automatic feature engineering
# =============================================================================
# =============================================================================
feature_extractor_train=model.predict(X_Train)
feature_extractor_test=model.predict(X_Test)

X_for_training  = feature_extractor.reshape(feature_extractor.shape[0], -1)

x_test_feature = model.predict(X_Test)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)

# =============================================================================
# =============================================================================
# # fitting the model using the conventional ML classifier 
# =============================================================================
# =============================================================================
model = xgb.XGBClassifier()

model.fit(X_for_training, y_train) 



# =============================================================================
# =============================================================================
# # Now predict using the trained 
# =============================================================================
# =============================================================================


prediction = model.predict(x_test_features)



# =============================================================================
# =============================================================================
# # print overall accuracy and confusion matrix 
# =============================================================================
# =============================================================================
from sklearn import metrics
acc = metrics.accuracy_score(y_test, prediction)
acc = acc*100
acc = round(acc,3)
print ("Validation Accuracy = ",acc)


# =============================================================================
# =============================================================================
# # Confusion Matrix 
# =============================================================================
# =============================================================================
from sklearn.metrics import confusion_matrix
import pylab as pl

cm = confusion_matrix(y_test, prediction)
ax = plt.axes()
sns.heatmap(cm, annot=True,cbar=False)
ax.set_title("Model accuracy is approx. "+str(acc)+"%")

plt.show()



# =============================================================================
# =============================================================================
# # Save model for future
# =============================================================================
# =============================================================================
#PathToSave_model=dr.SavedModel_path
#filename = dr.SavedModel_name
#savemodel_name = filename
#pickle.dump(model, open(PathToSave_model+filename, 'wb'))


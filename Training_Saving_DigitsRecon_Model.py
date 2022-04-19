# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2022

@author: AnushKolakalur
"""
# =============================================================================
#  XGBoost for image classification
#  pretrained weights (VGG16) as feature extractors.
# =============================================================================
    
from DigitsRecon_config import drconfig
dr = drconfig()

# =============================================================================
# =============================================================================
# # using a function to prepare training and testing imagesets 
# # to use VGGnet's Convolutional network layers
# =============================================================================
# =============================================================================
X_Train = dr.prep4VGGn(dr.X_train)
X_Test = dr.prep4VGGn(dr.X_test)

# =============================================================================
# =============================================================================
# # Use the Convolutinal neural networks layers 
# # as automatic feature engineering to engineer features foe both train and
# # test imagesets
# =============================================================================
# =============================================================================
feature_extractor_train=dr.VGG_net_mod.predict(X_Train)
feature_extractor_test=dr.VGG_net_mod.predict(X_Test)

X_train_features  = feature_extractor_train.reshape(
    feature_extractor_train.shape[0], -1)
x_test_features = feature_extractor_test.reshape(
    feature_extractor_test.shape[0], -1)
  
# =============================================================================
# =============================================================================
# #  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
# #                colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
# #                importance_type='gain', interaction_constraints='',
# #                learning_rate=0.300000012, max_delta_step=0, max_depth=6,
# #                min_child_weight=1, missing=nan, monotone_constraints='()',
# #                n_estimators=100, n_jobs=0, num_parallel_tree=1,
# #                objective='multi:softprob', random_state=0, reg_alpha=0,
# #                reg_lambda=1, scale_pos_weight=None, subsample=1,
# #                tree_method='exact', validate_parameters=1, verbosity=None)   
# =============================================================================
# =============================================================================
import xgboost as xgb
model = xgb.XGBClassifier() #  Fitting the model using the conventional 
                            #  ML classifier 
model.fit(X_train_features, dr.y_train) 

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
acc = metrics.accuracy_score(dr.y_test, prediction)
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
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

cm = confusion_matrix(dr.y_test, prediction)
ax = plt.axes()
sns.heatmap(cm, annot=True,cbar=False)
ax.set_title("Model accuracy is approx. "+str(acc)+"%")

plt.show()

# =============================================================================
# =============================================================================
# # Save model for future
# =============================================================================
# =============================================================================
PathToSave_model=dr.SavedModel_path
filename = dr.SavedModel_name
savemodel_name = filename
pickle.dump(model, open(PathToSave_model+filename, 'wb'))



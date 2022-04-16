# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2022

@author: AnushKolakalur
"""

import time
import zmq
from DigitsRecon_config import drconfig
dr = drconfig()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
# =============================================================================
# =============================================================================
# #     Wait for a request from client (test image input)
# =============================================================================
# =============================================================================
    for n in range(1,10): # # keeping running for 10 iterations
        
        test_image_path = socket.recv()
        test_image_path = int(test_image_path)
        
        print("\n The number recieved is %s \n" % test_image_path)
        
        print("\n The test image will be displayed for a while and then the model predicts the digit \n")
        test_image_path = test_image_path-1
        dr.fordisp(dr.X_test[test_image_path])
           
        num = test_image_path
        
        test_image = dr.prep4VGGn(dr.X_test[num])
            
        test_image_feature=dr.VGG_net_mod.predict(test_image)
        test_image_features = test_image_feature.reshape(
            test_image_feature.shape[0], -1)
        
        Trained_model = dr.load_trained_model()
        
        prediction = Trained_model.predict(test_image_features)[0] 
        
        print("\n Prediction is sent \n")
    
        #  Send reply back to client
        socket.send_string("\n The prediction is --> digit "+str(prediction))
        
        n += 1
    
    break
    


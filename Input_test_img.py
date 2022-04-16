# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:14:45 2022

@author: AnushKolakalur
"""
import zmq


context = zmq.Context()

#  Socket to talk to server
print("Connecting to Digits Recon server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

num = input(
    "There are 360 test images listed in an array so please enter a number in between 1-360 \n")

socket.send_string(num)

#  Get the reply.
pred = socket.recv()
print(pred)
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:40:03 2019

@author: Garima
"""

import pandas as pd 
import cv2 
import pickle
import imutils
from matplotlib import pyplot as plt

#to use the pretrained model
filename="Gesture KNN.sav"
model = pickle.load(open(filename, 'rb'))


def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        bins,[0, 180, 0, 256, 0, 256]) 
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()


tl_x=93+240
tl_y=39+20
br_x=173+320
br_y=119+80
#to process the image and grab the important part for clustering

# Draw a rectangle around the faces
"""
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
"""
dic=dict(A="Apple",B="Ball",C="Cat",
         D="Day",E="Egg",F="Full",G="Goat",H="Hello",
         I="Books",J="Hi",K="How",L="Love",M="Food",
         N="Night",O="Owl",P="Please",Q="Peace",R="Rest",
         S="Song",T="Teacher",U="Umbrella",V="Wind",W="Water",
         X="Bag",Y="Why",Z="Bye")

video_capture = cv2.VideoCapture(0)

while True:
    train_images=[]
    # Capture frame-by-frame
    ret, frame = video_capture.read()
        # Display the resulting frame
    img=frame
    cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 2)
    
    img=img[tl_y:br_y,tl_x:br_x]
    cv2.imshow('Part', img) 
    img=extract_color_histogram(img)
    train_images.append(img)
    
    pred = model.predict(train_images)
    a=chr(ord('A')+pred[0])
    print(a)

    cv2.putText(frame,dic[a], (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,thickness=12)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()







# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:39:46 2018

@author: CodersMine-Vinay
"""

"""
importing all the modules required to run the program 
np mathematical calculations
pd to do evil stuff on dataset
cv2 image processing
os to browse through the files of pc
glob The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order : tilde and shell variable expansion
tqdm  to prepare progress bar
random to randomise the picture order
pickle to preserve the data set blocks
basename
imutils to enable usage of cv2 efficiently

"""
import numpy as np 
import pandas as pd 
import cv2 
import os 
import glob
import pickle
from os.path import basename
import imutils

path="Dataset\Dataset"

path_dir=np.sort(glob.glob(path+'/*'))



# print(path_dir)
all_csv=[]
for i in path_dir:
	all_csv.append(glob.glob(str(i)+'/*.csv'))
"""
to get the values of all the folders present in the dataset folder
"""
# print(all_csv)


def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
    """
    An image histogram is a graphical representation of the number of pixels in          an image as a function of their intensity. Histograms are made up of bins, each bin representing a certain intensity value range
    
    
Here's How to Calculate the Number of Bins and the Bin Width for a Histogram. Calculate the number of bins by taking the square root of the number of data points and round up. Calculate the bin width by dividing the specification tolerance or range (USL-LSL or Max-Min value) by the # of bins.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    """
    imutils 0.5.2. A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV
    """
    
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)

    """Histogram equalization (or flattening) is the process of redistributing an input image's grey level values to produce an output image easily analyzed by the human eye.
    """
    return hist.flatten()




train_images=[]
labels=[]

for j in all_csv:
    df=pd.read_csv(str(j)[2:-2])

    for i in range(len(df)):
        image_path=os.path.join(path,df.image[i])
        img=cv2.imread(image_path,1)
        img=img[df.top_left_y[i]:df.bottom_right_y[i],df.top_left_x[i]:df.bottom_right_x[i]]

        img=extract_color_histogram(img)
        
        label=basename(os.path.join(path,df.image[i]))[0]

        train_images.append(img)
        labels.append(label)




from sklearn import preprocessing

enocoded_lables= preprocessing.LabelEncoder()
enocoded_lables.fit(labels)
new_lables=enocoded_lables.transform(labels)


print("Labels before Encoding \n"+ str(np.unique(labels)))
print("Labels after Encoding \n"+ str(np.unique(new_lables)))
print("**************************************")

#generating the train data and test data with random values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_images,new_lables,test_size=0.2)


print("train data size : " +str(len(X_train)))
print("test data size : " +str(len(y_test)))


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
print("Model Saved")


"""Saving the model generated """
filename="Gesture KNN.sav"
pickle.dump(classifier,open(filename,'wb'))
print("Model is now Trained and Saved as : "+filename)
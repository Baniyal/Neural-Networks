 #PROBLEM STATEMENT:Image Classification of a picture whether it's a dog or a cat
import numpy as np
import pandas as pd
#here since image_preprocessing part is already done in the format
# of storing the images in proper folders with different labelling

#--------------Importing libraries------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

# colored images converted to 3d array
# black and white converted to 2d array

#--------------INITIALIZING CNN ------------------
classifier = Sequential()
#step 1 convolution - - - - -Convolutional layer containing the feature maps
classifier.add(Convolution2D(32 , 3 , 3 , input_shape = ( 64 ,64 ,3     )   , activation = "relu" ))
# the first argument: nb_filter is the number of filters we apply which will create nb NUMBERS of feature maps created
# second & third  argument is the size of the square matrix of the filter
# forth argument is the input size : last  one is 3 for colored and 2 for black and white , the other is the size of input images
#step 2 is pooling: to reduce the size of th feature map 
classifier.add(MaxPooling2D( pool_size = (2,2) ))
# pool_size is the size of the array that we pass over the feature matrix 
#strides is the step we take to the right
# we take the maximum value within the given pool size 
# step 3  is Flattening : to convert the feature maps into 1d array
classifier.add(Flatten())
#step 4  is FULL CONNECTION:
classifier.add(Dense(output_dim = 128 ,activation = "relu" ))
#output_dim is the number of nodes in hidden layers(which you need to tell yourself)
# we pick 128 by expirimentation , but general rule of thumb should be not too low and not too high
classifier.add(Dense( output_dim = 1, activation = "sigmoid" ))



#----------------------------COMPILING CNN-------------------
classifier.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ["accuracy"])


#-------------------TRAINING OUR MODEL----------------------
#we need to do image augmentation so in order to avoid overfitting
#refer to keras documentation
# overfitting.
"""
image augmentation is a technique
that allows us to enrich our data set, our data set,
without adding more images and therefore that allows
us to get good performance results with little
or not overfitting, even with a small amount of images.
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2, # apply random transvections
        zoom_range=0.2,  #apply random zooms
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)


train_set = train_datagen.flow_from_directory(
        'C:/Users/AAYUSH BANIYAL/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset/training_set',
        target_size=(64, 64), # dimensions expected by our CNN model
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/AAYUSH BANIYAL/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data= test_set,
        validation_steps=2000)











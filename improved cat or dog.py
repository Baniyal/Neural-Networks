# we can improve our accuracry by building a deeper  CNN model
#or we can add another  fully connected layer






# you can also increase the accuracy by taking a higher value of the input size 
# that is more information in the pixel count which will in turn help to train the model better
# which the target size in line 50 and input shape in the convolutional layer
#--------------Importing libraries------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D


#--------------INITIALIZING CNN ------------------
classifier = Sequential()
classifier.add(Convolution2D(32 , 3 , 3 , input_shape = ( 64 ,64 ,3     )   , activation = "relu" ))
classifier.add(MaxPooling2D( pool_size = (2,2) ))
#------------so we add another convolutional layer to improve accuracy
classifier.add(Convolution2D(32 , 3 , 3  , activation = "relu" )) # since this  is the second layer we donot need to specify the input dimension
classifier.add(MaxPooling2D( pool_size = (2,2) ))
#----------------------------------------------------

classifier.add(Flatten())
#step 4  is FULL CONNECTION:
classifier.add(Dense(output_dim = 128 ,activation = "relu" ))
classifier.add(Dense( output_dim = 1, activation = "sigmoid" ))



#----------------------------COMPILING CNN-------------------
classifier.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ["accuracy"])


#-------------------TRAINING OUR MODEL----------------------
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
        steps_per_epoch = 8000,
        epochs=1,
        validation_data= test_set,
        validation_steps=2000)





# --------------------MAKING PREDICTION ----------------

#import a single picture using numpy
import numpy as np 
from keras.preprocessing import image 
test_image = image.load_img("C:/Users/AAYUSH BANIYAL/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg" , target_size = (64,64 ) ) 
test_image = image.img_to_array(test_image) 

#since neural network dont take a single input,so rather we make a batch even if it is of one single element
#thats why we need to make batch.
test_image = np.expand_dims(test_image , axis =  0  )
result = classifier.predict(test_image)












































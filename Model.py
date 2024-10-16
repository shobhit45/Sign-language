# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
from keras.models import model_from_json
import tensorflow as tf
from string import ascii_uppercase

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128 # size of the image

# Making of cnn model using sequential class.
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
# classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers in to 1d array
classifier.add(Flatten())

# Now we are building the neural network
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.30))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.30))
classifier.add(Dense(units=64, activation='relu'))#softmax for more than 2
classifier.add(Dense(units=36, activation='softmax')) 

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Categorical_crossentropy for more than 2
 


#Extracting the summary of our cnn mod
classifier.summary()

# Directories for data
train_dir = 'Dataset/isl_data_grey_split/train'
test_dir = 'Dataset/isl_data_grey_split/test'

# Now we are doing data augmentation 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, # So that flipped image can be detected
        validation_split=0.2  # Specify the validation split here
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data using ImageDataGenerator
training_set = train_datagen.flow_from_directory(train_dir,
                                                target_size=(sz, sz),
                                                batch_size=64, #Images will in the batches of 64
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='training')  # 80% of the data for training

validation_set = train_datagen.flow_from_directory(train_dir,
                                                target_size=(sz , sz),
                                                batch_size=64, #Images will in the batches of 64
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='validation')  # 20% of the data for validation

test_set = test_datagen.flow_from_directory(test_dir,#test folder
                                                target_size=(sz , sz),
                                                batch_size=64, #Images will in the batches of 64
                                                color_mode='grayscale',
                                                class_mode='categorical')

classifier.fit(
        training_set,
        steps_per_epoch=len(training_set),
        epochs=40,
        validation_data=validation_set,
        validation_steps=len(validation_set)
)


# Saving the model to json file
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')

# Now saving the weights of our model 
classifier.save_weights('model-bw.h5')
print('Weights saved')
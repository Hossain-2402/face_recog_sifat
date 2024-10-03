import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Change directory to the folder containing the images
os.chdir('C:/users/zulka/documents/face_recog_sifat')


# Set image size for input
IMAGE_SIZE = [224, 224]

# Define paths
train_path = 'training_data'
valid_path = 'testing_data'

# Load the InceptionV3 model without the top layers
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the pretrained layers
for layer in inception.layers:
    layer.trainable = False

# Get the number of output classes (3 classes: Person_1, Person_2, Unauthorized)
folders = glob('training_data/*')

# Add custom layers for your own classification task
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding dropout
prediction = Dense(len(folders), activation='softmax')(x)

# Create the model object
model = Model(inputs=inception.input, outputs=prediction)

# View the model summary
model.summary()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Image augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # Adding rotation
    brightness_range=[0.8, 1.2],  # Adjusting brightness
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2  # Vertical shift
)

training_set = train_datagen.flow_from_directory('training_data',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

# No augmentation for testing data, just rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('testing_data',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

# Train the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,  # You can increase this for better results
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save the model
model.save('facefeatures_new_model_final.h5')
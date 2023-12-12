import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import pairwise_distances
import requests
from PIL import Image
import pickle
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
#import streamlit as st
#use the below library while displaying the images in jupyter notebook
from IPython.display import display, Image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


print(' in fintune res')


class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('val_loss')<0.2:
            print("loss is low so stopped training")
            self.model.stop_training=True

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('val_loss')<0.2:
            print("loss is low so stopped training")
            self.model.stop_training=True

TRAINING_DIR = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/ntrain"


datagen = ImageDataGenerator(
      validation_split=0.2,
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# VALIDATION_DIR = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/test"
# validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  subset='training',
  batch_size=20
)

validation_generator = datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  subset='validation',
  batch_size=20
)


res_model = applications.ResNet50(include_top=False, weights='imagenet')

for layer in res_model.layers[:143]:
    layer.trainable = False
# Check the freezed was done ok
for i, layer in enumerate(res_model.layers):
    print(i, layer.name, "-", layer.trainable)

to_res = (150, 150)



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
model.add(res_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(43, activation='softmax'))


check_point = tf.keras.callbacks.ModelCheckpoint(filepath="fntune.h5",
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                metrics=['accuracy'])
history = model.fit(train_generator, batch_size=32, epochs=10, verbose=1,
                    validation_data = validation_generator,
                    callbacks=[check_point])
model.summary()
model.save("fntune.h5")


#model = tf.keras.models.load_model('fruitsv1.h5')


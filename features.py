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
import os
import numpy as np

img_width, img_height = 224, 224

#top_model_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_data_dir = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/training"

nb_train_samples = 200
epochs = 50
batch_size = 1

def extract_features():
    Itemcodes = []
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = applications.ResNet50(include_top=False, weights='imagenet')

    for folder in os.listdir(train_data_dir):
        folder_path = os.path.join(train_data_dir, folder)

        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder}")

            generator = datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
            for i in generator.filenames:
                Itemcodes.append(i[(i.find("/")+1):i.find(".")])
            extracted_features = model.predict_generator(generator, nb_train_samples // batch_size)
            extracted_features = extracted_features.reshape((200, 100352))

        np.save(open(f'/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/nfeatures/{folder}_ResNet_features.npy', 'wb'), extracted_features)
        np.save(open(f'/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/nfeatures/{folder}_ResNet_feature_product_ids.npy', 'wb'), np.array(Itemcodes))
    
print("Feature extraction completed.")

    
a = datetime.now()
extract_features()
print("Time taken in feature extraction", datetime.now()-a)





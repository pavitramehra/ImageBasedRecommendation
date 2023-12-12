from datetime import datetime
import os
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from IPython.display import display, Image
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



import faiss


extracted_features = np.load('/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/testfeatures/ntest_ResNet_features.npy')
Productids = np.load('/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/testfeatures/ntest_ResNet_feature_product_ids.npy')
index = faiss.IndexFlatL2(100352)
# add_vector_to_index( extract_features.mean(dim=1), index)


#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    # vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    
    vector = np.float32(embedding)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

add_vector_to_index( extracted_features, index)

faiss.write_index(index,"vector.index")

img_width, img_height = 224, 224
num_results = 5
results = []
nb_train_samples = 4251
epochs = 30
batch_size = 1
def load_features(features_file, product_ids_file):
    features = np.load(features_file)
    product_ids = np.load(product_ids_file)

    return features, product_ids

# # Define a function to save the results to a DataFrame
# def save_results_to_dataframe(results):
#     # columns = [f"Result_{i + 1}" for i in range(num_results)]
#     df_results = pd.DataFrame(results)
#     return df_results


def save_results_to_text_file(results, file_path):
    with open(file_path, 'w') as file:
        for result_array in results:
            line = ' '.join(map(str, result_array[0]))
            file.write(f"{line}\n")


def extract_features(image_path):
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = applications.ResNet50(include_top=False, weights='imagenet')
    print('imgpath', image_path)
    generator = datagen.flow_from_directory(
        image_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    extracted_features1 = model.predict_generator(generator, nb_train_samples // batch_size)
    extracted_features1 = extracted_features1.reshape((4251, 100352))
    return extracted_features1

def check_distances(testimg_folder, features_folder, path):
    # Loop through each image in the test folder
    start_time = time.time()
    print('start_time',start_time)
    for image_file in os.listdir(testimg_folder):
        # if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # image_path = os.path.join(testimg_folder, image_file)
            testimg_feature = extract_features(testimg_folder)
            # print(image_file)
            # print(testimg_feature)
            # Initialize lists to store features and product ids for each feature file
            
            # top_folders = [''] * num_results
            print('testimg',testimg_feature)
            count = 0
           
            for i in range(len(testimg_feature)):
                    
                vector = testimg_feature[i].reshape(1,100352)
                vector = np.float32(vector)
                print(vector.shape)
                faiss.normalize_L2(vector)
                index = faiss.read_index("vector.index")
                d,i = index.search(vector,5)
                print('distances:', d, 'indexes:', i)
                results.append(i)
    print('results',results)                
    # # Save the results to a DataFrame
    # df_results = save_results_to_dataframe(results)

    # # Display or save the DataFrame
    # print(df_results)
    # df_results.to_csv("/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/resultFAISS.csv", index=False)                   

    file_path = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/resFaiss.txt"
    save_results_to_text_file(results, file_path)              
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print('endtime',end_time)


                
# testimg_folder = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/ttrain"
testimg_folder = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/test"
testimg1 = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/test"

features_folder = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/testfeatures"
check_distances(testimg_folder, features_folder, testimg1)

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

fashion_df = pd.read_csv("/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/data/fashion.csv")
fashion_df
from datetime import datetime
import os
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from IPython.display import display, Image

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

# Define a function to save the results to a DataFrame
def save_results_to_dataframe(results):
    columns = [f"Result_{i + 1}" for i in range(num_results)]
    df_results = pd.DataFrame(results, columns=columns)
    return df_results

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
            print('testimg',testimg_feature.shape)
            count = 0
           
            for i in range(len(testimg_feature)):
                    
                # Load features for each feature file
                for folder_feature_file in os.listdir(features_folder):
                    # count+=1
                    # # print('count',count)
                    # if count == 20:
                    #     break
                    top_indices = [-1] * num_results  # Initialize with -1 to check if the list is not fully populated yet
                    top_distances = [float('inf')] * num_results
                    # print(folder_feature_file)
                    if folder_feature_file.endswith('_ResNet_features.npy') and os.path.exists(os.path.join(features_folder, folder_feature_file.replace('_ResNet_features.npy', '_ResNet_feature_product_ids.npy'))):
                        # folder_number = folder_feature_file[:2]  # Extract the number from the file name
                        features_file = os.path.join(features_folder, folder_feature_file)
                        product_ids_file = os.path.join(features_folder, folder_feature_file.replace('_ResNet_features.npy', '_ResNet_feature_product_ids.npy'))
                        
                        print(f"\nChecking distances for feature file: {folder_feature_file}")
                        class_features, class_product_ids = load_features(features_file, product_ids_file)
                        # print(class_features.shape)
                        # Calculate pairwise distances for the current test image and current feature file
                        # print('testf inside' , testimg_feature[i])
                        pairwise_dist = pairwise_distances(class_features, testimg_feature[i].reshape(1,100352))
                        indices = np.argsort(pairwise_dist.flatten())[:num_results]
                        pdists = np.sort(pairwise_dist.flatten())[:num_results]
                        #  print(pdists.shape)
                        # print('pdist',pdists)
                        # print('dist',pairwise_dist)
                        # Update the top indices and distances if a closer match is found
                        for i in range(num_results):
                            if pdists[i] < top_distances[i]:
                                top_distances[i] = pdists[i]
                                top_indices[i] = indices[i]
                                # top_folders[i] = folder_feature_file.replace('_ResNet_features.npy', '')


                # Display the final top 5 results for the current test image
                # print("=" * 20, f"Final Top 5 results for image: {image_file}", "=" * 20)
                results.append(top_indices)
                # for i in range(num_results):
                #     print(f"Result {i + 1}:")
                #     print(f"  Product ID: {top_indices[i]}")
                #     print(f"  Distance: {top_distances[i]}")
                #     print(top_indices[i])
                    # print(top_folders[i])
                    
    
                    # display(Image(os.path.join('/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/test/ntest', f"{class_product_ids[top_indices[i]]}.jpg"), width=224, height=224, embed=True))
    
    print('results',results)                
    # Save the results to a DataFrame
    df_results = save_results_to_dataframe(results)

    # Display or save the DataFrame
    print(df_results)
    df_results.to_csv("/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/results.csv", index=False)                   
                        
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print('endtime',end_time)
    # print("here")
    # for image_file in os.listdir(testimg_folder):
    #     if image_file.endswith(('.jpg', '.jpeg', '.png')):
    #         # image_path = os.path.join(testimg_folder, image_file)
    #         testimg_feature = extract_features(path)

    #         # Initialize lists to store features and product ids for each folder
    #         all_features = []
    #         all_product_ids = []
    #         print("here")

    #         # Initialize lists to store features and product ids for each feature file
    #         top_indices = [-1] * num_results  # Initialize with -1 to check if the list is not fully populated yet
    #         top_distances = [float('inf')] * num_results


    #         # Load features for each feature file
    #         count = 0
    #         for folder_feature_file in os.listdir(features_folder):
    #             count+=1
    #             if count ==3:
    #                 break
    #             prefix = folder_feature_file[:2]  # Assumes a two-digit prefix in your filenames
    #             if folder_feature_file.endswith('_ResNet_features.npy') and f"{prefix}_ResNet_feature_product_ids.npy" in os.listdir(features_folder):
    #                 folder_path = os.path.join(features_folder, folder_feature_file)

    #                 print(f"\nChecking distances for feature file: {folder_feature_file}")
    #                 class_features, class_product_ids = load_features(features_folder, prefix)

    #                 # Calculate pairwise distances for the current test image and current feature file
    #                 pairwise_dist = pairwise_distances(class_features, testimg_feature)
    #                 indices = np.argsort(pairwise_dist.flatten())[:num_results]
    #                 pdists = np.sort(pairwise_dist.flatten())[:num_results]

    #                 # Update the top indices and distances if a closer match is found
    #                 for i in range(num_results):
    #                     if pdists[i] < top_distances[i]:
    #                         top_distances[i] = pdists[i]
    #                         top_indices[i] = indices[i]

    #         # Display the final top 5 results for the current test image
    #         print("=" * 20, f"Final Top 5 results for image: {image_file}", "=" * 20)
    #         for i in range(num_results):
    #             print(f"Result {i + 1}:")
    #             print(f"  Product ID: {class_product_ids[top_indices[i]]}")
    #             print(f"  Distance: {top_distances[i]}")
    #             display(Image(os.path.join('/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/ntrain', prefix, f"{class_product_ids[index]}.jpg"), width=224, height=224, embed=True))

                   

                
# testimg_folder = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/ttrain"
testimg_folder = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/test"
testimg1 = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/imgs/test"

features_folder = "/uufs/chpc.utah.edu/common/home/u1472969/Recommendation-System/testfeatures"
check_distances(testimg_folder, features_folder, testimg1)

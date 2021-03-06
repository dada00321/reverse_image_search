"""
predictive model: Xception (pretrained)
-------------------
refer to:
Keras-Reverse-Image-Search-Using-LSH-and-Cosine-Similarity
https://github.com/TanyaChutani/Keras-Reverse-Image-Search-Using-LSH-and-Cosine-Similarity/blob/master/Reverse_Image_Search_Keras.ipynb
"""
from tqdm import tqdm
from time import time
import tarfile
import os 
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
from modules.get_probable_images_vgg16 import vgg16_get_probable_images

from tensorflow.keras.applications.xception import Xception, preprocess_input

#import shutil
#import pandas as pd 
#from sklearn.utils import shuffle
#from sklearn.preprocessing import LabelBinarizer
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.layers import Input 
#from tensorflow.keras.backend import reshape

"""
Unzip dataset
"""
def unzip_dataset():
    start_time = time()
    upzipped_path = "./dataset/101_ObjectCategories"
    zip_path = "./dataset/101_ObjectCategories.tar.gz"
    if os.path.exists(upzipped_path):
        print("[INFO] Detected the dataset is already unzipped.")
    else:
        print("[INFO] Unzipping dataset...")
        try:
            tar = tarfile.open(zip_path, "r:gz")
            file_names = tar.getnames()
            for file_name in tqdm(file_names, total=len(file_names)):
                dest_path = f"{upzipped_path}/{file_name}"
                tar.extract(file_name, dest_path)
            tar.close()
            print(f"[INFO] Time consumed for unzipping dataset: {round(time() - start_time, 2)} seconds", end=' ')
            # 55.21 seconds
        except Exception:
            print("[WARNING] Fail to unzip dataset.")

"""
Load and preprocess data
"""
def load_data(img_db_path):
    img_list = list()
    for category in sorted(os.listdir(img_db_path)):
        for file in sorted(os.listdir(f"{img_db_path}/{category}")):
            img_path = f"{img_db_path}/{category}/{file}"
            img_list.append(img_path)
    return img_list

def preprocess_image(img_path):
    FIXED_SIZE = (299, 299)
    cv_img = cv2.imread(img_path)
    cv_img = cv2.resize(cv_img, FIXED_SIZE, interpolation=cv2.INTER_NEAREST)
    cv_img = np.expand_dims(cv_img, axis=0)
    cv_img = preprocess_input(cv_img)
    return cv_img

"""
Define model and extract features
"""
def build_xception_model():
    model = Xception(weights="imagenet", include_top=False)
    for layer in model.layers:
        layer.trainable = False 
    #model.summary()
    return model

def extract_feature(cv_img, model):
    img_feature = model.predict(cv_img)
    img_feature = np.array(img_feature)
    img_feature = np.ndarray.flatten(img_feature)
    return img_feature  # 1-d vector

"""
Find similar images via cosine similarity and LSH
"""
def get_similar_image_indices_via_cos_sim(cv_img, model, img_features, top_K):
    new_img_feature = extract_feature(cv_img, model)
    neighbors = NearestNeighbors(n_neighbors=top_K, metric="cosine")\
                            .fit(img_features)
    distances, indices = neighbors.kneighbors([new_img_feature])
    return indices
'''
# LSH code taken from https://towardsdatascience.com/finding-similar-images-using-deep-learning-and-locality-sensitive-hashing-9528afee02f5
def result_vector_LSH(cv_img, img_path, img_features):
    k = 10 
    L = 5  
    d = 27648 
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
    for img_path, vec in (img_features):
        lsh.index(vec.flatten(), extra_data=img_path)
    response = lsh.query(cv_img, num_results= 1)
'''

"""
Result
"""
def show_input(raw_img):
    print("Query:\n", raw_img, sep='')
    cv_img = cv2.imread(raw_img)
    plt.title("Query Image")
    plt.axis('off')
    plt.imshow(cv_img)
  
def show_result(img_list, result):
    #fig = plt.figure(figsize=(12,8))
    '''
    for i in range(0,6):
        index_result = result[0][i]
        plt.subplot(2,3,i+1)
        plt.axis('off')
        plt.imshow(cv2.imread(img_list[index_result]))
    plt.show()
    '''
    print([img_list[img_idx] for img_idx in result[0]])
    # `result`
    # e.g., [[36 37 33 95 74 50]]
    for i, img_idx in enumerate(result[0]):
        ax = plt.subplot(2,3,i+1)
        ax.imshow(cv2.imread(img_list[img_idx]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

"""
Save result
"""
def get_querying_img_name(query_img_name):
    return '_'.join(query_img_name.split("/")[-2:])\
              .split(".")[0]
    
def save_image(output_path, img_path):
    img = cv2.imread(img_path)
    cv2.imwrite(output_path, img)

def save_time_record(time_consumed, save_dir):
    file_name = f"cost_{int(time_consumed)}_sec"
    output_path = f"{save_dir}/{file_name}.txt"
    with open(output_path, 'w') as fp:
        fp.write("")

def save_result(img_list, result, time_consumed, query_img_idx_or_raw_img_path, MAX_AMT):
    base_save_dir = "result/RESULT_clothes2u_reverseImgSearch/v1"
    if os.path.exists(base_save_dir):
        # Create dir for the current result
        if type(query_img_idx_or_raw_img_path) is int and query_img_idx_or_raw_img_path >= 0:
            save_dir = f"{base_save_dir}/[N={MAX_AMT}] {query_img_idx_or_raw_img_path}"
            query_method = "index-in-imglist"
            
        elif type(query_img_idx_or_raw_img_path) is str:
            # Get querying image name
            q_img_name = get_querying_img_name(query_img_idx_or_raw_img_path)
            save_dir = f"{base_save_dir}/[N={MAX_AMT}] {q_img_name}"
            query_method = "raw-img-path"
            
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
            print(f"[WARNING] Result-saving directory `{base_save_dir}` had already created!")
        else:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            print("[INFO] Saving result...")
            img_paths = [img_list[i] for i in result[0]]
            #print("img_paths:",img_paths,sep='')
            
            # Save result images
            for i, img_path in enumerate(img_paths):
                output_path = f"{save_dir}/r{i+1}.png"
                save_image(output_path, img_path)
            
            # Save querying image
            if query_method == "index-in-imglist":
                img_path = img_list[query_img_idx_or_raw_img_path]
                output_path = f"{save_dir}/q_{query_img_idx_or_raw_img_path}.png"
            elif query_method == "raw-img-path":
                img_path = query_img_idx_or_raw_img_path
                output_path = f"{save_dir}/q_{q_img_name}.png"
            
            print("[process: saving querying image]")
            print(f"output_path: {output_path}")
            print(f"img_path: {img_path}")
            save_image(output_path, img_path)
            
            # Save time-record text file
            save_time_record(time_consumed, save_dir)
    else:
        print(f"[WARNING] Base result-saving directory `{base_save_dir}` does not created!")
    
if __name__ == "__main__":
    #unzip_dataset()
    ''' Setting configurations '''
    start_time = time()
    img_db_path = "D:/MyPrograms/Python/py/??????/Cloth Image Classifier/dataset/img_db_3"
    top_K = 20
    # << Select single image to get result >>
    #   --- Option 1: Select by giving index of image ---
    #query_img_idx = 32
    #raw_img = img_list[query_img_idx]
    #   --- Option 2: Select by giving name of image ---
    raw_img = "D:/MyPrograms/Python/py/??????/Cloth Image Classifier/dataset/img_db_3/000102_??????_?????????/00000000 (21).jpg"
    input_cv_img = preprocess_image(raw_img)
    
    ''' Get all images and build a model '''
    #img_db_path = "./dataset/101_ObjectCategories"
    #img_list = load_data(img_db_path)
    
    #print("img_list[0]:")
    #print(img_list[0])
    
    probable_img_list = vgg16_get_probable_images(raw_img, img_db_path)
    model = build_xception_model()
    
    ''' Construct features (vectors) '''
    img_features = list()
    MAX_AMT = 100 # Limiting the data for training
    
    for i in probable_img_list[:MAX_AMT]:
    #for i in img_list[:MAX_AMT]:
    #for i in img_list:
        cv_img = preprocess_image(i)
        img_feature = extract_feature(cv_img, model)
        img_features.append(img_feature)
    img_features = np.array(img_features)
    # i.e.,
    # [[(feature of img-1)],[(...)],[(...)],...]
    
    ''' Get result for the single input image '''
    result = get_similar_image_indices_via_cos_sim(input_cv_img, model, img_features, top_K)
    #print(result)
    #print(len(result))
    
    #show_result(img_list, result)
    #show_input(raw_img)
    time_consumed = time() - start_time
    
    save_result(probable_img_list, result, time_consumed, raw_img, MAX_AMT)
    #save_result(img_list, result, time_consumed, query_img_idx, MAX_AMT)
    
from time import time
import os 
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
from modules.get_probable_images_vgg16 import vgg16_get_probable_images
#from tqdm import tqdm
#import tarfile

import_method_choices = (("v0", "smallVGG16 (customized model)"),
                         ("v1", "Xception (pre-trained model)"),
                         ("v2", "ResNet50 (pre-trained model)"))
import_method = import_method_choices[2][0]
if import_method == "v0":
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from modules.v0_module import load_vgg_model

elif import_method == "v1":
    from tensorflow.keras.applications.xception import preprocess_input
    from modules.v1_module import build_xception_model

elif import_method == "v2":
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from modules.v2_module import build_resnet_model
    
class ReverseImageSearch:
    def __init__(self, method, MAX_AMT, top_K):
        # =============================================================================
        # method: Decide the type of model used to extract (image) features
        #   "v0" => smallVGG16 (customized model)
        #   "v1" => Xception (pre-trained model)
        #   "v2" => ResNet50 (pre-trained model)
        # =============================================================================
        self.available = True
        
        """
        MAX_AMT: 
            the number of images to filter 
            after initially retrieving probable images
            from the result of `ClothImageClassifier`
            => # of probable images can be around
               6k, 8k or more.
            => *constraint: MAX_AMT >= 30
        """
        if type(MAX_AMT) is int:
            if MAX_AMT >= 30:
                self.MAX_AMT = MAX_AMT
            else:
                print("\n[WARNING] Value of argument `MAX_AMT` should be greater than or equal to 30.\n"+\
                      "exec_class: `ReverseImageSearch`\n"+\
                      "exec_method: `__init__`")
                self.available = False
        else:
            print("\n[WARNING] Data type of argument `MAX_AMT` should be an integer.\n"+\
                  "exec_class: `ReverseImageSearch`\n"+\
                  "exec_method: `__init__`")
            self.available = False

        """
        method:
            the type of model used to extract (image) features
        """
        self.available_method_types = ("v0", "v1", "v2")
        if type(method) is str:
            if method in self.available_method_types:
                self.method = method
            else:
                self.available = False
                print("\n[WARNING] Argument `method` is unavailable.\n"+\
                      "exec_class: `ReverseImageSearch`\n"+\
                      "exec_method: `__init__`\n"+\
                      f"There are some available choices of argument `method`: {self.available_method_types}")
        else:
            self.available = False
            print("\n[WARNING] Data type of argument `method` should be a string.\n"+\
                  "exec_class: `ReverseImageSearch`\n"+\
                  "exec_method: `__init__`")

        """
        top_K:
            return K images as the result finally
            => *constraint: top_K >= 10
        """
        self.top_K = top_K
        if type(top_K) is int:
            if top_K >= 10:
                self.MAX_AMT = MAX_AMT
            else:
                print("\n[WARNING] Value of argument `top_K` should be greater than or equal to 10.\n"+\
                      "exec_class: `ReverseImageSearch`\n"+\
                      "exec_method: `__init__`")
                self.available = False
        else:
            print("\n[WARNING] Data type of argument `top_K` should be integer.\n"+\
                  "exec_class: `ReverseImageSearch`\n"+\
                  "exec_method: `__init__`")
            self.available = False
        
    def preprocess_image(self, img_path):
        if self.method == "v0":
            FIXED_SIZE = (96, 96) # self-adjusted vgg16
            #FIXED_SIZE = (224, 224) # vgg16-default
        elif self.method == "v1":
            FIXED_SIZE = (299, 299) # xception-default
        elif self.method == "v2":
            FIXED_SIZE = (224, 224) # resnet-default
        
        cv_img = cv2.imread(img_path)
        cv_img = cv2.resize(cv_img, FIXED_SIZE, interpolation=cv2.INTER_NEAREST)
        cv_img = np.expand_dims(cv_img, axis=0)
        cv_img = preprocess_input(cv_img)
        return cv_img
    
    """
    Define model and extract features
    ( Models are packaged as the modules in directory `modules`,
      loaded/builded depending on selecting which type of argument `method` )
    """
    def extract_feature(self, cv_img, model):
        img_feature = model.predict(cv_img)
        img_feature = np.array(img_feature)
        img_feature = np.ndarray.flatten(img_feature)
        return img_feature  # 1-d vector

    """
    Find similar images via cosine similarity (alternative: LSH)
    """
    def get_similar_image_indices_via_cos_sim(self, cv_img, img_features, model):
        new_img_feature = self.extract_feature(cv_img, model)
        neighbors = NearestNeighbors(n_neighbors=self.top_K, metric="cosine")\
                                .fit(img_features)
        distances, indices = neighbors.kneighbors([new_img_feature])
        return indices
    
    """
    Show result
    """
    def show_input(self, raw_img):
        print("Query:\n", raw_img, sep='')
        cv_img = cv2.imread(raw_img)
        plt.title("Query Image")
        plt.axis('off')
        plt.imshow(cv_img)
      
    def show_result(self, img_list, result):
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
    def get_querying_img_name(self, query_img_name):
        return '_'.join(query_img_name.split("/")[-2:])\
                  .split(".")[0]

    def save_image(self, output_path, img_path):
        img = cv2.imread(img_path)
        cv2.imwrite(output_path, img)
    
    def save_time_record(self, time_consumed, save_dir):
        file_name = f"cost_{int(time_consumed)}_sec"
        output_path = f"{save_dir}/{file_name}.txt"
        with open(output_path, 'w') as fp:
            fp.write("")

    def save_result(self, img_list, result, time_consumed, query_img_idx_or_raw_img_path):
        base_save_dir = f"result/RESULT_clothes2u_reverseImgSearch/{self.method}"
        if os.path.exists(base_save_dir):
            # Create dir for the current result
            if type(query_img_idx_or_raw_img_path) is int and query_img_idx_or_raw_img_path >= 0:
                prior_save_dir = f"{base_save_dir}/[q={query_img_idx_or_raw_img_path}]"
                #save_dir = f"{base_save_dir}/[N={self.MAX_AMT}, K={self.top_K}] {query_img_idx_or_raw_img_path}"
                save_dir = f"{prior_save_dir}/[N={self.MAX_AMT}, K={self.top_K}] {query_img_idx_or_raw_img_path}"
                query_method = "index-in-imglist"

            elif type(query_img_idx_or_raw_img_path) is str:
                # Get querying image name
                q_img_name = self.get_querying_img_name(query_img_idx_or_raw_img_path)
                
                prior_save_dir = f"{base_save_dir}/[q={q_img_name}]"
                #save_dir = f"{base_save_dir}/[N={self.MAX_AMT}, K={self.top_K}] {q_img_name}"
                save_dir = f"{prior_save_dir}/[N={self.MAX_AMT}, K={self.top_K}] {q_img_name}"
                query_method = "raw-img-path"

            if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
                print(f"[WARNING] Result-saving directory `{save_dir}` had already created!")
            else:
                if not os.path.exists(prior_save_dir):
                    os.mkdir(prior_save_dir)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                print("[INFO] Saving result...")
                img_paths = [img_list[i] for i in result[0]]
                #print("img_paths:",img_paths,sep='')
                
                # Save result images
                for i, img_path in enumerate(img_paths):
                    output_path = f"{save_dir}/r{i+1}.png"
                    self.save_image(output_path, img_path)
                
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
                self.save_image(output_path, img_path)
                
                # Save time-record text file
                self.save_time_record(time_consumed, save_dir)
        else:
            print(f"[WARNING] Base result-saving directory `{base_save_dir}` does not created!")
        
    
    def exec_reverse_image_search(self):
        if self.available:
            #self.import_modules()
            
            ''' Setting configurations '''
            start_time = time()
            img_db_path = "D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3"
            # << Select single image to get result >>
            #   --- Option 1: Select by giving index of image ---
            #query_img_idx = 32
            #raw_img = img_list[query_img_idx]
            #   --- Option 2: Select by giving name of image ---
            raw_img = "D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3/000102_灰色_洋裝類/00000000 (21).jpg"
            querying_cv_img = self.preprocess_image(raw_img)
            
            ''' Get all images and build a model '''
            probable_img_list = vgg16_get_probable_images(raw_img, img_db_path)
            
            if self.method == "v0":
                model = load_vgg_model()
            elif self.method == "v1":
                model = build_xception_model()
            elif self.method == "v2":
                model = build_resnet_model()
            
            ''' Construct features (vectors) '''
            img_features = list()
            for i in probable_img_list[:self.MAX_AMT]:
            #for i in img_list[:self.MAX_AMT]:
            #for i in img_list:
                cv_img = self.preprocess_image(i)
                img_feature = self.extract_feature(cv_img, model)
                img_features.append(img_feature)
            img_features = np.array(img_features)
            #print("img_features:\n",img_features,sep='')
            # i.e.,
            # [[(feature of img-1)],[(...)],[(...)],...]
            
            ''' Get result for the single input image '''
            result = self.get_similar_image_indices_via_cos_sim(querying_cv_img, img_features, model)
            #print(result)
            #print(len(result))
            
            #self.show_input(raw_img)
            #self.show_result(img_list, result)
            #self.show_result(probable_img_list, result)
            time_consumed = time() - start_time
            
            ''' Record the result '''
            self.save_result(probable_img_list, result, time_consumed, raw_img)
            #self.save_result(img_list, result, time_consumed, query_img_idx)
            
if __name__ == "__main__":
    # =============================================================================
    # MAX_AMT: Limiting the data for training
    # =============================================================================

    method = "v2"  # method | choices: ("v1","v2","v3")
    MAX_AMT = N = 30  # MAX_AMT must >= 30 | suggestion: 100
    top_K = K = 30  # top_K must >= 10 | E.g., 20, 100, ...
    
    ris = ReverseImageSearch(method, MAX_AMT, top_K)
    ris.exec_reverse_image_search()
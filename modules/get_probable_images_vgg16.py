from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import pickle
from os import listdir
from time import time
"""
Get probable classes via customized vgg16 model
"""
def _sort_listA_by_listB(listA, listB, order="desc"):
    # Example:
    # listA = ["a", "b", "c"]
    # listB = [1, 3, 2]
    zipped_lists = zip(listB, listA)
    if order == "desc":
        sorted_zipped_lists = sorted(zipped_lists, reverse=True)
    elif order == "asc":
        sorted_zipped_lists = sorted(zipped_lists, reverse=False)
    sorted_listA = [element for _, element in sorted_zipped_lists]
    return sorted_listA

def get_classified_result(raw_img):
    # Preprocess input image
    cv_img = cv2.resize(raw_img, (96, 96))
    cv_img = cv_img.astype("float") / 255.0
    cv_img = img_to_array(cv_img)
    cv_img = np.expand_dims(cv_img, axis=0)
    
    # Obtain vgg16 classified result
    classified_result = dict()
    
    labelbin_path = "D:/MyPrograms/Python/py/專題/Cloth Image Classifier/result/003_0320/mlb.pickle"
    mlb = pickle.loads(open(labelbin_path, "rb").read())
    
    model_path = "D:/MyPrograms/Python/py/專題/Cloth Image Classifier/result/003_0320/cloth_classifier.model"
    model = load_model(model_path)
    probs = model.predict(cv_img)[0]
    
    for top_label, top_p in zip(_sort_listA_by_listB(mlb.classes_, probs), sorted(probs, reverse=True)[:4]):
        percentage = round(top_p*100, 2)
        #print(f"{top_label}: {percentage}%")
        classified_result.setdefault(top_label, percentage)
    return classified_result

"""
Get probable directories
and get probable images after
in the order of sorted probable classes (i.e., `classified_result`)
"""
def get_probable_dirs(classified_result, img_db_path, is_show=False): 
    probable_classes = tuple(classified_result.keys())
    # [*] Both colors and categories are sorted!
    colors = list(filter(lambda e: "色" in e, probable_classes))
    categories = list(filter(lambda e: "類" in e, probable_classes))
    #print(colors, "\n", categories, sep='')
    
    # Enumerate all combinations and append to list
    # (type_combinations is "sorted")
    if is_show:
        print("可能的分類:")
        type_combinations = list()
        for color in colors:
            for category in categories:
                print(f"{color}_{category}")
        print()

    type_combinations = [f"{color}_{category}" for color in colors for category in categories]
    #print("type_combinations:\n", type_combinations)
    
    # Sequentially obtain relative paths to a list,
    # and sort the list by 'type_combinations'
    probable_dirs = [f"{path}:{type_}" for path in listdir(img_db_path) for type_ in type_combinations if type_ in path]
    priority_weights = [type_combinations.index(path.split(":")[-1]) for path in probable_dirs]
    probable_dirs = [path.split(":")[0] for path in _sort_listA_by_listB(probable_dirs, priority_weights, "asc")]
    #print(probable_dirs)
    return probable_dirs

def get_probable_images(probable_dirs, img_db_path):
    probable_image_paths = list() # dict()
    for probable_dir in probable_dirs:
        #probable_image_paths.setdefault(probable_type, list())
        for img_path in listdir(f"{img_db_path}/{probable_dir}"):
            full_img_path = f"{img_db_path}/{probable_dir}/{img_path}"
            #print(full_img_path, exists(full_img_path))
            #probable_image_paths.append(full_img_path)
            probable_image_paths.append(full_img_path)
            #probable_image_paths[probable_dir].append(full_img_path)
    return probable_image_paths

def vgg16_get_probable_images(img_path, img_db_path):
    start_time = time()
    
    raw_img = cv2.imread(img_path)
    classified_result = get_classified_result(raw_img)
    #print(classified_result)
    '''
    ( output of `classified_result` )
    {'灰色': 98.68, '洋裝類': 92.57,
     '童裝類': 3.45, '白色': 2.13}
    '''
    
    is_show = False
    probable_dirs = get_probable_dirs(classified_result, img_db_path, is_show)
    #print(probable_dirs)
    '''
    probable classes: (=> if is_show )
    灰色_洋裝類
    灰色_童裝類
    白色_洋裝類
    白色_童裝類
    
    ( output of `probable_dirs` => sorted )
    ['000102_灰色_洋裝類', '000105_灰色_童裝類', '000126_白色_洋裝類', '000129_白色_童裝類']
    '''
    
    probable_image_paths = get_probable_images(probable_dirs, img_db_path)
    #print(probable_image_paths[:3])
    '''
    ['D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3/000102_灰色_洋裝類/00000000 (1).jpg', 'D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3/000102_灰色_洋裝類/00000000 (10).jpg', 'D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3/000102_灰色_洋裝類/00000000 (10).png']    
    '''
    
    time_consumed = round(time()-start_time, 2)
    print(f"Time consumed: {time_consumed} seconds")
    '''
    Time consumed: 10.34 seconds
    '''
    return probable_image_paths

if __name__ == "__main__":
    img_path = "D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3/000102_灰色_洋裝類/00000000 (21).jpg"
    img_db_path = "D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3"
    probable_image_paths = vgg16_get_probable_images(img_path, img_db_path)
    print(len(probable_image_paths)) # 1829
    print(probable_image_paths[:3])

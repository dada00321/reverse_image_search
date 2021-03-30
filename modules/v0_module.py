from tensorflow.keras.models import load_model
import pickle

def load_vgg_model():
    model_path = "C:/Users/user/Desktop/專題_開發/Image Search/model_training_result/003_0320/cloth_classifier.model"
    model = load_model(model_path)
    return model
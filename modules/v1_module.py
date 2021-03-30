from tensorflow.keras.applications.xception import Xception

def build_xception_model():
    model = Xception(weights="imagenet", include_top=False)
    for layer in model.layers:
        layer.trainable = False 
    #model.summary()
    return model
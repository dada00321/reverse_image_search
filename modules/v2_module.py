from tensorflow.keras.applications.resnet50 import ResNet50

def build_resnet_model():
    '''
    model = Xception(weights="imagenet", include_top=False)
    for layer in model.layers:
        layer.trainable = False 
    #model.summary()
    return model
    '''
    # 以訓練好的 ResNet50 為基礎來建立模型，
    # 捨棄 ResNet50 頂層的 fully connected layers
    '''
    model = ResNet50(weights="imagenet",include_top=False, 
                     input_tensor=None)
    '''
    model = ResNet50(weights="imagenet", include_top=False)
    return model
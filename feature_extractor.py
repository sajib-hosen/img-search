from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_modal = VGG16(weights="imagenet")
        self.modal = Model(inputs=base_modal.input, outputs=base_modal.get_layer("fc1").output)

    def extract(self, img):
        img = img.resize((224, 224)).convert("RGB") # resizing and converting to RGB image
        x = image.img_to_array(img)  # converting image to np.array
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.modal.predict(x)[0]

        # print(x)
        # print(feature, np.linalg.norm(feature))
        # print(np.linalg.norm(feature))
        # print(feature / np.linalg.norm(feature))

        return feature / np.linalg.norm(feature)

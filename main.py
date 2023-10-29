import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

from tensorflow.keras.applications.efficientnet import preprocess_input

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    CLASSIFIER_URL = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'
    df = pd.read_csv("https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv")
    IMAGE_RES=224

    model = tf.keras.Sequential([
        hub.KerasLayer(CLASSIFIER_URL)
    ])
    
    import keras.utils as image

    img_path = '1800.jpeg'
    img = image.load_img(img_path, target_size=(IMAGE_RES, IMAGE_RES))
    plt.imshow(img.convert('RGBA'))
    plt.show()

    x = image.img_to_array(img)
    x = 255 - x
    x /= 255
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)
    predicted_class = np.argmax(result)
    print("Номер класса:", predicted_class)
    print("Название класса:", df[df['id'].isin([predicted_class])]['name'].tolist())

if __name__ =="__main__":
    main()

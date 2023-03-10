import tensorflow as tf
import numpy as np
import cv2

def drowsiness_prediction():
    labels_new = ["yawn", "no_yawn", "Closed", "Open"]
    IMG_SIZE = 145
    def prepare(filepath, face_cas="./prediction_images/haarcascade_frontalface_default.xml"):
        img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img_array = img_array / 255
        resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    model = tf.keras.models.load_model("./drowiness_new6.h5")

    prediction = model.predict([prepare(input("Enter relative image path:"))])
    return labels_new[np.argmax(prediction)]

if __name__ == "__main__":
    output = drowsiness_prediction()
    print(output)
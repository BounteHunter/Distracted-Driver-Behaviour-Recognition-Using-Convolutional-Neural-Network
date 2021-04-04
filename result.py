import cv2
from keras.preprocessing.image import load_img, img_to_array
from helper import create_top_model, class_labels, target_size
import numpy as np
from keras.applications.vgg16 import VGG16
import operator
import sys
import threading
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

label = ''
frame = None
model = VGG16(include_top=False, weights='imagenet')

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        global frame
        # Loading the VGG16 network
        while (~(frame is None)):
            
            label = self.predict(frame)

    def predict(self, frame):
        global model
        
        image_arr = img_to_array(frame)
        image_arr /= 255
        image_arr = np.expand_dims(image_arr, axis=0)
        bottleneck_features = model.predict(image_arr)
        model1 = create_top_model("softmax", bottleneck_features.shape[1:])
        model1.load_weights("./res/_top_model_weights.h5")
        predicted = model1.predict(bottleneck_features)
        decoded_predictions = dict(zip(class_labels, predicted[0]))
        decoded_predictions = max(decoded_predictions.items(), key=operator.itemgetter(1))
        return decoded_predictions

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

ret, original = cap.read()

frame = cv2.resize(original, (224, 224))

keras_thread = MyThread()
keras_thread.start()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))
    # Display the predictions
    
    cv2.putText(original, "Label: {}".format(label), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (8, 5, 173), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
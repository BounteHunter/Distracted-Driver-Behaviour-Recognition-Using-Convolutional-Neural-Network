### ---------- Handle Command Line Arguments ----------

import argparse
import cv2
from keras.preprocessing.image import load_img, img_to_array
from helper import create_top_model, class_labels, target_size
import numpy as np
from keras import applications
import operator
import matplotlib.pyplot as plt
import argparse
import sys
import threading

def func():
    global model
    global decoded_predictions
    global sent
    ret, original = cap.read()
    frame = cv2.resize(original, (224, 224))
    image_arr = img_to_array(frame)
    image_arr /= 255
    image_arr = np.expand_dims(image_arr, axis=0)
    bottleneck_features = model.predict(image_arr)
    model1 = create_top_model("softmax", bottleneck_features.shape[1:])
    model1.load_weights("res/_top_model_weights.h5")
    predicted = model1.predict(bottleneck_features)
    decoded_predictions = dict(zip(class_labels, predicted[0]))
    decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)
    sent = False

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()


model = applications.VGG16(include_top=False, weights='imagenet')
sent = False
decoded_predictions = None

ret, original = cap.read()
frame = cv2.resize(original, (224, 224))
func()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))

    if sent == False:
        threading.Thread(target=func).start()

    cv2.imshow("Classification", original)
    cv2.putText(original, "Label: {}".format(decoded_predictions[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print()
    count = 1
    for key, value in decoded_predictions:
    	print("{}. {}: {:8f}%".format(count, key, value*100))
    	count += 1

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
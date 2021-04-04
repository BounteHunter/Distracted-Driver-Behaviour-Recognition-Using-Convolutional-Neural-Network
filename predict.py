import argparse

hide_img = False 

a = argparse.ArgumentParser(description="Predict the class of a given driver image.")
a.add_argument("--image", help="path to image", default="../../../dataset/split_data/test/c6/img_124.jpg")
a.add_argument("--hide_img", action="store_true", help="do NOT display image on prediction termination")
args = a.parse_args()

if args.hide_img:
    hide_img = True
    
if args.image is not None:
    img_path = args.image


from keras.preprocessing.image import load_img, img_to_array
from helper import create_top_model, class_labels, target_size
import numpy as np
from keras import applications
import operator
import matplotlib.pyplot as plt
import argparse

img_path = args.image

image = load_img(img_path, target_size=target_size)

image_arr = img_to_array(image)
image_arr /= 255

image_arr = np.expand_dims(image_arr, axis=0)
  
model = applications.VGG16(include_top=False, weights='imagenet')  

bottleneck_features = model.predict(image_arr) 

model = create_top_model("softmax", bottleneck_features.shape[1:])

model.load_weights("res/_top_model_weights.h5")

predicted = model.predict(bottleneck_features)
decoded_predictions = dict(zip(class_labels, predicted[0]))
decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

print()
count = 1
for key, value in decoded_predictions[:5]:
    print("{}. {}: {:8f}%".format(count, key, value*100))
    count += 1
    
if not hide_img:
    # printing image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
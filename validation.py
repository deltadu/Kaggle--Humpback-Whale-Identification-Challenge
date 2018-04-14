import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing.image import (random_rotation, random_shift, random_shear, random_zoom)

def get_image(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

train_df = pd.read_csv('train.csv')

num_categories = len(train_df['Id'].unique())
validation = np.zeros((224**2, num_categories))

i = 0
for id in train_df['Id']:
	im = train_df[train_df['Id'] == id].sample(1)
	name =  np.array(im.get('Image'))[0]

	im = get_image('train/' + name)

	# https://www.kaggle.com/lextoumbourou/humpback-whale-id-data-and-aug-exploration
	x = random.randint(0, 3)
	if x == 0:
		validation[i,:] = random_rotation(im, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
	if x == 1:
		validation[i,:] = random_shift(im, wrg=0.1, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
	if x == 2:
		validation[i,:] = random_shear(im, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
	if x == 3:
		validation[i,:] = random_zoom(im, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
	i = i + 1
import tensorflow as tf
import numpy as np
from custom_models import MADNet, colorize_img
from preprocessing import StereoDatasetCreator
import matplotlib.pyplot as plt


image_height = 320
image_width = 1216
input_size = (image_height, image_width)
batch_size = 1 # Set batch size to none to have a variable batch size
search_range = 2 # maximum dispacement (ie. smallest disparity)

predict_left_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/image_clean/left"
predict_right_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/image_clean/right"
model_output_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/models"


# Initialise the model
model = MADNet(height=image_height, width=image_width, search_range=search_range, batch_size=batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer, 
    run_eagerly = True   
)


# Get training data
predict_dataset = StereoDatasetCreator(
    left_dir=predict_left_dir, 
    right_dir=predict_right_dir, 
    batch_size=batch_size, 
    height=image_height, 
    width=image_width,
    shuffle=False,
    ) 

predict_ds = predict_dataset()


model.MAD_predict = True
model.num_adapt_modules = 1


disparities = model.predict(predict_ds, steps=3)



for i in range(disparities.shape[0]):

    fig = plt.figure(figsize=(8,6))
    plt.axis("off")
    plt.grid(b=None)
    plt.imshow(disparities[i])
    #plt.imshow(colorize_img(disparities[i], cmap='jet').numpy())
    plt.show()

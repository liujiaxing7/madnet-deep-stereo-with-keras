import os
import tensorflow as tf
import numpy as np
from custom_models import MADNet
from preprocessing import StereoDatasetCreator


print("\nTensorFlow Version: {}".format(tf.__version__))


image_height = 320
image_width = 1216
input_size = (image_height, image_width)
batch_size = 1 
search_range = 2 # maximum dispacement (ie. smallest disparity)

train_left_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/image_clean/left"
train_right_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/image_clean/right"
train_left_disp_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/disparity/left"
model_output_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/models"

# Initialise the model
model = MADNet(height=image_height, width=image_width, search_range=search_range, batch_size=batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer, 
    run_eagerly = True   
)

# Get training data
train_dataset = StereoDatasetCreator(
    left_dir=train_left_dir, 
    right_dir=train_right_dir, 
    batch_size=batch_size, 
    height=image_height, 
    width=image_width,
    shuffle=False,
    disp_dir=train_left_disp_dir
    ) 

train_ds = train_dataset()


# Create callbacks
def scheduler(epoch, lr):
    if epoch < 400000:
        return lr
    elif epoch < 600000:
        return lr * 0.5
    else:
        # learning_rate * decay_rate ^ (global_step / decay_steps)
        decay_rate = 0.5       
        return lr * 0.5 * decay_rate ** (epoch // 600000)


schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_output_dir + "/weights/epoch-{epoch:02d}-MADNet",
    save_freq=10,
    save_weights_only=False,
    verbose=0
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=model_output_dir + "/tensorboard",
    histogram_freq=1,
    write_steps_per_second=True,
    update_freq="batch"
    )

# Fit the model
history = model.fit(
    x=train_ds,
    epochs=2,
    verbose=1,
    steps_per_epoch=10,
    callbacks=[
        tensorboard_callback,
        save_callback,
        schedule_callback
    ]
)

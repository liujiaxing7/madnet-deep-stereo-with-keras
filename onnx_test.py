import time

import tensorflow as tf
from matplotlib import pyplot as plt

from madnet import colorize_img
from onnxmodel import ONNXModel
from PIL import Image
import numpy as np


def get_image( path):
    """
    Get a single image helper function
    Converts image to float32, normalises values to 0-1
    and resizes to the desired shape
    Args:
        path to image (will be in Tensor format, since its called in a graph)
    Return:
        Tensor in the shape (height, width, 3)
    """
    # Using tf.io.read_file since it can take a tensor as input
    raw = tf.io.read_file(path)
    # Converts to float32 and normalises values
    image = tf.io.decode_image(raw, channels=3, dtype=tf.float32, expand_animations=False)
    # Change dimensions to the desired model dimensions
    image = tf.image.resize(image, [320, 1216], method="bilinear")
    # if self.augment:
    #     image = tf.image.random_hue(image, 0.08)
    #     image = tf.image.random_saturation(image, 0.6, 1.6)
    #     image = tf.image.random_contrast(image, 0.7, 1.3)
    return image

start_time = time.time()

net = ONNXModel("MADNet_12.onnx")

end_time = time.time()
print("load time :",end_time-start_time)

limg_tensor = get_image("test_images/left/000000_00.png")
rimg_tensor = get_image("test_images/right/000000_00.png")


limg_tensor = tf.expand_dims(limg_tensor, axis=0)
limg_tensor = tf.transpose(limg_tensor, perm=[0, 3, 1, 2])
rimg_tensor = tf.expand_dims(rimg_tensor, axis=0)
rimg_tensor = tf.transpose(rimg_tensor, perm=[0, 3, 1, 2])

limg=limg_tensor.numpy()
# limg=limg.per
rimg=rimg_tensor.numpy()


#测试时间时用了for循环，所以表格中不是第一次时间，是跑起来的一个状态
start_time_inter = time.time()
output = net.forward(limg, rimg)
end_time_inter = time.time()
print("interface time :",end_time_inter-start_time_inter)

# View disparity predictions

plt.axis("off")
plt.grid(visible=None)
# disp = tf.expand_dims(output, axis=0)
disp = output
plt.imshow(colorize_img(output[0])[0], cmap='jet')
plt.show()

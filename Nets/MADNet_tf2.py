import os
import tensorflow as tf
import numpy as np
from keras.engine import data_adapter
import cv2

print("\nTensorFlow Version: {}".format(tf.__version__))


# dummpy data for the images
image_height = 320
image_width = 1216
input_size = (image_height, image_width)
batch_size = 1 # Set batch size to none to have a variable batch size
search_range = 2 # maximum dispacement (ie. smallest disparity)

left_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/image_clean/left"
right_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/image_clean/right"
left_disp_dir = "C:/Users/Christian/Documents/BiglyBT Downloads/FlyingThings3D_subset/train/disparity/left"



class SSIMLoss(tf.keras.losses.Loss):
    """
    SSIM dissimilarity measure
    Used for self-supervised training
    Args:
        x: target image
        y: predicted image
    """
    def __init__(self, name="mean_SSIM_l1"):
        super(SSIMLoss, self).__init__(name=name)
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(3,3) ,strides=(1,1), padding='valid')

    def call(self, x, y):
        C1 = 0.01**2
        C2 = 0.03**2
        mu_x = self.pool(x)
        mu_y = self.pool(y)

        sigma_x = self.pool(x**2) - mu_x**2
        sigma_y = self.pool(y**2) - mu_y**2
        sigma_xy = self.pool(x*y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d
        SSIM = tf.clip_by_value((1-SSIM)/2, 0 ,1)

        mean_SSIM = tf.reduce_mean(SSIM)

        sum_l1 = tf.reduce_sum(tf.abs(x - y))

        return 0.85 * mean_SSIM + 0.15 * sum_l1


class ReconstructionLoss(tf.keras.losses.Loss):
    """
    Reconstruction loss function (mean l1)
    Per pixel absolute error between groundtruth 
    disparity and predicted disparity
    Used for supervised training
    Args:
        x: target image
        y: predicted image
    """
    def __init__(self, name="mean_l1"):
        super(ReconstructionLoss, self).__init__(name=name)

    def call(self, x, y):
        return tf.reduce_sum(tf.abs(x-y))




# https://github.com/philferriere/tfoptflow/blob/bdc7a72e78008d1cd6db46e4667dffc2bab1fe9e/tfoptflow/core_costvol.py
class StereoCostVolume(tf.keras.layers.Layer):
    """Build cost volume for associating a pixel from the left image with its corresponding pixels in the right image.
    Args:
        c1: Level of the feature pyramid of the left image
        warp: Warped level of the feature pyramid of the right image
        search_range: Search range (maximum displacement)
    """
    def __init__(self, name="cost_volume"):
        super(StereoCostVolume, self).__init__(name=name)

    def call(self, c1, warp, search_range):
        padded_lvl = tf.pad(warp, [[0, 0], [0, 0], [search_range, search_range], [0, 0]])
        width = c1.shape.as_list()[2]
        max_offset = search_range * 2 + 1

        cost_vol = []
        for i in range(0, max_offset):

            slice = padded_lvl[:, :, i:width+i, :]
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)

        cost_vol = tf.concat(cost_vol, axis=3)

        cost_curve = tf.concat([c1, cost_vol], axis=3)

        return cost_curve

class BuildIndices(tf.keras.layers.Layer):

    def __init__(self, name="build_indices", batch_size=1):
        super(BuildIndices, self).__init__(name=name)
        self.batch_size = batch_size

    def call(self, coords):

        _, height, width, _ = coords.get_shape().as_list()

        pixel_coords = np.ones((1, height, width, 2), dtype=np.float32)
        batches_coords = np.ones((self.batch_size, height, width, 1), dtype=np.float32)

        # pixel_coords = tf.ones((1, height, width, 2), dtype=tf.float32)
        # batches_coords = tf.ones((self.batch_size, height, width, 1), dtype=tf.float32)        

        for i in range(0, self.batch_size):
            batches_coords[i][:][:][:] = i
        # build pixel coordinates and their disparity
        for i in range(0, height):
            for j in range(0, width):
                pixel_coords[0][i][j][0] = j
                pixel_coords[0][i][j][1] = i

        pixel_coords = tf.constant(pixel_coords, tf.float32)

        output = tf.concat([batches_coords, pixel_coords + coords], -1)            

        return output


class Warp(tf.keras.layers.Layer):

    def __init__(self, name="warp"):
        super(Warp, self).__init__(name=name)

    def call(self, imgs, coords):
            
        coord_b, coords_x, coords_y = tf.split(coords, [1, 1, 1], axis=3)

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1],dtype=tf.float32)

        x0_safe = tf.clip_by_value(x0, zero[0], x_max)
        y0_safe = tf.clip_by_value(y0, zero[0], y_max)
        x1_safe = tf.clip_by_value(x1, zero[0], x_max)

        # bilinear interp weights, with points outside the grid having weight 0
        wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')


        im00 = tf.cast(tf.gather_nd(imgs, tf.cast(
            tf.concat([coord_b, y0_safe, x0_safe], -1), 'int32')), 'float32')
        im01 = tf.cast(tf.gather_nd(imgs, tf.cast(
            tf.concat([coord_b, y0_safe, x1_safe], -1), 'int32')), 'float32')

        output = tf.add_n([
            wt_x0 * im00, wt_x1 * im01
        ])

        return output


class StereoContextNetwork(tf.keras.Model):

    def __init__(self, name="residual_refinement_network", batch_size=1):
        super(StereoContextNetwork, self).__init__(name=name)
        print(f"\nStarted Initialization context network")
        self.batch_size = batch_size
        act = tf.keras.layers.Activation(tf.nn.leaky_relu)
        self.final_reprojection_loss = None
        self.x = None
        # Loss function for self-supervised training (no groundtruth disparity)
        self.loss_fn = SSIMLoss()
        self.context1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=1, padding="same", activation=act, use_bias=True, name="context1")
        self.context2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=2, padding="same", activation=act, use_bias=True, name="context2")
        self.context3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=4, padding="same", activation=act, use_bias=True, name="context3")
        self.context4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), dilation_rate=8, padding="same", activation=act, use_bias=True, name="context4")
        self.context5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), dilation_rate=16, padding="same", activation=act, use_bias=True, name="context5")
        self.context6 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), dilation_rate=1, padding="same", activation=act, use_bias=True, name="context6")
        self.context7 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), dilation_rate=1, padding="same", activation="linear", use_bias=True, name="context7")
        self.add = tf.keras.layers.Add(name="context_disp")
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.warp = Warp(name="warp_final")
        self.build_indices = BuildIndices(name="build_indices_final", batch_size=self.batch_size)

    def call(self, input, disp, final_left, final_right, training=False):
        volume = self.concat([input, disp])

        self.x = self.context1(volume)
        self.x = self.context2(self.x)
        self.x = self.context3(self.x)
        self.x = self.context4(self.x)
        self.x = self.context5(self.x)
        self.x = self.context6(self.x)
        self.x = self.context7(self.x)

        context_disp = self.add([disp, self.x])
        final_disparity = tf.keras.layers.Resizing(name="final_disparity", height=final_left.shape[1], width=final_left.shape[2], interpolation='bilinear')(context_disp)

        # warp right image with final disparity to get final reprojection loss
        final_coords = self.concat([final_disparity, tf.zeros_like(final_disparity)])
        final_indices = self.build_indices(final_coords)
        # Warp the right image into the left using final disparity
        final_warped_left = self.warp(final_right, final_indices)    

        if training == True:
            self.final_reprojection_loss = self.loss_fn(final_warped_left, final_left)      

        return final_disparity, self.final_reprojection_loss


class StereoEstimator(tf.keras.Model):
    """
    This is the stereo estimation network at resolution n.
    It uses the costs (from the pixel difference between the warped right image 
    and the left image) combined with the upsampled disparity from the previous
    layer (when the layer is not the last layer).

    The output is predicted disparity for the network at resolution n.
    """

    def __init__(self, name="volume_filtering"):
        super(StereoEstimator, self).__init__(name=name)
        act = tf.keras.layers.Activation(tf.nn.leaky_relu)
        #self.x = None
        self.disp1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp1")
        self.disp2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp2")
        self.disp3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp3")
        self.disp4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp4")
        self.disp5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp5")
        self.disp6 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding="same", activation="linear", use_bias=True, name="disp6")
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, costs, upsampled_disp=None):
        if upsampled_disp is not None:
            volume = self.concat([costs, upsampled_disp])
        else:
            volume = costs
        x = self.disp1(volume)
        x = self.disp2(x)
        x = self.disp3(x)
        x = self.disp4(x)
        x = self.disp5(x)
        x = self.disp6(x)
        return x


class ModuleM(tf.keras.Model):
    """
    Module MX is a sub-module of MADNet, which can be trained individually for 
    online adaptation using the MAD (Modular ADaptaion) method.
    """
    def __init__(self, name="MX", layer="X", search_range=2, batch_size=1):
        print(f"\nStarted Initialization ModuleM {layer}")
        super(ModuleM, self).__init__(name=name)
        self.search_range = search_range
        self.batch_size = batch_size
        self.reprojection_loss = None
        self.layer = layer
        # Loss function for self-supervised training (no groundtruth disparity)
        self.loss_fn = SSIMLoss()
        self.cost_volume = StereoCostVolume(name=f"cost_{self.layer}")
        self.stereo_estimator = StereoEstimator(name=f"volume_filtering_{self.layer}")

    def call(self, left, right, prev_disp=None, training=False):
        # print(f"\nStarted call ModuleM {self.layer}")

        # print(f"left: {left.shape}")
        # print(f"right: {right.shape}")

        height, width = (left.shape.as_list()[1], left.shape.as_list()[2])
        # Check if layer is the bottom of the pyramid
        if prev_disp is not None:
            # Upsample disparity from previous layer
            upsampled_disp = tf.keras.layers.Resizing(name=f"upsampled_disp_{self.layer}", height=height, width=width, interpolation='bilinear')(prev_disp)
            coords = tf.keras.layers.concatenate([upsampled_disp, tf.zeros_like(upsampled_disp)], -1)
            indices = BuildIndices(name=f"build_indices_{self.layer}", batch_size=self.batch_size)(coords)
            # Warp the right image into the left using upsampled disparity
            warped_left = Warp(name=f"warp_{self.layer}")(right, indices)
        else:
            # No previous disparity exits, so use right image instead of warped left
            warped_left = right

        if training == True:
            # add loss estimating the reprojection accuracy of the pyramid level (for self supervised training/MAD)
            self.reprojection_loss = self.loss_fn(warped_left, left)

        costs = self.cost_volume(left, warped_left, self.search_range)

        # Get the disparity using cost volume between left and warped left images
        module_disparity = self.stereo_estimator(costs)

        return module_disparity, self.reprojection_loss


# ------------------------------------------------------------------------
# Model Creation
class MADNet(tf.keras.Model):
    """
    The main MADNet model
    """
    def __init__(self, name="MADNet", height=320, width=1216, search_range=2, batch_size=1):
        super(MADNet, self).__init__(name=name)
        print("\nStarted Initialization MADNet")
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.search_range = search_range
        self.batch_size = batch_size
        self.losses_dict = {}
        # Selects whether to perform MAD when running .predict
        self.MAD_predict = True 
        # Loss function for supervised training (with groundtruth)
        self.loss_fn = ReconstructionLoss() 

        act = tf.keras.layers.Activation(tf.nn.leaky_relu)
        # Left image feature pyramid (feature extractor)
        self.left_pyramid = None
        # F1
        self.left_conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv1", 
        input_shape=(self.height, self.width, 3, ))
        self.left_conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv2")
        # F2
        self.left_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv3")
        self.left_conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv4")
        # F3
        self.left_conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv5")
        self.left_conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv6")
        # F4
        self.left_conv7 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv7")
        self.left_conv8 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv8")
        # F5
        self.left_conv9 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv9")
        self.left_conv10 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv10")
        # F6
        self.left_conv11 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv11")
        self.left_conv12 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv12")       
        # Right image feature pyramid (feature extractor)
        self.right_pyramid = None
        # F1
        self.right_conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv1", 
        input_shape=(self.height, self.width, 3, ))
        self.right_conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv2")
        # F2
        self.right_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv3")
        self.right_conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv4")
        # F3
        self.right_conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv5")
        self.right_conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv6")
        # F4
        self.right_conv7 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv7")
        self.right_conv8 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv8")
        # F5
        self.right_conv9 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv9")
        self.right_conv10 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv10")
        # F6
        self.right_conv11 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv11")
        self.right_conv12 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv12")

        #############################SCALE 6#################################
        self.M6 = ModuleM(name="M6", layer="6", search_range=self.search_range, batch_size=self.batch_size)
        ############################SCALE 5###################################
        self.M5 = ModuleM(name="M5", layer="5", search_range=self.search_range, batch_size=self.batch_size)
        ############################SCALE 4###################################
        self.M4 = ModuleM(name="M4", layer="4", search_range=self.search_range, batch_size=self.batch_size)
        ############################SCALE 3###################################
        self.M3 = ModuleM(name="M3", layer="3", search_range=self.search_range, batch_size=self.batch_size)
        ############################SCALE 2###################################
        self.M2 = ModuleM(name="M2", layer="2", search_range=self.search_range, batch_size=self.batch_size)
        ############################REFINEMENT################################
        self.refinement_module = StereoContextNetwork(batch_size=self.batch_size)

    def train_step(self, data):
        #print("\nInside train_step MADNet")
        # Left, right image inputs and groundtruth target disparity
        #inputs, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        inputs, gt = data
        left_input = inputs["left_input"]
        right_input = inputs["right_input"]        

        # Check the image shape and resize if necessary
        left_shape = left_input.shape[:3]
        right_shape = right_input.shape[:3]  
        desired_shape = (self.batch_size, self.height, self.width)
        print(f"data: {data}")
        # print(f"Left input: {left_shape}")
        # print(f"Right input: {right_shape}")
        # print(f"gt: {gt}")


        if left_shape != desired_shape or right_shape != desired_shape:
            print("WARNING: image input shape is different to the expected value")
            print(f"Resizing images from shape: left: {left_shape} right: {right_shape} to shape {desired_shape}")
            left_input = tf.image.resize(left_input, [self.height, self.width], method="bilinear")
            right_input = tf.image.resize(right_input, [self.height, self.width], method="bilinear")
        if gt is not None and gt.shape[:3] != desired_shape:
            print(f"Resizing groundtruth from shape: {gt.shape[:3]} to shape {desired_shape}")
            gt = tf.image.resize(gt, [self.height, self.width], method="bilinear")

        # Set the shape of the inputs to the desired width and height

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            _ = self(inputs={'left_input': left_input, 'right_input': right_input}, target=gt, training=True)


        #((((((((((((((((((((((((Select modules using losses))))))))))))))))))))))))
        # Not selecting modules to update in train step
        # Will either perform offline supervised adaptation
        # or full MAD (depending on whether gt images are provided)

        #^^^^^^^^^^^^^^^^^^^^^^^^Compute Gradients^^^^^^^^^^^^^^^^^^^^^^^^
        #############################SCALE 6#################################
        left_F6_grads = tape.gradient(self.losses_dict["D6"], self.left_conv12.trainable_weights)
        left_F06_grads = tape.gradient(self.losses_dict["D6"], self.left_conv11.trainable_weights)
        right_F6_grads = tape.gradient(self.losses_dict["D6"], self.right_conv12.trainable_weights)
        right_F06_grads = tape.gradient(self.losses_dict["D6"], self.right_conv11.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M6_grads = tape.gradient(self.losses_dict["D5"], self.M6.trainable_weights)        
        ############################SCALE 5###################################
        left_F5_grads = tape.gradient(self.losses_dict["D5"], self.left_conv10.trainable_weights)
        left_F05_grads = tape.gradient(self.losses_dict["D5"], self.left_conv9.trainable_weights)
        right_F5_grads = tape.gradient(self.losses_dict["D5"], self.right_conv10.trainable_weights)
        right_F05_grads = tape.gradient(self.losses_dict["D5"], self.right_conv9.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M5_grads = tape.gradient(self.losses_dict["D4"], self.M5.trainable_weights)   
        ############################SCALE 4###################################
        left_F4_grads = tape.gradient(self.losses_dict["D4"], self.left_conv8.trainable_weights)
        left_F04_grads = tape.gradient(self.losses_dict["D4"], self.left_conv7.trainable_weights)
        right_F4_grads = tape.gradient(self.losses_dict["D4"], self.right_conv8.trainable_weights)
        right_F04_grads = tape.gradient(self.losses_dict["D4"], self.right_conv7.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M4_grads = tape.gradient(self.losses_dict["D3"], self.M4.trainable_weights)            
        ############################SCALE 3###################################
        left_F3_grads = tape.gradient(self.losses_dict["D3"], self.left_conv6.trainable_weights)
        left_F03_grads = tape.gradient(self.losses_dict["D3"], self.left_conv5.trainable_weights)
        right_F3_grads = tape.gradient(self.losses_dict["D3"], self.right_conv6.trainable_weights)
        right_F03_grads = tape.gradient(self.losses_dict["D3"], self.right_conv5.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M3_grads = tape.gradient(self.losses_dict["D2"], self.M3.trainable_weights)            
        ############################SCALE 2###################################            
        left_F2_grads = tape.gradient(self.losses_dict["D2"], self.left_conv4.trainable_weights)
        left_F02_grads = tape.gradient(self.losses_dict["D2"], self.left_conv3.trainable_weights)
        right_F2_grads = tape.gradient(self.losses_dict["D2"], self.right_conv4.trainable_weights)
        right_F02_grads = tape.gradient(self.losses_dict["D2"], self.right_conv3.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M2_grads = tape.gradient(self.losses_dict["final_loss"], self.M2.trainable_weights) 
        ############################SCALE 1###################################
        # Scale 1 doesnt have a module, so need to use the loss from scales 2's module
        left_F1_grads = tape.gradient(self.losses_dict["D2"], self.left_conv2.trainable_weights)
        left_F01_grads = tape.gradient(self.losses_dict["D2"], self.left_conv1.trainable_weights)
        right_F1_grads = tape.gradient(self.losses_dict["D2"], self.right_conv2.trainable_weights)
        right_F01_grads = tape.gradient(self.losses_dict["D2"], self.right_conv1.trainable_weights)
        ############################REFINEMENT################################
        refinement_grads = tape.gradient(self.losses_dict["final_loss"], self.refinement_module.trainable_weights)


        #**************************Apply Gradients***************************
        #############################SCALE 6#################################
        self.optimizer.apply_gradients(zip(left_F6_grads, self.left_conv12.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F06_grads, self.left_conv11.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F6_grads, self.right_conv12.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F06_grads, self.right_conv11.trainable_weights))
        self.optimizer.apply_gradients(zip(M6_grads, self.M6.trainable_weights))        
        ############################SCALE 5###################################
        self.optimizer.apply_gradients(zip(left_F5_grads, self.left_conv10.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F05_grads, self.left_conv9.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F5_grads, self.right_conv10.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F05_grads, self.right_conv9.trainable_weights))
        self.optimizer.apply_gradients(zip(M5_grads, self.M5.trainable_weights))           
        ############################SCALE 4###################################
        self.optimizer.apply_gradients(zip(left_F4_grads, self.left_conv8.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F04_grads, self.left_conv7.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F4_grads, self.right_conv8.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F04_grads, self.right_conv7.trainable_weights))
        self.optimizer.apply_gradients(zip(M4_grads, self.M4.trainable_weights))  
        ############################SCALE 3###################################
        self.optimizer.apply_gradients(zip(left_F3_grads, self.left_conv6.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F03_grads, self.left_conv5.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F3_grads, self.right_conv6.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F03_grads, self.right_conv5.trainable_weights))
        self.optimizer.apply_gradients(zip(M3_grads, self.M3.trainable_weights))  
        ############################SCALE 2###################################
        self.optimizer.apply_gradients(zip(left_F2_grads, self.left_conv4.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F02_grads, self.left_conv3.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F2_grads, self.right_conv4.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F02_grads, self.right_conv3.trainable_weights))
        self.optimizer.apply_gradients(zip(M2_grads, self.M2.trainable_weights))    
        ############################SCALE 1###################################
        self.optimizer.apply_gradients(zip(left_F1_grads, self.left_conv2.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F01_grads, self.left_conv1.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F1_grads, self.right_conv2.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F01_grads, self.right_conv1.trainable_weights))
        ############################REFINEMENT################################
        self.optimizer.apply_gradients(zip(refinement_grads, self.refinement_module.trainable_weights))

        return self.losses_dict


    # Forward pass of the model
    def call(self, inputs, target=None, training=False):
        #print("\nStarted Call MADNet")
        # Left and right image inputs
        left_input = inputs["left_input"]
        right_input = inputs["right_input"]

        # print(f"Left input: {left_input.shape}")
        # print(f"Right input: {right_input.shape}")

        #######################PYRAMID FEATURES###############################
        # Left image feature pyramid (feature extractor)
        # F1
        self.left_pyramid = self.left_conv1(left_input)
        left_F1 = self.left_conv2(self.left_pyramid)
        # F2
        self.left_pyramid = self.left_conv3(left_F1)
        left_F2 = self.left_conv4(self.left_pyramid)
        # F3
        self.left_pyramid = self.left_conv5(left_F2)
        left_F3 = self.left_conv6(self.left_pyramid)
        # F4
        self.left_pyramid = self.left_conv7(left_F3)
        left_F4 = self.left_conv8(self.left_pyramid)
        # F5
        self.left_pyramid = self.left_conv9(left_F4)
        left_F5 = self.left_conv10(self.left_pyramid)
        # F6
        self.left_pyramid = self.left_conv11(left_F5)
        left_F6 = self.left_conv12(self.left_pyramid)


        # Right image feature pyramid (feature extractor)
        # F1
        self.right_pyramid = self.right_conv1(right_input)
        right_F1 = self.right_conv2(self.right_pyramid)
        # F2
        self.right_pyramid = self.right_conv3(right_F1)
        right_F2 = self.right_conv4(self.right_pyramid)
        # F3
        self.right_pyramid = self.right_conv5(right_F2)
        right_F3 = self.right_conv6(self.right_pyramid)
        # F4
        self.right_pyramid = self.right_conv7(right_F3)
        right_F4 = self.right_conv8(self.right_pyramid)
        # F5
        self.right_pyramid = self.right_conv9(right_F4)
        right_F5 = self.right_conv10(self.right_pyramid)
        # F6
        self.right_pyramid = self.right_conv11(right_F5)
        right_F6 = self.right_conv12(self.right_pyramid)

        #############################SCALE 6#################################
        D6, self.losses_dict["D6"] = self.M6(left_F6, right_F6, None, training)            
        ############################SCALE 5###################################
        D5, self.losses_dict["D5"] = self.M5(left_F5, right_F5, D6, training)       
        ############################SCALE 4###################################
        D4, self.losses_dict["D4"] = self.M4(left_F4, right_F4, D5, training) 
        ############################SCALE 3###################################
        D3, self.losses_dict["D3"] = self.M3(left_F3, right_F3, D4, training)
        ############################SCALE 2###################################
        D2, self.losses_dict["D2"] = self.M2(left_F2, right_F2, D3, training)  
        ############################REFINEMENT################################
        final_disparity, self.losses_dict["final_loss"] = self.refinement_module(left_F2, D2, left_input, right_input, training)  

        
        # Override warping losses using the groundtruth (if its available)
        # For supervised training only
        if training == True and target is not None:
            self.losses_dict["final_loss"] = self.loss_fn(target, final_disparity)
            # Resize groundtruth to match lower resolution layers
            # using same target and decreasing resolution for each layer
            target = tf.image.resize(target, [tf.shape(D2)[1], tf.shape(D2)[2]], method="bilinear")
            self.losses_dict["D2"] = self.loss_fn(target, D2)
            target = tf.image.resize(target, [tf.shape(D3)[1], tf.shape(D3)[2]], method="bilinear")
            self.losses_dict["D3"] = self.loss_fn(target, D3)
            target = tf.image.resize(target, [tf.shape(D4)[1], tf.shape(D4)[2]], method="bilinear")
            self.losses_dict["D4"] = self.loss_fn(target, D4)
            target = tf.image.resize(target, [tf.shape(D5)[1], tf.shape(D5)[2]], method="bilinear")
            self.losses_dict["D5"] = self.loss_fn(target, D5)
            target = tf.image.resize(target, [tf.shape(D6)[1], tf.shape(D6)[2]], method="bilinear")
            self.losses_dict["D6"] = self.loss_fn(target, D6)            
    
        return final_disparity

    def predict_step(self, data):
        """The logic for one inference step.
        This method is called by `Model.make_predict_function`.
        This method contains the mathematical logic for one step of inference.
        By default this will run in MAD mode.
        In MAD mode a single module will be updated each inferencing step.
        MAD mode can be turned off by setting the models attribute MAD_predict = False.
        This will improve performance but the model will not improve using
        the self-supervised MAD offline training.

        Args:
            data: A nested structure of `Tensor`s.
        Returns:
            The result of one inference step, final disparity
        """
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            final_disparity = self(x, training=self.MAD_predict)


        #((((((((((((((((((((((((Select module for adaptation))))))))))))))))))))))))      
        if self.MAD_predict == True:
            # Convert losses to a probability distribution for Modular adaptation
            H = tf.nn.softmax(list(self.losses_dict.values()))
            best_prob = tf.reduce_max(H)


            
            H_dict = {key: value for key, value in zip(self.losses_dict.keys(), H)}
            # H_dict = {}
            # for key, value in zip(self.losses_dict.keys(), H):
            #     H_dict[key] = value


            
            # Only selecting the module with the highest probability (this can be changed to select multiple modules)
            adaptation_dict = {key: True if value == best_prob else False for key, value in H_dict.items()}
            #adaptation_dict = {}


            # print(f"\nlosses_dict: {self.losses_dict}")
            # print(f"H: {H}")
            # print(f"adaptation_dict: {adaptation_dict}")
        else:
            # Set all modules to not update
            # adaptation_dict = {}
            adaptation_dict = {key: False for key in self.losses_dict.keys()}

        #^^^^^^^^^^^^^^^^^^^^^^^^Compute + Apply Gradients^^^^^^^^^^^^^^^^^^^^^^^^
        if adaptation_dict["D6"]:
            #############################SCALE 6#################################
            left_F6_grads = tape.gradient(self.losses_dict["D6"], self.left_conv12.trainable_weights)
            left_F06_grads = tape.gradient(self.losses_dict["D6"], self.left_conv11.trainable_weights)
            right_F6_grads = tape.gradient(self.losses_dict["D6"], self.right_conv12.trainable_weights)
            right_F06_grads = tape.gradient(self.losses_dict["D6"], self.right_conv11.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M6_grads = tape.gradient(self.losses_dict["D5"], self.M6.trainable_weights) 
            # Applying gradients
            self.optimizer.apply_gradients(zip(left_F6_grads, self.left_conv12.trainable_weights))
            self.optimizer.apply_gradients(zip(left_F06_grads, self.left_conv11.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F6_grads, self.right_conv12.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F06_grads, self.right_conv11.trainable_weights))
            self.optimizer.apply_gradients(zip(M6_grads, self.M6.trainable_weights))             
        
        if adaptation_dict["D5"]:
            ############################SCALE 5###################################
            left_F5_grads = tape.gradient(self.losses_dict["D5"], self.left_conv10.trainable_weights)
            left_F05_grads = tape.gradient(self.losses_dict["D5"], self.left_conv9.trainable_weights)
            right_F5_grads = tape.gradient(self.losses_dict["D5"], self.right_conv10.trainable_weights)
            right_F05_grads = tape.gradient(self.losses_dict["D5"], self.right_conv9.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M5_grads = tape.gradient(self.losses_dict["D4"], self.M5.trainable_weights)   
            # Applying gradients
            self.optimizer.apply_gradients(zip(left_F5_grads, self.left_conv10.trainable_weights))
            self.optimizer.apply_gradients(zip(left_F05_grads, self.left_conv9.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F5_grads, self.right_conv10.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F05_grads, self.right_conv9.trainable_weights))
            self.optimizer.apply_gradients(zip(M5_grads, self.M5.trainable_weights))  

        if adaptation_dict["D4"]:
            ############################SCALE 4###################################
            left_F4_grads = tape.gradient(self.losses_dict["D4"], self.left_conv8.trainable_weights)
            left_F04_grads = tape.gradient(self.losses_dict["D4"], self.left_conv7.trainable_weights)
            right_F4_grads = tape.gradient(self.losses_dict["D4"], self.right_conv8.trainable_weights)
            right_F04_grads = tape.gradient(self.losses_dict["D4"], self.right_conv7.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M4_grads = tape.gradient(self.losses_dict["D3"], self.M4.trainable_weights)     
            # Applying gradients
            self.optimizer.apply_gradients(zip(left_F4_grads, self.left_conv8.trainable_weights))
            self.optimizer.apply_gradients(zip(left_F04_grads, self.left_conv7.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F4_grads, self.right_conv8.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F04_grads, self.right_conv7.trainable_weights))
            self.optimizer.apply_gradients(zip(M4_grads, self.M4.trainable_weights))         
        
        if adaptation_dict["D3"]:
            ############################SCALE 3###################################
            left_F3_grads = tape.gradient(self.losses_dict["D3"], self.left_conv6.trainable_weights)
            left_F03_grads = tape.gradient(self.losses_dict["D3"], self.left_conv5.trainable_weights)
            right_F3_grads = tape.gradient(self.losses_dict["D3"], self.right_conv6.trainable_weights)
            right_F03_grads = tape.gradient(self.losses_dict["D3"], self.right_conv5.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M3_grads = tape.gradient(self.losses_dict["D2"], self.M3.trainable_weights)            
            # Applying gradients
            self.optimizer.apply_gradients(zip(left_F3_grads, self.left_conv6.trainable_weights))
            self.optimizer.apply_gradients(zip(left_F03_grads, self.left_conv5.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F3_grads, self.right_conv6.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F03_grads, self.right_conv5.trainable_weights))
            self.optimizer.apply_gradients(zip(M3_grads, self.M3.trainable_weights))      

        if adaptation_dict["D2"]: 
            ############################SCALE 2###################################           
            left_F2_grads = tape.gradient(self.losses_dict["D2"], self.left_conv4.trainable_weights)
            left_F02_grads = tape.gradient(self.losses_dict["D2"], self.left_conv3.trainable_weights)
            right_F2_grads = tape.gradient(self.losses_dict["D2"], self.right_conv4.trainable_weights)
            right_F02_grads = tape.gradient(self.losses_dict["D2"], self.right_conv3.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M2_grads = tape.gradient(self.losses_dict["final_loss"], self.M2.trainable_weights) 
            # Applying gradients
            self.optimizer.apply_gradients(zip(left_F2_grads, self.left_conv4.trainable_weights))
            self.optimizer.apply_gradients(zip(left_F02_grads, self.left_conv3.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F2_grads, self.right_conv4.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F02_grads, self.right_conv3.trainable_weights))
            self.optimizer.apply_gradients(zip(M2_grads, self.M2.trainable_weights))    
            ############################SCALE 1###################################
            # Scale 1 doesnt have a module, so need to use the loss from scales 2's module
            left_F1_grads = tape.gradient(self.losses_dict["D2"], self.left_conv2.trainable_weights)
            left_F01_grads = tape.gradient(self.losses_dict["D2"], self.left_conv1.trainable_weights)
            right_F1_grads = tape.gradient(self.losses_dict["D2"], self.right_conv2.trainable_weights)
            right_F01_grads = tape.gradient(self.losses_dict["D2"], self.right_conv1.trainable_weights)
            # Applying gradients
            self.optimizer.apply_gradients(zip(left_F1_grads, self.left_conv2.trainable_weights))
            self.optimizer.apply_gradients(zip(left_F01_grads, self.left_conv1.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F1_grads, self.right_conv2.trainable_weights))
            self.optimizer.apply_gradients(zip(right_F01_grads, self.right_conv1.trainable_weights))
        
        if adaptation_dict["final_loss"]:
            ############################REFINEMENT################################
            refinement_grads = tape.gradient(self.losses_dict["final_loss"], self.refinement_module.trainable_weights)
            # Applying gradients
            self.optimizer.apply_gradients(zip(refinement_grads, self.refinement_module.trainable_weights))

        return final_disparity




model = MADNet(height=image_height, width=image_width, search_range=search_range, batch_size=batch_size)

model.compile(
    optimizer='adam', 
    run_eagerly = True   
)

# ---------------------------------------------------------------------------
# Train the model

# model.summary()
#tf.keras.utils.plot_model(model, "G:/My Drive/repos/Real-time-self-adaptive-deep-stereo/images/MADNet Model Structure.png", show_layer_names=True)


# --------------------------------------------------------------------------------
# Data Preperation

# class StereoGenerator(tf.keras.utils.Sequence):
#     """
#     This method is currently not working.
#     The Input data has shape (None, None, None, None) for each image when training
#     Takes paths to left and right stereo image directories
#     and creates a generator that returns a batch of left 
#     and right images.
    
#     """
#     def __init__(self, left_dir, right_dir, batch_size, height, width, shuffle):
#         self.left_dir = left_dir
#         self.right_dir = right_dir
#         self.batch_size = batch_size
#         self.height = height
#         self.width = width
#         self.shuffle = shuffle

#         self.left_paths = [path for path in os.listdir(left_dir) if os.path.isfile(f"{self.left_dir}/{path}")]
#         self.right_paths = [path for path in os.listdir(right_dir) if os.path.isfile(f"{self.right_dir}/{path}")]
#         # Check that there is a left image for every right image
#         self.num_left = len(self.left_paths)
#         self.num_right = len(self.right_paths)
#         if self.num_left != self.num_right:
#             raise ValueError(f"Number of right and left images do now match. Left number: {self.num_left}. Right number: {self.num_right}")
#         # Check if images names are identical
#         self.left_paths.sort()
#         self.right_paths.sort()
#         if self.left_paths != self.right_paths:
#             raise ValueError("Left and right image names do not match. Please make sure left and right image names are identical")

#     def __len__(self):
#         # Denotes the number of batches per epoch
#         return self.num_left // self.batch_size


#     def __get_image(self, image_dir, image_name):
#         # get a single image helper function
#         image = tf.keras.preprocessing.image.load_img(f"{image_dir}/{image_name}")
#         image_arr = tf.keras.preprocessing.image.img_to_array(image)
#         image_arr = tf.image.resize(image_arr, (self.height, self.width)).numpy()
#         return image_arr/255.

#     def __getitem__(self, batch_index):
#         index = batch_index * self.batch_size
#         left_batch = self.left_paths[index: self.batch_size + index]
#         right_batch = self.right_paths[index: self.batch_size + index]
#         print("\nInside Generator getitem")
#         print(f"left batch: {left_batch}")
#         print(f"right batch: {right_batch}")

#         left_images = tf.constant([self.__get_image(self.left_dir, image_name) for image_name in left_batch])
#         right_images = tf.constant([self.__get_image(self.right_dir, image_name) for image_name in right_batch])
#         return {'left_input': left_images, 'right_input': right_images}, None


# # steps_per_epoch = math.ceil(left_generator.samples / batch_size)        

# stereo_gen = StereoGenerator(
#     left_dir=left_dir, 
#     right_dir=right_dir, 
#     batch_size=batch_size, 
#     height=image_height, 
#     width=image_width,
#     shuffle=False
#     ) 






# path = left_dir + "/0000001.png"
# raw = tf.io.read_file(path)
# image = tf.io.decode_image(raw, channels=3, dtype=tf.float32, expand_animations=False)
# # Converts to float32 and normalises values
# #image = tf.image.convert_image_dtype(image, tf.float32)
# # Change dimensions to the desired model dimensions
# image = tf.image.resize(image, [540, 960], method="bilinear")
# #image = tf.reshape(image, [self.batch_size, self.height, self.width, 3])

class StereoDatasetCreator():
    """
    Takes paths to left and right stereo image directories
    and creates a dataset that returns a batch of left 
    and right images, (Optional) returns the disparities as a target
    using the disparities directories.
    Init Args:
        left_dir: path to left images folder
        right_dir: path to right images folder
        batch_size: desired batch size 
        height: desired height of the image (will be reshaped to this height if necessary)
        width: desired width of the image (will be reshaped to this width if necessary)
        shuffle: True/False
        (Optional) disp_dir: path to disparity maps folder
    Returns:
        object that can be called to return a tf.data.Dataset
        dataset will return values of the form: 
            {'left_input': (batch, height, width, 3), 'right_input': (batch, height, width, 3)}, (Optional) (batch, height, width, 1) else None

    This can prepare MADNet data for training/evaluation and prediction
    """
    def __init__(self, left_dir, right_dir, batch_size, height, width, shuffle, disp_dir=None):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.disp_dir = disp_dir
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shuffle = shuffle

        self.left_names = tf.constant([name for name in os.listdir(left_dir) if os.path.isfile(f"{self.left_dir}/{name}")])
        self.right_names = tf.constant([name for name in os.listdir(right_dir) if os.path.isfile(f"{self.right_dir}/{name}")])
        if self.disp_dir is not None:
            self.disp_names = tf.constant([name for name in os.listdir(disp_dir) if os.path.isfile(f"{self.disp_dir}/{name}")])

        # Check that there is a left image for every right image
        self.num_left = len(self.left_names)
        self.num_right = len(self.right_names)
        if self.num_left != self.num_right:
            raise ValueError(f"Number of right and left images do now match. Left number: {self.num_left}. Right number: {self.num_right}")

    def __get_image(self, path):
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
        image = tf.image.resize(image, [self.height, self.width], method="bilinear")
        return image

    def readPFM(self, file):
        """
        Load a pfm file as a numpy array
        Args:
            file: path to the file to be loaded
        Returns:
            content of the file as a numpy array
            with shape (height, width, channels)
        """
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dims = file.readline()
        try:
            width, height = list(map(int, dims.split()))
        except:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width, 1)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

    def __get_pfm(self, path):
        """
        Reads a single pfm disparity file and
        returns a numpy disparity map
        Args:
            path: path to the disparity file (will be in Tensor format, since its called in a graph)
        Returns:
            Tensor disparity map with shape (height, width, 1) 
        """
        # Convert tensor to a string
        path = path.numpy().decode("ascii")

        #disp_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        disp_map = self.readPFM(path)

        # Set inf values to 0 (0 is infinitely far away, so basically the same)
        disp_map[disp_map==np.inf] = 0
        # Normalize
        disp_map = disp_map/256.0
        # convert values to positive
        if disp_map.mean() < 0:
            disp_map *= -1
        # Change dimensions to the desired (height, width, channels)
        #disp_map = tf.expand_dims(disp_map, axis=-1)
        disp_map = tf.image.resize(disp_map, [self.height, self.width], method="bilinear")
        return disp_map

    def __process_single_batch(self, index):
        """
        Processes a single batch using index to find the files
        Args: 
            index: Tensor integer
        Returns:
            stereo input dictionary, target
        """
        left_name = self.left_names[index]
        right_name = self.right_names[index]
        left_image = self.__get_image(f"{self.left_dir}/" +  left_name)
        right_image = self.__get_image(f"{self.right_dir}/" + right_name)

        disp_map = None  
        if self.disp_dir is not None:
            disp_name = self.disp_names[index]  
            # wrapping in py_function so that the function can execute eagerly and run non tensor ops
            disp_map = tf.py_function(func=self.__get_pfm, inp=[f"{self.disp_dir}/" + disp_name], Tout=tf.float32)

        return {'left_input': left_image, 'right_input': right_image}, disp_map

    def __call__(self):
        """
        Creates and returns a tensorflow data.Dataset
        The dataset is shuffled, batched and prefetched
        """
        indexes = list(range(self.num_left))   
        indexes_ds = tf.data.Dataset.from_tensor_slices(indexes)
        if self.shuffle == True:
            indexes_ds.shuffle(buffer_size=self.num_left, seed=101, reshuffle_each_iteration=False)

        ds = indexes_ds.map(self.__process_single_batch)
        ds = ds.batch(batch_size=self.batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=10)
        return ds



stereo_dataset = StereoDatasetCreator(
    left_dir=left_dir, 
    right_dir=right_dir, 
    batch_size=batch_size, 
    height=image_height, 
    width=image_width,
    shuffle=False,
    #disp_dir=left_disp_dir
    ) 

train_ds = stereo_dataset()


# for index, element in enumerate(train_ds):
#     inputs, gt = element
#     cv2.imshow("left", inputs["left_input"].numpy()[0])
#     cv2.imshow("right", inputs["right_input"].numpy()[0])
#     cv2.imshow("groundtruth", gt.numpy()[0])
#     cv2.waitKey(0)
#     if index > 2:
#         break


# for index, element in enumerate(train_ds):
#     print(element)
#     inputs, gt = element

#     if index > 2:
#         break





# history = model.fit(
#     x=train_ds,
#     epochs=2,
#     verbose=2,
#     steps_per_epoch=5
# )

model.MAD_predict = True

disparity = model.predict(train_ds, steps=5)

print(f"disparity shape: {disparity.shape}")


cv2.imshow("predicted disp", disparity[0])
cv2.waitKey(0)

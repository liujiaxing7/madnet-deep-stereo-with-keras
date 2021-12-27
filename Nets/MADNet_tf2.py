import os
import tensorflow as tf
import numpy as np

print("\nTensorFlow Version: {}".format(tf.__version__))


# dummpy data for the images
image_height = 320
image_width = 1216
input_size = (image_height, image_width)
batch_size = 1 # Set batch size to none to have a variable batch size

search_range = 2 # maximum dispacement (ie. smallest disparity)

class SSIMLoss(tf.keras.losses.Loss):
    """
    SSIM dissimilarity measure
    Args:
        x: target image
        y: predicted image
    """
    def __init__(self, name="mean_SSIM_l1"):
        super(SSIMLoss, self).__init__(name=name)
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(3,3) ,strides=(1,1), padding='valid')

    def call(self, x, y):
        print("\nInside loss")
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

        print("\nCall stereo cost volume")
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

        # print(f"c1: {c1.shape}")
        # print(f"warp: {warp.shape}")
        # print(f"cost_curve: {cost_curve.shape}")

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
        self.x = None
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

    def call(self, input, disp, final_left, final_right):
        print(f"\nStarted call context network")
        volume = self.concat([input, disp])

        # print(f"input: {input.shape}")
        # print(f"disp: {disp.shape}")
        # print(f"volume: {volume.shape}")

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

        final_reprojection_loss = self.loss_fn(final_warped_left, final_left)
        # print(f"final_warped_left: {final_warped_left.shape}")
        # print(f"Final Reprojection Loss: {final_reprojection_loss}")         

        return final_disparity, final_reprojection_loss


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
        self.layer = layer
        self.loss_fn = SSIMLoss()
        self.cost_volume = StereoCostVolume(name=f"cost_{self.layer}")
        self.stereo_estimator = StereoEstimator(name=f"volume_filtering_{self.layer}")

    def call(self, left, right, prev_disp=None):
        print(f"\nStarted call ModuleM {self.layer}")

        print(f"left: {left.shape}")
        print(f"right: {right.shape}")

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
        # add loss estimating the reprojection accuracy of the pyramid level (for self supervised training/MAD)
        reprojection_loss = self.loss_fn(warped_left, left)

        # print(f"warped_left: {warped_left.shape}")
        # print(f"Reprojection Loss: {reprojection_loss}")

        costs = self.cost_volume(left, warped_left, self.search_range)

        # Get the disparity using cost volume between left and warped left images
        module_disparity = self.stereo_estimator(costs)

        # print(f"module_disparity: {module_disparity.shape}")

        return module_disparity, reprojection_loss


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

    def train_step(self, inputs):
        print("\nInside train_step MADNet")
        # Left and right image inputs
        #print(f"Inputs: {inputs}")

        left_input = inputs[0]["left_input"]
        right_input = inputs[0]["right_input"]        

        print(f"Left input: {left_input.shape}")
        print(f"Right input: {right_input.shape}")
        with tf.GradientTape(persistent=True) as tape:
            #######################PYRAMID FEATURES###############################
            # Left image feature pyramid (feature extractor)
            # F1
            left_F01 = self.left_conv1(left_input)
            left_F1 = self.left_conv2(left_F01)
            # F2
            left_F02 = self.left_conv3(left_F1)
            left_F2 = self.left_conv4(left_F02)
            # F3
            left_F03 = self.left_conv5(left_F2)
            left_F3 = self.left_conv6(left_F03)
            # F4
            left_F04 = self.left_conv7(left_F3)
            left_F4 = self.left_conv8(left_F04)
            # F5
            left_F05 = self.left_conv9(left_F4)
            left_F5 = self.left_conv10(left_F05)
            # F6
            left_F06 = self.left_conv11(left_F5)
            left_F6 = self.left_conv12(left_F06)


            # Right image feature pyramid (feature extractor)
            # F1
            right_F01 = self.right_conv1(right_input)
            right_F1 = self.right_conv2(right_F01)
            # F2
            right_F02 = self.right_conv3(right_F1)
            right_F2 = self.right_conv4(right_F02)
            # F3
            right_F03 = self.right_conv5(right_F2)
            right_F3 = self.right_conv6(right_F03)
            # F4
            right_F04 = self.right_conv7(right_F3)
            right_F4 = self.right_conv8(right_F04)
            # F5
            right_F05 = self.right_conv9(right_F4)
            right_F5 = self.right_conv10(right_F05)
            # F6
            right_F06 = self.right_conv11(right_F5)
            right_F6 = self.right_conv12(right_F06)

            losses = {}
            #############################SCALE 6#################################
            D6, losses["D6"] = self.M6(left_F6, right_F6)            
            ############################SCALE 5###################################
            D5, losses["D5"] = self.M5(left_F5, right_F5, D6)       
            ############################SCALE 4###################################
            D4, losses["D4"] = self.M4(left_F4, right_F4, D5) 
            ############################SCALE 3###################################
            D3, losses["D3"] = self.M3(left_F3, right_F3, D4)
            ############################SCALE 2###################################
            D2, losses["D2"] = self.M2(left_F2, right_F2, D3)  
            ############################REFINEMENT################################
            final_disparity, losses["final_reprojection_loss"] = self.refinement_module(left_F2, D2, left_input, right_input)  


        #((((((((((((((((((((((((Select modules using losses))))))))))))))))))))))))






        #^^^^^^^^^^^^^^^^^^^^^^^^Compute Gradients^^^^^^^^^^^^^^^^^^^^^^^^
        print(f"\n\nComputing gradients now")
        #############################SCALE 6#################################
        left_F6_grads = tape.gradient(losses["D6"], self.left_conv12.trainable_weights)
        left_F06_grads = tape.gradient(losses["D6"], self.left_conv11.trainable_weights)
        right_F6_grads = tape.gradient(losses["D6"], self.right_conv12.trainable_weights)
        right_F06_grads = tape.gradient(losses["D6"], self.right_conv11.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M6_grads = tape.gradient(losses["D5"], self.M6.trainable_weights)        
        ############################SCALE 5###################################
        left_F5_grads = tape.gradient(losses["D5"], self.left_conv10.trainable_weights)
        left_F05_grads = tape.gradient(losses["D5"], self.left_conv9.trainable_weights)
        right_F5_grads = tape.gradient(losses["D5"], self.right_conv10.trainable_weights)
        right_F05_grads = tape.gradient(losses["D5"], self.right_conv9.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M5_grads = tape.gradient(losses["D4"], self.M5.trainable_weights)   
        ############################SCALE 4###################################
        left_F4_grads = tape.gradient(losses["D4"], self.left_conv8.trainable_weights)
        left_F04_grads = tape.gradient(losses["D4"], self.left_conv7.trainable_weights)
        right_F4_grads = tape.gradient(losses["D4"], self.right_conv8.trainable_weights)
        right_F04_grads = tape.gradient(losses["D4"], self.right_conv7.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M4_grads = tape.gradient(losses["D3"], self.M4.trainable_weights)            
        ############################SCALE 3###################################
        left_F3_grads = tape.gradient(losses["D3"], self.left_conv6.trainable_weights)
        left_F03_grads = tape.gradient(losses["D3"], self.left_conv5.trainable_weights)
        right_F3_grads = tape.gradient(losses["D3"], self.right_conv6.trainable_weights)
        right_F03_grads = tape.gradient(losses["D3"], self.right_conv5.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M3_grads = tape.gradient(losses["D2"], self.M3.trainable_weights)            
        ############################SCALE 2###################################            
        left_F2_grads = tape.gradient(losses["D2"], self.left_conv4.trainable_weights)
        left_F02_grads = tape.gradient(losses["D2"], self.left_conv3.trainable_weights)
        right_F2_grads = tape.gradient(losses["D2"], self.right_conv4.trainable_weights)
        right_F02_grads = tape.gradient(losses["D2"], self.right_conv3.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M2_grads = tape.gradient(losses["final_reprojection_loss"], self.M2.trainable_weights) 
        ############################SCALE 1###################################
        # Scale 1 doesnt have a module, so need to use the loss from scales 2's module
        left_F1_grads = tape.gradient(losses["D2"], self.left_conv2.trainable_weights)
        left_F01_grads = tape.gradient(losses["D2"], self.left_conv1.trainable_weights)
        right_F1_grads = tape.gradient(losses["D2"], self.right_conv2.trainable_weights)
        right_F01_grads = tape.gradient(losses["D2"], self.right_conv1.trainable_weights)
        ############################REFINEMENT################################
        refinement_grads = tape.gradient(losses["final_reprojection_loss"], self.refinement_module.trainable_weights)



        #**************************Apply Gradients***************************
        print("\n\nApplying gradients now")
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




        print(f"\n\nAfter applying gradients")


        return losses

    # def make_train_function(self, force=False):
    #     """Creates a function that executes one step of training.
    #     This method can be overridden to support custom training logic.
    #     This method is called by `Model.fit` and `Model.train_on_batch`.
    #     Typically, this method directly controls `tf.function` and
    #     `tf.distribute.Strategy` settings, and delegates the actual training
    #     logic to `Model.train_step`.
    #     This function is cached the first time `Model.fit` or
    #     `Model.train_on_batch` is called. The cache is cleared whenever
    #     `Model.compile` is called. You can skip the cache and generate again the
    #     function with `force=True`.
    #     Args:
    #         force: Whether to regenerate the train function and skip the cached
    #         function if available.
    #     Returns:
    #         Function. The function created by this method should accept a
    #         `tf.data.Iterator`, and return a `dict` containing values that will
    #         be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
    #         `{'loss': 0.2, 'accuracy': 0.7}`.
    #     """
    #     # helper functions
    #     def _minimum_control_deps(outputs):
    #         """Returns the minimum control dependencies to ensure step succeeded."""
    #         if tf.executing_eagerly():
    #             return []  # Control dependencies not needed.
    #         outputs = tf.nest.flatten(outputs, expand_composites=True)
    #         for out in outputs:
    #             # Variables can't be control dependencies.
    #             if not isinstance(out, tf.Variable):
    #                 return [out]  # Return first Tensor or Op from outputs.
    #         return []  # No viable Tensor or Op to use for control deps.

    #     # def reduce_per_replica(values, strategy, reduction='first'):
    #     #     """Reduce PerReplica objects.
    #     #     Args:
    #     #         values: Structure of `PerReplica` objects or `Tensor`s. `Tensor`s are
    #     #         returned as-is.
    #     #         strategy: `tf.distribute.Strategy` object.
    #     #         reduction: One of 'first', 'concat'.
    #     #     Returns:
    #     #         Structure of `Tensor`s.
    #     #     """

    #     #     def _reduce(v):
    #     #         """Reduce a single `PerReplica` object."""
    #     #         if reduction == 'concat' and _collective_all_reduce_multi_worker(strategy):
    #     #             return _multi_worker_concat(v, strategy)
    #     #         if not _is_per_replica_instance(v):
    #     #             return v
    #     #         elif reduction == 'first':
    #     #             return strategy.unwrap(v)[0]
    #     #         elif reduction == 'concat':
    #     #             if _is_tpu_multi_host(strategy):
    #     #                 return _tpu_multi_host_concat(v, strategy)a
    #     #             else:
    #     #                 return concat(strategy.unwrap(v))
    #     #         else:
    #     #             raise ValueError('`reduction` must be "first" or "concat". Received: '
    #     #                             f'reduction={reduction}.')

    #     #     return tf.nest.map_structure(_reduce, values)


    #     if self.train_function is not None and not force:
    #         return self.train_function

    #     def step_function(model, iterator):
    #         """Runs a single training step."""
    #         print("\nInside step_function")
    #         print(f"Iterator: {iterator}")


    #         def run_step(data):
    #             outputs = model.train_step(data)
    #             # Ensure counter is updated only if `train_step` succeeds.
    #             with tf.control_dependencies(_minimum_control_deps(outputs)):
    #                 model._train_counter.assign_add(1)  # pylint: disable=protected-access
    #             return outputs

    #         data = next(iterator)
    #         #print(f"data in step_function: {data}")

    #         outputs = model.distribute_strategy.run(run_step, args=(data,))
    #         # outputs = reduce_per_replica(
    #         #     outputs, self.distribute_strategy, reduction='first')
    #         #write_scalar_summaries(outputs, step=model._train_counter)  # pylint: disable=protected-access
    #         return outputs

    #     if (self._steps_per_execution is None or
    #         self._steps_per_execution.numpy().item() == 1):

    #         def train_function(iterator):
    #             """Runs a training execution with one step."""
    #             return step_function(self, iterator)

    #     else:

    #         def train_function(iterator):
    #             """Runs a training execution with multiple steps."""
    #             for _ in tf.range(self._steps_per_execution):
    #                 outputs = step_function(self, iterator)
    #             return outputs

    #     if not self.run_eagerly:
    #         train_function = tf.function(
    #             train_function, experimental_relax_shapes=True)
    #         self.train_tf_function = train_function

    #     self.train_function = train_function

    #     if self._cluster_coordinator:
    #         self.train_function = lambda iterator: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
    #             train_function, args=(iterator,))

    #     return self.train_function


    # Forward pass of the model
    def call(self, inputs):
        print("\nStarted Call MADNet")
        # Left and right image inputs
        #left_input, right_input = inputs
        left_input = inputs["left_input"]
        right_input = inputs["right_input"]

        #print(f"Inputs: {inputs}")
        print(f"Left input: {left_input.shape}")
        print(f"Right input: {right_input.shape}")

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

        losses = {}
        #############################SCALE 6#################################
        D6, losses["D6"] = self.M6(left_F6, right_F6)            
        ############################SCALE 5###################################
        D5, losses["D5"] = self.M5(left_F5, right_F5, D6)       
        ############################SCALE 4###################################
        D4, losses["D4"] = self.M4(left_F4, right_F4, D5) 
        ############################SCALE 3###################################
        D3, losses["D3"] = self.M3(left_F3, right_F3, D4)
        ############################SCALE 2###################################
        D2, losses["D2"] = self.M2(left_F2, right_F2, D3)  
        ############################REFINEMENT################################
        final_disparity, losses["final_reprojection_loss"] = self.refinement_module(left_F2, D2, left_input, right_input)     
    
        print(f"Losses: \n{losses}")
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


class StereoGenerator(tf.keras.utils.Sequence):
    """
    Takes paths to left and right stereo image directories
    and creates a generator that returns a batch of left 
    and right images.
    
    """
    def __init__(self, left_dir, right_dir, batch_size, height, width, shuffle):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shuffle = shuffle

        self.left_paths = [path for path in os.listdir(left_dir) if os.path.isfile(f"{self.left_dir}/{path}")]
        self.right_paths = [path for path in os.listdir(right_dir) if os.path.isfile(f"{self.right_dir}/{path}")]
        # Check that there is a left image for every right image
        self.num_left = len(self.left_paths)
        self.num_right = len(self.right_paths)
        if self.num_left != self.num_right:
            raise ValueError(f"Number of right and left images do now match. Left number: {self.num_left}. Right number: {self.num_right}")
        # Check if images names are identical
        self.left_paths.sort()
        self.right_paths.sort()
        if self.left_paths != self.right_paths:
            raise ValueError("Left and right image names do not match. Please make sure left and right image names are identical")

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.num_left // self.batch_size


    def __get_image(self, image_dir, image_name):
        # get a single image helper function
        image = tf.keras.preprocessing.image.load_img(f"{image_dir}/{image_name}")
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (self.height, self.width)).numpy()
        return image_arr/255.

    def __getitem__(self, batch_index):
        index = batch_index * self.batch_size
        left_batch = self.left_paths[index: self.batch_size + index]
        right_batch = self.right_paths[index: self.batch_size + index]
        print("\nInside Generator getitem")
        print(f"left batch: {left_batch}")
        print(f"right batch: {right_batch}")

        left_images = tf.constant([self.__get_image(self.left_dir, image_name) for image_name in left_batch])
        right_images = tf.constant([self.__get_image(self.right_dir, image_name) for image_name in right_batch])
        return {'left_input': left_images, 'right_input': right_images}, None


left_dir = "G:/My Drive/Data Files/2011_09_26_drive_0002_sync/left/data"
right_dir = "G:/My Drive/Data Files/2011_09_26_drive_0002_sync/right/data"

# steps_per_epoch = math.ceil(left_generator.samples / batch_size)        

stereo_gen = StereoGenerator(
    left_dir=left_dir, 
    right_dir=right_dir, 
    batch_size=batch_size, 
    height=image_height, 
    width=image_width,
    shuffle=False
    ) 

# gen = lambda: (image for image in stereo_gen)

# # Convert generator to tf.data.Dataset
# stereo_dataset = tf.data.Dataset.from_generator(
#     gen,
#     output_signature=(
#         {'left_input': tf.TensorSpec(shape=[None, None, None, None]), 'right_input': tf.TensorSpec(shape=[None, None, None, None])},
#         #tf.TensorSpec(shape=[None])
#     )
# )


left_input = tf.constant(np.random.random((1, 1, image_height, image_width, 3)), dtype=tf.float32)
right_input = tf.constant(np.random.random((1, 1, image_height, image_width, 3)), dtype=tf.float32)

stereo_dataset = tf.data.Dataset.from_tensor_slices(({'left_input': left_input, 'right_input': right_input}, None))

stereo_dataset.element_spec






history = model.fit(
    x=stereo_dataset,
    epochs=2,
    verbose=2,
    #steps_per_epoch=steps_per_epoch
)

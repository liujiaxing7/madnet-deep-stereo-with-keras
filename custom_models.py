import tensorflow as tf
import numpy as np
from keras.engine import data_adapter
from matplotlib import cm
from losses_and_metrics import SSIMLoss, ReconstructionLoss, calculate_metrics


def colorize_img(value, vmin=None, vmax=None, cmap='jet'):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size,height, width,1]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')
    
    Returns a 3D tensor of shape [batch_size,height, width,3].
    """
    # Uncomment the code below if disparity isnt normalised already
    # # normalize
    # vmin = tf.reduce_min(value) if vmin is None else vmin
    # vmax = tf.reduce_max(value) if vmax is None else vmax
    # value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = tf.cast(tf.round(value[:,:,:,0]*255), dtype=tf.int32)

    # gather
    color_map = cm.get_cmap(cmap)
    colors = color_map(np.arange(256))[:,:3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    return value


# https://github.com/philferriere/tfoptflow/blob/bdc7a72e78008d1cd6db46e4667dffc2bab1fe9e/tfoptflow/core_costvol.py
class StereoCostVolume(tf.keras.layers.Layer):
    """Build cost volume for associating a pixel from the left image with its corresponding pixels in the right image.
    Args:
        c1: Level of the feature pyramid of the left image
        warp: Warped level of the feature pyramid of the right image
        search_range: Search range (maximum displacement)
    """
    def __init__(self, name="cost_volume", **kwargs):
        super(StereoCostVolume, self).__init__(name=name, **kwargs)

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

    def get_config(self):
        config = super(StereoCostVolume, self).get_config()
        return config


class BuildIndices(tf.keras.layers.Layer):
    """
	Given a flow or disparity generate the coordinates 
    of source pixels to sample from [batch, height_t, width_t, 2]
    Args:
	    coords: Generic optical flow or disparity 
    Returns:
        coordinates to sample from.   
    
    """

    def __init__(self, name="build_indices", batch_size=1, **kwargs):
        super(BuildIndices, self).__init__(name=name, **kwargs)
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

    def get_config(self):
        config = super(BuildIndices, self).get_config()
        config.update({"batch_size": self.batch_size})
        return config


class Warp(tf.keras.layers.Layer):
    """
    Construct a new image by bilinear sampling from the input image.
    The right image is warpt into the lefts position.
    Points falling outside the source image boundary have value 0.
    Args:
        imgs: source right images to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t, width_t, 2]. 
            height_t/width_t correspond to the dimensions of the outputimage (don't need to be the same as height_s/width_s). 
            The two channels correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels],
        which ideally is very similar to the left image
    """
    def __init__(self, name="warp", **kwargs):
        super(Warp, self).__init__(name=name, **kwargs)

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

    def get_config(self):
        config = super(Warp, self).get_config()
        return config


class StereoContextNetwork(tf.keras.Model):
    """
    Final Layer in MADNet.
    Calculates the reprojection loss if training=True.
    Args:
        input: left_F2 tensor
        disp: D2 disparity from M2 module
        final_left: full resolution RGB left image
        final_right: full resolution RGB right image
    Returns:
        Full resolution disparity in float32 normalized 0-1
    """

    def __init__(self, name="residual_refinement_network", batch_size=1, **kwargs):
        super(StereoContextNetwork, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        act = tf.keras.layers.Activation(tf.nn.leaky_relu)
        self.loss = None
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

        if training == True:
            # warp right image with final disparity to get final reprojection loss
            final_coords = self.concat([final_disparity, tf.zeros_like(final_disparity)])
            final_indices = self.build_indices(final_coords)
            # Warp the right image into the left using final disparity
            final_warped_left = self.warp(final_right, final_indices) 
            # reprojection loss   
            self.loss = self.loss_fn(final_warped_left, final_left)      

        return final_disparity

    def get_config(self):
        config = super(StereoContextNetwork, self).get_config()
        config.update({
            "loss": self.loss, 
            "act": self.act, 
            "batch_size": self.batch_size,
            "loss_fn": self.loss_fn,
            "x": self.x,
            "context1": self.context1, 
            "context2": self.context2,
            "context3": self.context3,
            "context4": self.context4,
            "context5": self.context5,
            "context6": self.context6,
            "context7": self.context7,
            "add": self.add,
            "concat": self.concat,
            "warp": self.warp,
            "build_indices": self.build_indices
            })
        return config



class StereoEstimator(tf.keras.Model):
    """
    This is the stereo estimation network at resolution n.
    It uses the costs (from the pixel difference between the warped right image 
    and the left image) combined with the upsampled disparity from the previous
    layer (when the layer is not the last layer).

    The output is predicted disparity for the network at resolution n.
    """

    def __init__(self, name="volume_filtering", **kwargs):
        super(StereoEstimator, self).__init__(name=name, **kwargs)
        act = tf.keras.layers.Activation(tf.nn.leaky_relu)
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

    def get_config(self):
        config = super(StereoEstimator, self).get_config()
        config.update({
            "act": self.act, 
            "disp1": self.disp1, 
            "disp2": self.disp2,
            "disp3": self.disp3,
            "disp4": self.disp4,
            "disp5": self.disp5,
            "disp6": self.disp6,
            "concat": self.concat,
            })
        return config


class ModuleM(tf.keras.Model):
    """
    Module MX is a sub-module of MADNet, which can be trained individually for 
    online adaptation using the MAD (Modular ADaptaion) method.
    """
    def __init__(self, name="MX", layer="X", search_range=2, batch_size=1, **kwargs):
        super(ModuleM, self).__init__(name=name, **kwargs)
        self.search_range = search_range
        self.batch_size = batch_size
        self.loss = None
        self.layer = layer
        # Loss function for self-supervised training (no groundtruth disparity)
        self.loss_fn = SSIMLoss()
        self.cost_volume = StereoCostVolume(name=f"cost_{self.layer}")
        self.stereo_estimator = StereoEstimator(name=f"volume_filtering_{self.layer}")

    def call(self, left, right, prev_disp=None, training=False):

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
            self.loss = self.loss_fn(warped_left, left)

        costs = self.cost_volume(left, warped_left, self.search_range)

        # Get the disparity using cost volume between left and warped left images
        module_disparity = self.stereo_estimator(costs)

        return module_disparity

    def get_config(self):
        config = super(ModuleM, self).get_config()
        config.update({
            "layer": self.layer, 
            "search_range": self.search_range, 
            "batch_size": self.batch_size,
            "loss": self.loss,
            "loss_fn": self.loss_fn,
            "cost_volume": self.cost_volume,
            "stereo_estimator": self.stereo_estimator
            })
        return config


# ------------------------------------------------------------------------
# Model Creation
class MADNet(tf.keras.Model):
    """
    The main MADNet model.
    Contains the logic for training, evaluation and and prediction.

    Training contains 2 modes, full MAD (unsupervised training) and 
    supervised training with groundtruth disparities.

    Prediction can run with no adaptation, full MAD or inbetween,
    update 1-5 modules. The number of modules to adapt can be selected 
    by changeing num_adapt_modules attribute. Adaptation can be turned off
    by setting the attribute MAD_predict = False.  

    The keras methods .fit, .predict and .evaluate all work with this model.
    """
    def __init__(self, name="MADNet", height=320, width=1216, search_range=2, batch_size=1, **kwargs):
        super(MADNet, self).__init__(name=name, **kwargs)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.search_range = search_range
        # for converstion between tensor and dict
        self.module_indexes = {"D6": 0, "D5": 1, "D4": 2, "D3": 3, "D2": 4, "final_loss": 5} 
        # Selects whether to perform MAD when running .predict
        self.MAD_predict = True 
        # Selects the number of modules to perform MAD during predict. 
        # Default is 1, >= 6 is full adaptation
        self.num_adapt_modules = 1
        # Loss function for supervised training (with groundtruth)
        self.loss_fn = ReconstructionLoss() 

        act = tf.keras.layers.Activation(tf.nn.leaky_relu)
        # Left image feature pyramid (feature extractor)
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
        # Left, right image inputs and groundtruth target disparity
        inputs, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        #inputs, gt = data
        left_input = inputs["left_input"]
        right_input = inputs["right_input"]        

        # Check the image shape and resize if necessary
        left_shape = left_input.shape[:3]
        right_shape = right_input.shape[:3]  
        desired_shape = (self.batch_size, self.height, self.width)


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
            final_disparity = self(inputs={'left_input': left_input, 'right_input': right_input}, target=gt, training=True)


        # Tensorboard images
        tf.summary.image('01_predicted_disparity', colorize_img(final_disparity, cmap='jet'), step=self._train_counter, max_outputs=1)
        if gt is not None:
            tf.summary.image('02_groundtruth_disparity', colorize_img(gt, cmap='jet'), step=self._train_counter, max_outputs=1)
        tf.summary.image('03_left_image', left_input, step=self._train_counter, max_outputs=1)
        tf.summary.image('04_right_image', right_input, step=self._train_counter, max_outputs=1)


        #((((((((((((((((((((((((Select modules using losses))))))))))))))))))))))))
        # Not selecting modules to update in train_step
        # Will either perform offline supervised adaptation
        # or full MAD (depending on whether gt images are provided)

        #^^^^^^^^^^^^^^^^^^^^^^^^Compute Gradients^^^^^^^^^^^^^^^^^^^^^^^^
        #############################SCALE 6#################################
        left_F6_grads = tape.gradient(self.M6.loss, self.left_conv12.trainable_weights)
        left_F06_grads = tape.gradient(self.M6.loss, self.left_conv11.trainable_weights)
        right_F6_grads = tape.gradient(self.M6.loss, self.right_conv12.trainable_weights)
        right_F06_grads = tape.gradient(self.M6.loss, self.right_conv11.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M6_grads = tape.gradient(self.M5.loss, self.M6.trainable_weights)        
        ############################SCALE 5###################################
        left_F5_grads = tape.gradient(self.M5.loss, self.left_conv10.trainable_weights)
        left_F05_grads = tape.gradient(self.M5.loss, self.left_conv9.trainable_weights)
        right_F5_grads = tape.gradient(self.M5.loss, self.right_conv10.trainable_weights)
        right_F05_grads = tape.gradient(self.M5.loss, self.right_conv9.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M5_grads = tape.gradient(self.M4.loss, self.M5.trainable_weights)   
        ############################SCALE 4###################################
        left_F4_grads = tape.gradient(self.M4.loss, self.left_conv8.trainable_weights)
        left_F04_grads = tape.gradient(self.M4.loss, self.left_conv7.trainable_weights)
        right_F4_grads = tape.gradient(self.M4.loss, self.right_conv8.trainable_weights)
        right_F04_grads = tape.gradient(self.M4.loss, self.right_conv7.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M4_grads = tape.gradient(self.M3.loss, self.M4.trainable_weights)            
        ############################SCALE 3###################################
        left_F3_grads = tape.gradient(self.M3.loss, self.left_conv6.trainable_weights)
        left_F03_grads = tape.gradient(self.M3.loss, self.left_conv5.trainable_weights)
        right_F3_grads = tape.gradient(self.M3.loss, self.right_conv6.trainable_weights)
        right_F03_grads = tape.gradient(self.M3.loss, self.right_conv5.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M3_grads = tape.gradient(self.M2.loss, self.M3.trainable_weights)            
        ############################SCALE 2###################################            
        left_F2_grads = tape.gradient(self.M2.loss, self.left_conv4.trainable_weights)
        left_F02_grads = tape.gradient(self.M2.loss, self.left_conv3.trainable_weights)
        right_F2_grads = tape.gradient(self.M2.loss, self.right_conv4.trainable_weights)
        right_F02_grads = tape.gradient(self.M2.loss, self.right_conv3.trainable_weights)
        # The current modules output is used in the following modules loss function 
        M2_grads = tape.gradient(self.refinement_module.loss, self.M2.trainable_weights) 
        ############################SCALE 1###################################
        # Scale 1 doesnt have a module, so need to use the loss from scales 2's module
        left_F1_grads = tape.gradient(self.M2.loss, self.left_conv2.trainable_weights)
        left_F01_grads = tape.gradient(self.M2.loss, self.left_conv1.trainable_weights)
        right_F1_grads = tape.gradient(self.M2.loss, self.right_conv2.trainable_weights)
        right_F01_grads = tape.gradient(self.M2.loss, self.right_conv1.trainable_weights)
        ############################REFINEMENT################################
        refinement_grads = tape.gradient(self.refinement_module.loss, self.refinement_module.trainable_weights)


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

        losses_metrics_dict = {
            "D6_loss": self.M6.loss, 
            "D5_loss": self.M5.loss, 
            "D4_loss": self.M4.loss, 
            "D3_loss": self.M3.loss, 
            "D2_loss": self.M2.loss, 
            "loss": self.refinement_module.loss
            }

        if gt is not None:
            metrics_dict = calculate_metrics(gt, final_disparity, 3)
            losses_metrics_dict.update(metrics_dict)

        return losses_metrics_dict
  

    # Forward pass of the model
    def call(self, inputs, target=None, training=False):
        """
        This is the forward pass of the model.
        Call is used in training, evaluation and prediction.
        """
        # Left and right image inputs
        left_input = inputs["left_input"]
        right_input = inputs["right_input"]

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

        # Select loss calculation method
        use_MAD = False
        if training == True and target is None:
            use_MAD = True

        #############################SCALE 6#################################
        D6 = self.M6(left_F6, right_F6, None, use_MAD)     
        ############################SCALE 5###################################
        D5 = self.M5(left_F5, right_F5, D6, use_MAD)    
        ############################SCALE 4###################################
        D4 = self.M4(left_F4, right_F4, D5, use_MAD) 
        ############################SCALE 3###################################
        D3 = self.M3(left_F3, right_F3, D4, use_MAD)
        ############################SCALE 2###################################
        D2 = self.M2(left_F2, right_F2, D3, use_MAD) 
        ############################REFINEMENT################################
        final_disparity = self.refinement_module(left_F2, D2, left_input, right_input, use_MAD) 

        # Override warping losses using loss from the groundtruth (if its available)
        # For supervised training only
        if training == True and target is not None:
            self.refinement_module.loss = self.loss_fn(target, final_disparity)
            # Resize groundtruth to match lower resolution layers
            # using same target and decreasing resolution for each layer
            target = tf.image.resize(target, [tf.shape(D2)[1], tf.shape(D2)[2]], method="bilinear")
            self.M2.loss = self.loss_fn(target, D2)
            target = tf.image.resize(target, [tf.shape(D3)[1], tf.shape(D3)[2]], method="bilinear")
            self.M3.loss = self.loss_fn(target, D3)
            target = tf.image.resize(target, [tf.shape(D4)[1], tf.shape(D4)[2]], method="bilinear")
            self.M4.loss = self.loss_fn(target, D4)
            target = tf.image.resize(target, [tf.shape(D5)[1], tf.shape(D5)[2]], method="bilinear")
            self.M5.loss = self.loss_fn(target, D5)
            target = tf.image.resize(target, [tf.shape(D6)[1], tf.shape(D6)[2]], method="bilinear")
            self.M6.loss = self.loss_fn(target, D6)           
    
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
        Adaption on 1-5 modules is only working with eager mode on.
        To change models to adapt update the models attribute num_adapt_modules to 1-6.

        Args:
            data: A nested structure of `Tensor`s.
        Returns:
            The result of one inference step, final disparity
        """
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            final_disparity = self(x, training=self.MAD_predict)


        #((((((((((((((((((((((((Select module/s for adaptation))))))))))))))))))))))))
        losses = [
                self.M6.loss, 
                self.M5.loss,
                self.M4.loss,
                self.M3.loss,
                self.M2.loss,
                self.refinement_module.loss
                ]
        none_losses = [None, None, None, None, None, None]
        if self.MAD_predict == True and self.num_adapt_modules >= 6:
            # Full MAD, update all modules
            adapt_losses = losses
            adapt_dict = {"D6": True, "D5": True, "D4": True, "D3": True, "D2": True, "final_loss": True} 
        elif self.MAD_predict == True and self.num_adapt_modules == 1:
            # Default MAD mode
            # This currently only works with eager execution turned on
            # Convert losses to a probability distribution for Modular adaptation
            H = tf.nn.softmax(losses)
            adapt_index = tf.cast(tf.argmax(H), tf.int32)

            adapt_dict = {"D6": False, "D5": False, "D4": False, "D3": False, "D2": False, "final_loss": False}

            for key in adapt_dict.keys():
                # This is not graph executable (boolean unsupported op on tensor)
                if tf.equal(self.module_indexes[key], adapt_index):
                    adapt_dict[key] = True    
                    break             

        elif self.MAD_predict == True:
            # Adapting only the highest probability modules 
            # This currently only works with eager execution turned on
            # Convert losses to a probability distribution for Modular adaptation
            H = tf.nn.softmax(losses)
            adapt_indexes = tf.math.top_k(H, k=self.num_adapt_modules).indices 
            adapt_losses = tf.gather(losses, adapt_indexes)

            adapt_dict = {"D6": False, "D5": False, "D4": False, "D3": False, "D2": False, "final_loss": False}

            for key in adapt_dict.keys():
                # This is not graph executable (tensor is not iterable)
                if losses[self.module_indexes[key]] in adapt_losses:
                    adapt_dict[key] = True                      
        else:
            # No adaptation, only a forward pass is performed
            losses = none_losses  
            adapt_dict = {"D6": False, "D5": False, "D4": False, "D3": False, "D2": False, "final_loss": False}   


        #^^^^^^^^^^^^^^^^^^^^^^^^Compute + Apply Gradients^^^^^^^^^^^^^^^^^^^^^^^^
        if self.M6.loss is not None and adapt_dict["D6"]:
            #############################SCALE 6#################################
            left_F6_grads = tape.gradient(self.M6.loss, self.left_conv12.trainable_weights)
            left_F06_grads = tape.gradient(self.M6.loss, self.left_conv11.trainable_weights)
            right_F6_grads = tape.gradient(self.M6.loss, self.right_conv12.trainable_weights)
            right_F06_grads = tape.gradient(self.M6.loss, self.right_conv11.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M6_grads = tape.gradient(self.M5.loss, self.M6.trainable_weights) 
        else:
            # set grads to zero (no change to weights)
            left_F6_grads = [tf.zeros_like(weights) for weights in self.left_conv12.trainable_weights]
            left_F06_grads = [tf.zeros_like(weights) for weights in self.left_conv11.trainable_weights]
            right_F6_grads = [tf.zeros_like(weights) for weights in self.right_conv12.trainable_weights]
            right_F06_grads = [tf.zeros_like(weights) for weights in self.right_conv11.trainable_weights]
            # The current modules output is used in the following modules loss function 
            M6_grads = [tf.zeros_like(weights) for weights in self.M6.trainable_weights] 

        # Applying gradients
        self.optimizer.apply_gradients(zip(left_F6_grads, self.left_conv12.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F06_grads, self.left_conv11.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F6_grads, self.right_conv12.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F06_grads, self.right_conv11.trainable_weights))
        self.optimizer.apply_gradients(zip(M6_grads, self.M6.trainable_weights))             
        
        if self.M5.loss is not None and adapt_dict["D5"]:
            ############################SCALE 5###################################
            left_F5_grads = tape.gradient(self.M5.loss, self.left_conv10.trainable_weights)
            left_F05_grads = tape.gradient(self.M5.loss, self.left_conv9.trainable_weights)
            right_F5_grads = tape.gradient(self.M5.loss, self.right_conv10.trainable_weights)
            right_F05_grads = tape.gradient(self.M5.loss, self.right_conv9.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M5_grads = tape.gradient(self.M4.loss, self.M5.trainable_weights)   
        else:
            # set grads to zero (no change to weights)
            left_F5_grads = [tf.zeros_like(weights) for weights in self.left_conv10.trainable_weights]
            left_F05_grads = [tf.zeros_like(weights) for weights in self.left_conv9.trainable_weights]
            right_F5_grads = [tf.zeros_like(weights) for weights in self.right_conv10.trainable_weights]
            right_F05_grads = [tf.zeros_like(weights) for weights in self.right_conv9.trainable_weights]
            # The current modules output is used in the following modules loss function 
            M5_grads = [tf.zeros_like(weights) for weights in self.M5.trainable_weights] 

        # Applying gradients
        self.optimizer.apply_gradients(zip(left_F5_grads, self.left_conv10.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F05_grads, self.left_conv9.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F5_grads, self.right_conv10.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F05_grads, self.right_conv9.trainable_weights))
        self.optimizer.apply_gradients(zip(M5_grads, self.M5.trainable_weights))  

        if self.M4.loss is not None and adapt_dict["D4"]:
            ############################SCALE 4###################################
            left_F4_grads = tape.gradient(self.M4.loss, self.left_conv8.trainable_weights)
            left_F04_grads = tape.gradient(self.M4.loss, self.left_conv7.trainable_weights)
            right_F4_grads = tape.gradient(self.M4.loss, self.right_conv8.trainable_weights)
            right_F04_grads = tape.gradient(self.M4.loss, self.right_conv7.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M4_grads = tape.gradient(self.M3.loss, self.M4.trainable_weights)  
        else:
            # set grads to zero (no change to weights)
            left_F4_grads = [tf.zeros_like(weights) for weights in self.left_conv8.trainable_weights]
            left_F04_grads = [tf.zeros_like(weights) for weights in self.left_conv7.trainable_weights]
            right_F4_grads = [tf.zeros_like(weights) for weights in self.right_conv8.trainable_weights]
            right_F04_grads = [tf.zeros_like(weights) for weights in self.right_conv7.trainable_weights]
            # The current modules output is used in the following modules loss function 
            M4_grads = [tf.zeros_like(weights) for weights in self.M4.trainable_weights]             

        # Applying gradients
        self.optimizer.apply_gradients(zip(left_F4_grads, self.left_conv8.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F04_grads, self.left_conv7.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F4_grads, self.right_conv8.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F04_grads, self.right_conv7.trainable_weights))
        self.optimizer.apply_gradients(zip(M4_grads, self.M4.trainable_weights))         
        
        if self.M3.loss is not None and adapt_dict["D3"]:
            ############################SCALE 3###################################
            left_F3_grads = tape.gradient(self.M3.loss, self.left_conv6.trainable_weights)
            left_F03_grads = tape.gradient(self.M3.loss, self.left_conv5.trainable_weights)
            right_F3_grads = tape.gradient(self.M3.loss, self.right_conv6.trainable_weights)
            right_F03_grads = tape.gradient(self.M3.loss, self.right_conv5.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M3_grads = tape.gradient(self.M2.loss, self.M3.trainable_weights)  
        else:
            # set grads to zero (no change to weights)
            left_F3_grads = [tf.zeros_like(weights) for weights in self.left_conv6.trainable_weights]
            left_F03_grads = [tf.zeros_like(weights) for weights in self.left_conv5.trainable_weights]
            right_F3_grads = [tf.zeros_like(weights) for weights in self.right_conv6.trainable_weights]
            right_F03_grads = [tf.zeros_like(weights) for weights in self.right_conv5.trainable_weights]
            # The current modules output is used in the following modules loss function 
            M3_grads = [tf.zeros_like(weights) for weights in self.M3.trainable_weights]              

        # Applying gradients
        self.optimizer.apply_gradients(zip(left_F3_grads, self.left_conv6.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F03_grads, self.left_conv5.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F3_grads, self.right_conv6.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F03_grads, self.right_conv5.trainable_weights))
        self.optimizer.apply_gradients(zip(M3_grads, self.M3.trainable_weights))      

        if self.M2.loss is not None and adapt_dict["D2"]:
            ############################SCALE 2###################################           
            left_F2_grads = tape.gradient(self.M2.loss, self.left_conv4.trainable_weights)
            left_F02_grads = tape.gradient(self.M2.loss, self.left_conv3.trainable_weights)
            right_F2_grads = tape.gradient(self.M2.loss, self.right_conv4.trainable_weights)
            right_F02_grads = tape.gradient(self.M2.loss, self.right_conv3.trainable_weights)
            # The current modules output is used in the following modules loss function 
            M2_grads = tape.gradient(self.refinement_module.loss, self.M2.trainable_weights) 

            ############################SCALE 1###################################
            # Scale 1 doesnt have a module, so need to use the loss from scales 2's module
            left_F1_grads = tape.gradient(self.M2.loss, self.left_conv2.trainable_weights)
            left_F01_grads = tape.gradient(self.M2.loss, self.left_conv1.trainable_weights)
            right_F1_grads = tape.gradient(self.M2.loss, self.right_conv2.trainable_weights)
            right_F01_grads = tape.gradient(self.M2.loss, self.right_conv1.trainable_weights)
        else:
            # set grads to zero (no change to weights)
            left_F2_grads = [tf.zeros_like(weights) for weights in self.left_conv4.trainable_weights]
            left_F02_grads = [tf.zeros_like(weights) for weights in self.left_conv3.trainable_weights]
            right_F2_grads = [tf.zeros_like(weights) for weights in self.right_conv4.trainable_weights]
            right_F02_grads = [tf.zeros_like(weights) for weights in self.right_conv3.trainable_weights]
            # The current modules output is used in the following modules loss function 
            M2_grads = [tf.zeros_like(weights) for weights in self.M2.trainable_weights] 
            # set grads to zero (no change to weights)
            left_F1_grads = [tf.zeros_like(weights) for weights in self.left_conv2.trainable_weights]
            left_F01_grads = [tf.zeros_like(weights) for weights in self.left_conv1.trainable_weights]
            right_F1_grads = [tf.zeros_like(weights) for weights in self.right_conv2.trainable_weights]
            right_F01_grads = [tf.zeros_like(weights) for weights in self.right_conv1.trainable_weights]            

        # Applying gradients
        self.optimizer.apply_gradients(zip(left_F2_grads, self.left_conv4.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F02_grads, self.left_conv3.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F2_grads, self.right_conv4.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F02_grads, self.right_conv3.trainable_weights))
        self.optimizer.apply_gradients(zip(M2_grads, self.M2.trainable_weights))  
        # Applying gradients
        self.optimizer.apply_gradients(zip(left_F1_grads, self.left_conv2.trainable_weights))
        self.optimizer.apply_gradients(zip(left_F01_grads, self.left_conv1.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F1_grads, self.right_conv2.trainable_weights))
        self.optimizer.apply_gradients(zip(right_F01_grads, self.right_conv1.trainable_weights))
        
        if self.refinement_module.loss is not None and adapt_dict["final_loss"]:
            ############################REFINEMENT################################
            refinement_grads = tape.gradient(self.refinement_module.loss, self.refinement_module.trainable_weights)
        else:
            # set grads to zero (no change to weights)
            refinement_grads = [tf.zeros_like(weights) for weights in self.refinement_module.trainable_weights]            
        # Applying gradients
        self.optimizer.apply_gradients(zip(refinement_grads, self.refinement_module.trainable_weights))

        return final_disparity

    def get_config(self):
        config = super(MADNet, self).get_config()
        config.update({
            "height": self.height, 
            "width": self.width,
            "search_range": self.search_range, 
            "batch_size": self.batch_size,
            "module_indexes": self.module_indexes,
            "MAD_predict": self.MAD_predict,
            "num_adapt_modules": self.num_adapt_modules,
            "loss_fn": self.loss_fn,
            "act": self.act,
            "left_conv1": self.left_conv1, 
            "left_conv2": self.left_conv2,
            "left_conv3": self.left_conv3,
            "left_conv4": self.left_conv4,
            "left_conv5": self.left_conv5,
            "left_conv6": self.left_conv6,
            "left_conv7": self.left_conv7,
            "left_conv8": self.left_conv8,
            "left_conv9": self.left_conv9,
            "left_conv10": self.left_conv10,
            "left_conv11": self.left_conv11,
            "left_conv12": self.left_conv12,
            "right_conv1": self.right_conv1, 
            "right_conv2": self.right_conv2,
            "right_conv3": self.right_conv3,
            "right_conv4": self.right_conv4,
            "right_conv5": self.right_conv5,
            "right_conv6": self.right_conv6,
            "right_conv7": self.right_conv7,
            "right_conv8": self.right_conv8,
            "right_conv9": self.right_conv9,
            "right_conv10": self.right_conv10,
            "right_conv11": self.right_conv11,
            "right_conv12": self.right_conv12,
            "M6": self.M6,
            "M5": self.M5,
            "M4": self.M4,
            "M3": self.M3,
            "M2": self.M2,
            "refinement_module": self.refinement_module
            })
        return config



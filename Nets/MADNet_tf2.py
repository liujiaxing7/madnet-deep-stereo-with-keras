import tensorflow as tf
import numpy as np
import math
from Losses.loss_factory import mean_SSIM_L1

print("\nTensorFlow Version: {}".format(tf.__version__))


# dummpy data for the images
image_height = 320
image_width = 1216
input_size = (image_height, image_width)
batch_size = 1 # Set batch size to none to have a variable batch size

search_range = 2 # maximum dispacement (ie. smallest disparity)

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
        # # add loss estimating the reprojection accuracy of the pyramid level (for self supervised training/MAD)
        # reprojection_loss = mean_SSIM_L1(warp, c1)
        # self.add_loss(reprojection_loss)

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

    def __init__(self, name="build_indices"):
        super(BuildIndices, self).__init__(name=name)

    def call(self, coords):

        batches, height, width, channels = coords.get_shape().as_list()

        pixel_coords = np.ones((1, height, width, 2), dtype=np.float32)
        batches_coords = np.ones((batches, height, width, 1), dtype=np.float32)

        for i in range(0, batches):
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

    def __init__(self, name="residual_refinement_network"):
        super(StereoContextNetwork, self).__init__(name=name)
        act = tf.keras.layers.Activation(tf.nn.leaky_relu)
        self.x = None
        self.context1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=1, padding="same", activation=act, use_bias=True, name="context1")
        self.context2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=2, padding="same", activation=act, use_bias=True, name="context2")
        self.context3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=4, padding="same", activation=act, use_bias=True, name="context3")
        self.context4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), dilation_rate=8, padding="same", activation=act, use_bias=True, name="context4")
        self.context5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), dilation_rate=16, padding="same", activation=act, use_bias=True, name="context5")
        self.context6 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), dilation_rate=1, padding="same", activation=act, use_bias=True, name="context6")
        self.context7 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), dilation_rate=1, padding="same", activation="linear", use_bias=True, name="context7")

    def call(self, input, disp):

        volume = tf.keras.layers.concatenate([input, disp], axis=-1)
        # Need to check if context was created previously,
        # so variable doesnt get created multiple times (for autograph)
        if self.x is None:
            self.x = self.context1(volume)
            self.x = self.context2(self.x)
            self.x = self.context3(self.x)
            self.x = self.context4(self.x)
            self.x = self.context5(self.x)
            self.x = self.context6(self.x)
            self.x = self.context7(self.x)

        final_disp = tf.keras.layers.add([disp, self.x], name="final_disp")

        return final_disp


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
        self.x = None
        self.disp1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp1")
        self.disp2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp2")
        self.disp3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp3")
        self.disp4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp4")
        self.disp5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp5")
        self.disp6 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding="same", activation="linear", use_bias=True, name="disp6")

    def call(self, costs, upsampled_disp=None):
        if upsampled_disp is not None:
            volume = tf.keras.layers.concatenate([costs, upsampled_disp], axis=-1)
        else:
            volume = costs
        # Need to check if disp was created previously,
        # so variable doesnt get created multiple times (for autograph)
        if self.x is None:
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
    def __init__(self, name="MX", layer="X", search_range=2):
        super(ModuleM, self).__init__(name=name)
        self.module_disparity = None
        self.final_disparity = None
        self.context_disparity = None
        self.search_range = search_range
        self.layer = layer
        self.cost_volume = StereoCostVolume(name=f"cost_{layer}")
        self.stereo_estimator = StereoEstimator(name=f"volume_filtering_{layer}")
        self.context_network = StereoContextNetwork()

    def call(self, left, right, prev_disp=None, is_final_module=False):

        height, width = (left.shape.as_list()[1], left.shape.as_list()[2])
        # Check if module disparity was previously calculated to prevent retracing (for autograph)
        if self.module_disparity is not None:
            # Check if layer is the bottom of the pyramid
            if prev_disp is not None:
                # Upsample disparity from previous layer
                upsampled_disp = tf.keras.layers.Resizing(name=f"upsampled_disp_{self.layer}", height=height, width=width, interpolation='bilinear')(prev_disp)
                coords = tf.keras.layers.concatenate([upsampled_disp, tf.zeros_like(upsampled_disp)], -1)
                indices = BuildIndices(name=f"build_indices_{self.layer}")(coords)
                # Warp the right image into the left using upsampled disparity
                warped_left = Warp(name=f"warp_{self.layer}")(right, indices)
            else:
                # No previous disparity exits, so use right image instead of warped left
                warped_left = right

            # add loss estimating the reprojection accuracy of the pyramid level (for self supervised training/MAD)
            reprojection_loss = mean_SSIM_L1(warped_left, left)
            self.add_loss(reprojection_loss)

            costs = self.cost_volume(left, warped_left, self.search_range)
            # Get the disparity using cost volume between left and warped left images
            self.module_disparity = self.stereo_estimator(costs)

        # Add the residual refinement network to the final layer
        # also check if disparity was created previously (for autograph)
        if is_final_module and self.final_disparity is not None:
            self.context_disparity = self.context_network(left, self.module_disparity)
            self.final_disparity = tf.keras.layers.Resizing(name="final_disparity", height=height, width=width, interpolation='bilinear')(self.context_disparity)


        return self.final_disparity if is_final_module else self.module_disparity




# ------------------------------------------------------------------------
# Model Creation
class MADNet(tf.keras.Model):
    """
    The main MADNet model
    """
    def __init__(self, name="MADNet", height=320, width=1216, search_range=2):
        super(MADNet, self).__init__(name=name)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.search_range = search_range

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
        self.M6 = ModuleM(name="M6", layer="6", search_range=self.search_range)
        ############################SCALE 5###################################
        self.M5 = ModuleM(name="M5", layer="5", search_range=self.search_range)
        ############################SCALE 4###################################
        self.M4 = ModuleM(name="M4", layer="4", search_range=self.search_range)
        ############################SCALE 3###################################
        self.M3 = ModuleM(name="M3", layer="3", search_range=self.search_range)
        ############################SCALE 2###################################
        self.M2 = ModuleM(name="M2", layer="2", search_range=self.search_range)




    # Forward pass of the model
    def call(self, inputs):
        # # Left and right image inputs
        # left_input = tf.keras.Input(shape=(self.height, self.width, 3, ), batch_size=self.batch_size, name="left_image_input", dtype=tf.float32)
        # right_input = tf.keras.Input(shape=(self.height, self.width, 3, ), batch_size=self.batch_size, name="right_image_input", dtype=tf.float32)
        left_input, right_input = inputs

        #######################PYRAMID FEATURES###############################
        # Left image feature pyramid (feature extractor)
        if self.left_pyramid is None:
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
        if self.right_pyramid is None:
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
        D6 = self.M6(left_F6, right_F6)
        losses["D6"] = D6.losses
        self.add_loss(losses["D6"])
        ############################SCALE 5###################################
        D5 = self.M5(left_F5, right_F5, D6)
        losses["D5"] = D5.losses
        self.add_loss(losses["D5"])        
        ############################SCALE 4###################################
        D4 = self.M4(left_F4, right_F4, D5)
        losses["D4"] = D4.losses     
        self.add_loss(losses["D4"])   
        ############################SCALE 3###################################
        D3 = self.M3(left_F3, right_F3, D4)
        losses["D3"] = D3.losses     
        self.add_loss(losses["D3"])   
        ############################SCALE 2###################################
        D2 = self.M2(left_F2, right_F2, D3, True)
        losses["D2"] = D2.losses  
        self.add_loss(losses["D2"])      

        return D2

model = MADNet()

# MADNet = tf.keras.Model(inputs=[left_input, right_input], outputs=M2, name="MADNet")


model.compile(
    optimizer='adam'
)

#model.summary()
#tf.keras.utils.plot_model(MADNet, "./images/MADNet Model Structure.png", show_layer_names=True)


# --------------------------------------------------------------------------------
# Data Preperation

left_dir = "G:/My Drive/Data Files/2011_09_26_drive_0002_sync/left"
right_dir = "G:/My Drive/Data Files/2011_09_26_drive_0002_sync/right"

# Create datagenerator object for loading and preparing image data for training
left_dataflow_kwargs = dict(
    directory = left_dir, 
    target_size = input_size, 
    class_mode = None,
    batch_size = batch_size,
    shuffle = False,     
    interpolation = "bilinear",
    )

right_dataflow_kwargs = dict(
    directory = right_dir, 
    target_size = input_size, 
    class_mode = None,
    batch_size = batch_size,
    shuffle = False,     
    interpolation = "bilinear",
    )


# Normalize pixel values
datagen_args = dict(
    rescale = 1./255
        )

datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args)

left_generator = datagen.flow_from_directory(**left_dataflow_kwargs)
right_generator = datagen.flow_from_directory(**right_dataflow_kwargs)

def generator(left_generator, right_generator):
    """Combines the left and right image generators into a 
        single image generator with two inputs for training.
        
        Make sure the left and right images have the same ID,
        otherwise the order might change which will pair the wrong
        left and right images."""
    while True:
        left = left_generator.next()
        right = right_generator.next()
        yield [left, right], None

steps_per_epoch = math.ceil(left_generator.samples / batch_size)


# ---------------------------------------------------------------------------
# Train the model

history = model.fit(
    x=generator(left_generator, right_generator),
    batch_size=batch_size,
    epochs=1,
    verbose=2,
    steps_per_epoch=steps_per_epoch
)




# # Stereo estimator model
# def _stereo_estimator(num_filters=1, model_name="fgc-volume-filtering"):
#     volume = tf.keras.Input(shape=(None, None, num_filters, ), name="cost_volume", dtype=tf.float32)

#     disp = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp1")(volume)
#     disp = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp2")(disp)
#     disp = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp3")(disp)
#     disp = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp4")(disp)
#     disp = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp5")(disp)
#     disp = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding="same", activation=None, use_bias=True, name="disp6")(disp)

#     return tf.keras.Model(inputs=[volume], outputs=[disp], name=model_name)

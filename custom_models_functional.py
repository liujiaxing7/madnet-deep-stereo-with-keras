import tensorflow as tf
import numpy as np
from keras.engine import data_adapter
from matplotlib import cm


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
def StereoCostVolume(name="cost_volume", search_range=2):
    """Build cost volume for associating a pixel from the left image with its corresponding pixels in the right image.
    Args:
        c1: Level of the feature pyramid of the left image
        warp: Warped level of the feature pyramid of the right image
        search_range: Search range (maximum displacement)
    """
    def _block(inputs):

        def internal_fn(inputs):
            c1, warp = inputs
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

        inputs = [tf.keras.layers.Input(shape=input.shape[1:]) for input in inputs]
        output = internal_fn(inputs)

        cost_model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
        cost_model.shape = output.shape

        return cost_model

    return _block


def BuildIndices(name="build_indices", batch_size=1):
    """
    Given a flow or disparity generate the coordinates
    of source pixels to sample from [batch, height_t, width_t, 2]
    Args:
        coords: Generic optical flow or disparity
    Returns:
        coordinates to sample from.

    """
    def _block(coords):

        def internal_fn(coords):
            _, height, width, _ = coords.get_shape().as_list()

            pixel_coords = np.ones((1, height, width, 2), dtype=np.float32)
            batches_coords = np.ones((batch_size, height, width, 1), dtype=np.float32)

            for i in range(0, batch_size):
                batches_coords[i][:][:][:] = i
            # build pixel coordinates and their disparity
            for i in range(0, height):
                for j in range(0, width):
                    pixel_coords[0][i][j][0] = j
                    pixel_coords[0][i][j][1] = i

            pixel_coords = tf.constant(pixel_coords, tf.float32)
            output = tf.concat([batches_coords, pixel_coords + coords], -1)
            return output

        inputs = tf.keras.layers.Input(shape=coords.shape[1:])
        output = internal_fn(inputs)

        indices_model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
        indices_model.shape = output.shape

        return indices_model
    return _block


def Warp(name="warp"):
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

    def _block(inputs):
        def internal_fn(inputs):
            imgs, coords = inputs
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

        inputs = [tf.keras.layers.Input(shape=input.shape[1:]) for input in inputs]
        output = internal_fn(inputs)

        warp_model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
        warp_model.shape = output.shape

        return warp_model

    return _block


def StereoContextNetwork(name="residual_refinement_network", batch_size=1, output_height=320, output_width=1216):
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
    act = tf.keras.layers.Activation(tf.nn.leaky_relu)
    context1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=1, padding="same", activation=act, use_bias=True, name="context1")
    context2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=2, padding="same", activation=act, use_bias=True, name="context2")
    context3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=4, padding="same", activation=act, use_bias=True, name="context3")
    context4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), dilation_rate=8, padding="same", activation=act, use_bias=True, name="context4")
    context5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), dilation_rate=16, padding="same", activation=act, use_bias=True, name="context5")
    context6 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), dilation_rate=1, padding="same", activation=act, use_bias=True, name="context6")
    context7 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), dilation_rate=1, padding="same", activation="linear", use_bias=True, name="context7")
    add = tf.keras.layers.Add(name="context_disp")
    concat = tf.keras.layers.Concatenate(axis=-1)

    def _block(inputs):
        def internal_fn(inputs):
            input, disp = inputs
            #volume = concat([input, disp])
            volume = tf.keras.layers.concatenate([input, disp], axis=-1)

            x = context1(volume)
            x = context2(x)
            x = context3(x)
            x = context4(x)
            x = context5(x)
            x = context6(x)
            x = context7(x)

            context_disp = add([disp, x])
            final_disparity = tf.keras.layers.Resizing(name="final_disparity", height=output_height, width=output_width, interpolation='bilinear')(context_disp)

            return final_disparity

        inputs = [tf.keras.layers.Input(shape=input.shape[1:]) for input in inputs]
        output = internal_fn(inputs)

        refinement_model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
        refinement_model.shape = output.shape
        return refinement_model

    return _block


def StereoEstimator(name="volume_filtering"):
    """
    This is the stereo estimation network at resolution n.
    It uses the costs (from the pixel difference between the warped right image 
    and the left image) combined with the upsampled disparity from the previous
    layer (when the layer is not the last layer).

    The output is predicted disparity for the network at resolution n.
    """
    act = tf.keras.layers.Activation(tf.nn.leaky_relu)
    disp1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp1")
    disp2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp2")
    disp3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp3")
    disp4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp4")
    disp5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp5")
    disp6 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding="same", activation="linear", use_bias=True, name="disp6")
    concat = tf.keras.layers.Concatenate(axis=-1)

    def _block(inputs):
        def internal_fn(inputs):
            if type(inputs) is list:
                costs, upsampled_disp = inputs
                # volume = concat([costs, upsampled_disp])
                volume = tf.keras.layers.concatenate([costs, upsampled_disp], axis=-1)
            else:
                volume = inputs

            x = disp1(volume)
            x = disp2(x)
            x = disp3(x)
            x = disp4(x)
            x = disp5(x)
            x = disp6(x)
            return x

        if type(inputs) is list:
            inputs = [tf.keras.layers.Input(shape=input.shape[1:]) for input in inputs]
        else:
            inputs = tf.keras.layers.Input(shape=inputs.shape[1:])

        output = internal_fn(inputs)

        estimator_model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
        estimator_model.shape = output.shape
        return estimator_model

    return _block

def ModuleM(name, layer, search_range=2, batch_size=1):
    """
    Module MX is a sub-module of MADNet, which can be trained individually for 
    online adaptation using the MAD (Modular ADaptaion) method.
    """
    cost_volume = StereoCostVolume(name=f"cost_{layer}", search_range=search_range)
    stereo_estimator = StereoEstimator(name=f"volume_filtering_{layer}")
    build_indices = BuildIndices(name=f"build_indices_{layer}", batch_size=batch_size)
    warp = Warp(name=f"warp_{layer}")

    def _block(inputs):
        def internal_fn(inputs):
            # Check if layer is the bottom of the pyramid
            if len(inputs) == 3:
                left, right, prev_disp = inputs
                # Upsample disparity from previous layer
                upsampled_disp = tf.keras.layers.Resizing(name=f"upsampled_disp_{layer}", height=height, width=width, interpolation='bilinear')(prev_disp)
                coords = tf.keras.layers.concatenate([upsampled_disp, tf.zeros_like(upsampled_disp)], -1)
                indices = build_indices(coords)
                # Warp the right image into the left using upsampled disparity
                warped_left = warp([right, indices])
            else:
                left, right = inputs
                # No previous disparity exits, so use right image instead of warped left
                warped_left = right

            costs = cost_volume([left, warped_left])

            # Get the disparity using cost volume between left and warped left images
            if len(inputs) == 3:
                module_disparity = stereo_estimator([costs, prev_disp])
            else:
                module_disparity = stereo_estimator(costs)

            return module_disparity

        if len(inputs) == 3:
            inputs = [tf.keras.layers.Input(shape=input.shape[1:]) for input in inputs]
        else:
            inputs = [tf.keras.layers.Input(shape=inputs[0].shape[1:]), tf.keras.layers.Input(shape=inputs[1].shape[1:])]
        output = internal_fn(inputs)
        #output = tf.keras.layers.Lambda(internal_fn)(inputs)


        module_model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
        module_model.shape = output.shape
        return module_model

    return _block




height = 320
width = 1216
search_range = 2
batch_size = 1

# Initializing the layers
act = tf.keras.layers.Activation(tf.nn.leaky_relu)
# Left image feature pyramid (feature extractor)
# F1
left_conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv1", 
input_shape=(height, width, 3, ))
left_conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv2")
# F2
left_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv3")
left_conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv4")
# F3
left_conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv5")
left_conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv6")
# F4
left_conv7 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv7")
left_conv8 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv8")
# F5
left_conv9 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv9")
left_conv10 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv10")
# F6
left_conv11 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv11")
left_conv12 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv12")       
# Right image feature pyramid (feature extractor)
# F1
right_conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv1", 
input_shape=(height, width, 3, ))
right_conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv2")
# F2
right_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv3")
right_conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv4")
# F3
right_conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv5")
right_conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv6")
# F4
right_conv7 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv7")
right_conv8 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv8")
# F5
right_conv9 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv9")
right_conv10 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv10")
# F6
right_conv11 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv11")
right_conv12 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv12")

#############################SCALE 6#################################
M6 = ModuleM(name="M6", layer="6", search_range=search_range, batch_size=batch_size)
############################SCALE 5###################################
M5 = ModuleM(name="M5", layer="5", search_range=search_range, batch_size=batch_size)
############################SCALE 4###################################
M4 = ModuleM(name="M4", layer="4", search_range=search_range, batch_size=batch_size)
############################SCALE 3###################################
M3 = ModuleM(name="M3", layer="3", search_range=search_range, batch_size=batch_size)
############################SCALE 2###################################
M2 = ModuleM(name="M2", layer="2", search_range=search_range, batch_size=batch_size)
############################REFINEMENT################################
refinement_module = StereoContextNetwork(batch_size=batch_size, output_height=height, output_width=width)


# Build the model
# Left and right image inputs
left_input = tf.keras.layers.Input(shape=[height, width, 3])
right_input = tf.keras.layers.Input(shape=[height, width, 3])

#######################PYRAMID FEATURES###############################
# Left image feature pyramid (feature extractor)
# F1
left_pyramid = left_conv1(left_input)
left_F1 = left_conv2(left_pyramid)
# F2
left_pyramid = left_conv3(left_F1)
left_F2 = left_conv4(left_pyramid)
# F3
left_pyramid = left_conv5(left_F2)
left_F3 = left_conv6(left_pyramid)
# F4
left_pyramid = left_conv7(left_F3)
left_F4 = left_conv8(left_pyramid)
# F5
left_pyramid = left_conv9(left_F4)
left_F5 = left_conv10(left_pyramid)
# F6
left_pyramid = left_conv11(left_F5)
left_F6 = left_conv12(left_pyramid)

# Right image feature pyramid (feature extractor)
# F1
right_pyramid = right_conv1(right_input)
right_F1 = right_conv2(right_pyramid)
# F2
right_pyramid = right_conv3(right_F1)
right_F2 = right_conv4(right_pyramid)
# F3
right_pyramid = right_conv5(right_F2)
right_F3 = right_conv6(right_pyramid)
# F4
right_pyramid = right_conv7(right_F3)
right_F4 = right_conv8(right_pyramid)
# F5
right_pyramid = right_conv9(right_F4)
right_F5 = right_conv10(right_pyramid)
# F6
right_pyramid = right_conv11(right_F5)
right_F6 = right_conv12(right_pyramid)


#############################SCALE 6#################################
D6 = M6([left_F6, right_F6])
############################SCALE 5###################################
D5 = M5([left_F5, right_F5, D6])
############################SCALE 4###################################
D4 = M4([left_F4, right_F4, D5])
############################SCALE 3###################################
D3 = M3([left_F3, right_F3, D4])
############################SCALE 2###################################
D2 = M2([left_F2, right_F2, D3])
############################REFINEMENT################################
final_disparity = refinement_module([left_F2, D2])


model = tf.keras.Model(inputs={"left_input": left_input, "right_input": right_input}, outputs=final_disparity, name="MADNet")
model.summary()

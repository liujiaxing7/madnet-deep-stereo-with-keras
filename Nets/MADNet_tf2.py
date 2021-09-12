import tensorflow as tf
import numpy as np

print("\nTensorFlow Version: {}".format(tf.__version__))


# dummpy data for the images
image_height = 320
image_width = 1216
batch_size = 1 # Set batch size to none to have a variable batch size

left_image = np.zeros((image_height, image_width, 3))
right_image = np.zeros((image_height, image_width, 3))

# Left and right image inputs
left_input = tf.keras.Input(shape=(image_height, image_width, 3, ), batch_size=batch_size, name="left_image_input", dtype=tf.float32)
right_input = tf.keras.Input(shape=(image_height, image_width, 3, ), batch_size=batch_size, name="right_image_input", dtype=tf.float32)

#######################PYRAMID FEATURES###############################
act = tf.keras.layers.Activation(tf.nn.leaky_relu)
# Left image feature pyramid
# F1
left_pyramid = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv1")(left_input)
left_F1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv2")(left_pyramid)
# F2
left_pyramid = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv3")(left_F1)
left_F2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv4")(left_pyramid)
# F3
left_pyramid = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv5")(left_F2)
left_F3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv6")(left_pyramid)
# F4
left_pyramid = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv7")(left_F3)
left_F4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv8")(left_pyramid)
# F5
left_pyramid = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv9")(left_F4)
left_F5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv10")(left_pyramid)
# F6
left_pyramid = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="left_conv11")(left_F5)
left_F6 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="left_conv12")(left_pyramid)


# Right image feature pyramid
# F1
right_pyramid = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv1")(right_input)
right_F1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv2")(right_pyramid)
# F2
right_pyramid = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv3")(right_F1)
right_F2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv4")(right_pyramid)
# F3
right_pyramid = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv5")(right_F2)
right_F3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv6")(right_pyramid)
# F4
right_pyramid = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv7")(right_F3)
right_F4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv8")(right_pyramid)
# F5
right_pyramid = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv9")(right_F4)
right_F5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv10")(right_pyramid)
# F6
right_pyramid = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, padding="same", activation=act, use_bias=True, name="right_conv11")(right_F5)
right_F6 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="right_conv12")(right_pyramid)


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

class StereoEstimator(tf.keras.layers.Layer):

    def __init__(self, name="volume_filtering"):
        super(StereoEstimator, self).__init__(name=name)

    def call(self,  costs, upsampled_disp=None):

        if upsampled_disp is not None:
            volume = tf.keras.layers.concatenate([costs, upsampled_disp], axis=-1)
        else:
            volume = costs

        disp = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp1")(volume)
        disp = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp2")(disp)
        disp = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp3")(disp)
        disp = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp4")(disp)
        disp = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation=act, use_bias=True, name="disp5")(disp)
        disp = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding="same", activation=None, use_bias=True, name="disp6")(disp)

        return disp


# # https://github.com/philferriere/tfoptflow/blob/bdc7a72e78008d1cd6db46e4667dffc2bab1fe9e/tfoptflow/core_warp.py
# def dense_image_warp(image, flow, name='dense_image_warp'):
#     """Image warping using per-pixel flow vectors.
#     Apply a non-linear warp to the image, where the warp is specified by a dense
#     flow field of offset vectors that define the correspondences of pixel values
#     in the output image back to locations in the  source image. Specifically, the
#     pixel value at output[b, j, i, c] is
#     images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
#     The locations specified by this formula do not necessarily map to an int
#     index. Therefore, the pixel value is obtained by bilinear
#     interpolation of the 4 nearest pixels around
#     (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
#     of the image, we use the nearest pixel values at the image boundary.
#     Args:
#       image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
#       flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
#       name: A name for the operation (optional).
#       Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
#       and do not necessarily have to be the same type.
#     Returns:
#       A 4-D float `Tensor` with shape`[batch, height, width, channels]`
#         and same type as input image.
#     Raises:
#       ValueError: if height < 2 or width < 2 or the inputs have the wrong number
#                   of dimensions.
#     """

#     batch_size, height, width, channels = array_ops.unstack(array_ops.shape(image))
#     # The flow is defined on the image grid. Turn the flow into a list of query
#     # points in the grid space.
#     grid_x, grid_y = array_ops.meshgrid(math_ops.range(width), math_ops.range(height))
#     stacked_grid = math_ops.cast(array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
#     batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
#     query_points_on_grid = batched_grid - flow
#     query_points_flattened = array_ops.reshape(query_points_on_grid, [batch_size, height * width, 2])
#     # Compute values at the query points, then reshape the result back to the
#     # image grid.
#     interpolated = _interpolate_bilinear(image, query_points_flattened)
#     interpolated = array_ops.reshape(interpolated, [batch_size, height, width, channels])
#     return interpolated

def _build_indeces(coords):
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








search_range = 2 # maximum dispacement

#############################SCALE 6#################################
# Get the disparity
disp_v6 = StereoCostVolume(name="cost_6")(left_F6, right_F6, search_range)
V6 = StereoEstimator(name="volume_filtering_6")(disp_v6)


############################SCALE 5###################################
# Upsample disparity from previous layer
height_5, width_5 = (left_F5.shape.as_list()[1], left_F5.shape.as_list()[2])
u5 = tf.keras.layers.Resizing(height=height_5, width=width_5, interpolation='bilinear')(V6)
# Get warped right image
coords5 = tf.keras.layers.concatenate([u5, tf.zeros_like(u5)], -1)
indices_5 = _build_indeces(coords5)
warp_5 = Warp(name="warp_5")(right_F5, indices_5)
# Get the disparity
disp_v5 = StereoCostVolume(name="cost_5")(left_F5, warp_5, search_range)
V5 = StereoEstimator(name="volume_filtering_5")(disp_v5, u5)


############################SCALE 4###################################
# Upsample disparity from previous layer
height_4, width_4 = (left_F4.shape.as_list()[1], left_F4.shape.as_list()[2])
u4 = tf.keras.layers.Resizing(height=height_4, width=width_4, interpolation='bilinear')(V5)
# Get warped right image
coords4 = tf.keras.layers.concatenate([u4, tf.zeros_like(u4)], -1)
indices_4 = _build_indeces(coords4)
warp_4 = Warp(name="warp_4")(right_F4, indices_4)
# Get the disparity
disp_v4 = StereoCostVolume(name="cost_4")(left_F4, warp_4, search_range)
V4 = StereoEstimator(name="volume_filtering_4")(disp_v4, u4)


############################SCALE 3###################################
# Upsample disparity from previous layer
height_3, width_3 = (left_F3.shape.as_list()[1], left_F3.shape.as_list()[2])
u3 = tf.keras.layers.Resizing(height=height_3, width=width_3, interpolation='bilinear')(V4)
# Get warped right image
coords3 = tf.keras.layers.concatenate([u3, tf.zeros_like(u3)], -1)
indices_3 = _build_indeces(coords3)
warp_3 = Warp(name="warp_3")(right_F3, indices_3)
# Get the disparity
disp_v3 = StereoCostVolume(name="cost_3")(left_F3, warp_3, search_range)
V3 = StereoEstimator(name="volume_filtering_3")(disp_v3, u3)


############################SCALE 2###################################
# Upsample disparity from previous layer
height_2, width_2 = (left_F2.shape.as_list()[1], left_F2.shape.as_list()[2])
u2 = tf.keras.layers.Resizing(height=height_2, width=width_2, interpolation='bilinear')(V3)
# Get warped right image
coords2 = tf.keras.layers.concatenate([u2, tf.zeros_like(u2)], -1)
indices_2 = _build_indeces(coords2)
warp_2 = Warp(name="warp_2")(right_F2, indices_2)
# Get the disparity
disp_v2 = StereoCostVolume(name="cost_2")(left_F2, warp_2, search_range)
V2 = StereoEstimator(name="volume_filtering_2")(disp_v2, u2)









MADNet = tf.keras.Model(inputs=[left_input, right_input], outputs=disparity, name="MADNet")


MADNet.summary()
tf.keras.utils.plot_model(MADNet, "./images/MADNet Model Structure.png", show_layer_names=True)













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

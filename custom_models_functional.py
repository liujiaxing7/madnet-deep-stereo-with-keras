from keras import backend
import tensorflow as tf
from keras.utils import layer_utils
from keras import layers
import numpy as np
from matplotlib import cm


def colorize_img(value, vmin=None, vmax=None, cmap='jet'):
    """
    A utility function for TensorFlow that maps a grayscale image to
    a matplotlib colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1
    before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size,height, width,1]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')
    
    Returns a 3D tensor of shape [batch_size,height, width,3].
    """
    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = tf.cast(tf.round(value[:, :, :, 0]*255), dtype=tf.int32)

    # gather
    color_map = cm.get_cmap(cmap)
    colors = color_map(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    return value


# https://github.com/philferriere/tfoptflow/blob/bdc7a72e78008d1cd6db46e4667dffc2bab1fe9e/tfoptflow/core_costvol.py
def _cost_volume_block(c1, warp, search_range=2):
    """Build cost volume for associating a pixel from the
    left image with its corresponding pixels in the right image.
    Args:
        c1: Level of the feature pyramid of the left image
        warp: Warped level of the feature pyramid of the right image
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [0, 0], [search_range, search_range], [0, 0]])
    width = c1.shape[2]
    max_offset = search_range * 2 + 1

    cost_vol = []
    for i in range(0, max_offset):
        slice = padded_lvl[:, :, i:width+i, :]
        cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
        cost_vol.append(cost)

    cost_vol = tf.concat(cost_vol, axis=3)
    cost_curve = tf.concat([c1, cost_vol], axis=3)

    return cost_curve


def bilinear_sampler(imgs, coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,width_t, 2].
                height_t/width_t correspond to the dimensions of the output image
                (don't need to be the same as height_s/width_s). The two channels
                correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = tf.shape(imgs)
    #inp_size = imgs.shape
    coord_size = tf.shape(coords)
    #coord_size = coords.shape
    out_size = [coord_size[0], coord_size[1], coord_size[2], inp_size[3]]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(inp_size[1] - 1, 'float32')
    x_max = tf.cast(inp_size[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    wt_x0 = x1 - coords_x
    wt_x1 = coords_x - x0
    wt_y0 = y1 - coords_y
    wt_y1 = coords_y - y0

    x0_safe = tf.clip_by_value(x0, zero[0], x_max)
    y0_safe = tf.clip_by_value(y0, zero[0], y_max)
    x1_safe = tf.clip_by_value(x1, zero[0], x_max)
    y1_safe = tf.clip_by_value(y1, zero[0], y_max)

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1, coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1]
    )

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = x0_safe + base_y0
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, [-1, inp_size[3]])
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])

    return output


def _warp_image_block(img, flow):
    """
    Given an image and a flow generate the warped image,
    for stereo img is the right image, flow is the disparity alligned with left.
    img: image that needs to be warped
    flow: Generic optical flow or disparity
    """

    def build_coords(immy):
        max_height = 2048
        max_width = 2048
        pixel_coords = np.ones((1, max_height, max_width, 2))

        # build pixel coordinates and their disparity
        for i in range(0, max_height):
            for j in range(0, max_width):
                pixel_coords[0][i][j][0] = j
                pixel_coords[0][i][j][1] = i

        pixel_coords = tf.constant(pixel_coords, tf.float32)
        real_height = tf.shape(immy)[1]
        real_width = tf.shape(immy)[2]
        real_pixel_coord = pixel_coords[:, 0:real_height, 0:real_width, :]
        immy = tf.concat([immy, tf.zeros_like(immy)], axis=-1)
        output = real_pixel_coord - immy

        return output

    coords = build_coords(flow)
    warped = bilinear_sampler(img, coords)
    return warped


def _refinement_block(input, disp, output_shape):
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
    layer_kwargs = {
        "kernel_size": (3, 3),
        "padding": "same",
        "activation": tf.keras.layers.Activation(tf.nn.leaky_relu),
        "use_bias": True
    }
    context1 = tf.keras.layers.Conv2D(filters=128, dilation_rate=1, name="context1", **layer_kwargs)
    context2 = tf.keras.layers.Conv2D(filters=128, dilation_rate=2, name="context2", **layer_kwargs)
    context3 = tf.keras.layers.Conv2D(filters=128, dilation_rate=4, name="context3", **layer_kwargs)
    context4 = tf.keras.layers.Conv2D(filters=96, dilation_rate=8, name="context4", **layer_kwargs)
    context5 = tf.keras.layers.Conv2D(filters=64, dilation_rate=16, name="context5", **layer_kwargs)
    context6 = tf.keras.layers.Conv2D(filters=32, dilation_rate=1, name="context6", **layer_kwargs)
    context7 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        dilation_rate=1,
        padding="same",
        activation="linear",
        use_bias=True,
        name="context7"
    )

    volume = tf.keras.layers.concatenate([input, disp], axis=-1)
    x = context1(volume)
    x = context2(x)
    x = context3(x)
    x = context4(x)
    x = context5(x)
    x = context6(x)
    x = context7(x)

    context_disp = tf.keras.layers.add([disp, x])
    final_disparity = tf.image.resize(
        images=context_disp,
        name="final_disparity",
        size=(output_shape[0], output_shape[1]),
        method='bilinear'
    )
    return final_disparity


def _stereo_estimator_block(name, costs, upsampled_disp=None):
    """
    This is the stereo estimation network at resolution n.
    It uses the costs (from the pixel difference between the warped right image 
    and the left image) combined with the upsampled disparity from the previous
    layer (when the layer is not the last layer).

    The output is predicted disparity for the network at resolution n.
    """
    layer_kwargs = {
        "kernel_size": (3, 3),
        "strides": 1,
        "padding": "same",
        "activation": tf.keras.layers.Activation(tf.nn.leaky_relu),
        "use_bias": True
    }
    disp1 = tf.keras.layers.Conv2D(filters=128, name=f"{name}_disp1", **layer_kwargs)
    disp2 = tf.keras.layers.Conv2D(filters=128, name=f"{name}_disp2", **layer_kwargs)
    disp3 = tf.keras.layers.Conv2D(filters=96, name=f"{name}_disp3", **layer_kwargs)
    disp4 = tf.keras.layers.Conv2D(filters=64, name=f"{name}_disp4", **layer_kwargs)
    disp5 = tf.keras.layers.Conv2D(filters=32, name=f"{name}_disp5", **layer_kwargs)
    disp6 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3,3),
        strides=1,
        padding="same",
        activation="linear",
        use_bias=True,
        name=f"{name}_disp6"
    )

    if upsampled_disp is not None:
        volume = tf.keras.layers.concatenate([costs, upsampled_disp], axis=-1)
    else:
        volume = costs

    x = disp1(volume)
    x = disp2(x)
    x = disp3(x)
    x = disp4(x)
    x = disp5(x)
    x = disp6(x)
    return x


def ModuleM(layer, search_range=2):
    """
    Module MX is a sub-module of MADNet, which can be trained individually for 
    online adaptation using the MAD (Modular ADaptaion) method.
    """

    def _block(inputs):
        # Check if layer is the bottom of the pyramid
        if len(inputs) == 3:
            left, right, prev_disp = inputs
            mod_height, mod_width = left.shape[1], left.shape[2]
            # Upsample disparity from previous layer
            upsampled_disp = tf.image.resize(
                images=prev_disp,
                name=f"upsampled_disp_{layer}",
                size=(mod_height, mod_width),
                method='bilinear'
            )
            # Warp the right image into the left using upsampled disparity
            warped_left = _warp_image_block(right, upsampled_disp)
        else:
            left, right = inputs
            # No previous disparity exits, so use right image instead of warped left
            warped_left = right

        costs = _cost_volume_block(left, warped_left, search_range)

        # Get the disparity using cost volume between left and warped left images
        if len(inputs) == 3:
            module_disparity = _stereo_estimator_block(f"volume_filtering_{layer}", costs, upsampled_disp)
        else:
            module_disparity = _stereo_estimator_block(f"volume_filtering_{layer}", costs)

        return module_disparity

    return _block


def MADNet(input_shape=None,
           weights=None,
           input_tensor=None,
           search_range=2
           ):
    """
    Instantiates the MADNet architecture

    Reference:
        - [MADNet: Real-time self-adaptive deep stereo](
          https://arxiv.org/abs/1810.05424) (CVPR 2019)

    Args:
        input_shape: Optional shape tuple, to be specified if you would
            like to use a model with an input image resolution that is not
            (480, 640, 3).
            It should have exactly 3 inputs channels (480, 640, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
        weights: String, one of `None` (random initialization),
            or the path to the weights file to be loaded.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        search_range: maximum search displacement for the cost volume

    Returns:
        A `keras.Model` instance.
    """
    if not (weights in {None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.  '
                         f'Received `weights={weights}`')
    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    layer_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError(
                    f'input_tensor: {input_tensor}'
                    'is not type input_tensor. '
                    f'Received `type(input_tensor)={type(input_tensor)}`'
                )
        if is_input_t_tensor:
            if backend.image_data_format() == 'channels_first':
                raise ValueError('Detected input_tensor in channels_first mode '
                                 'please ensure channels are last`; '
                                 'Received `input_tensor.shape='
                                 f'{input_tensor.shape}')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError(
                        'input_tensor.shape[2] must equal input_shape[1]; '
                        'Received `input_tensor.shape='
                        f'{input_tensor.shape}`, '
                        f'`input_shape={input_shape}`')
        else:
            raise ValueError('input_tensor is not a Keras tensor; '
                             f'Received `input_tensor={input_tensor}`')


    default_shape = (480, 640, 3)
    # If input_shape is None, infer shape from input_tensor.
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor must be a valid Keras tensor type; '
                             f'Received {input_tensor} of type {type(input_tensor)}')

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            input_shape = default_shape
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                raise ValueError('Detected input_tensor in channels_first mode '
                                 'please ensure channels are last`; '
                                 'Received `input_tensor.shape='
                                 f'{input_tensor.shape}')
            else:
                input_shape = (backend.int_shape(input_tensor)[1],
                               backend.int_shape(input_tensor)[2],
                               backend.int_shape(input_tensor)[3])

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        input_shape = default_shape

    # If input_shape is not None, assume default size.
    else:
        if backend.image_data_format() == 'channels_first':
            raise ValueError('Detected input_tensor in channels_first mode '
                             'please ensure channels are last`; '
                             'Received `input_tensor.shape='
                             f'{input_tensor.shape}')
    # left and right image inputs are set to the same resolution
    left_input = layers.Input(shape=input_shape, name="left_input")
    right_input = layers.Input(shape=input_shape, name="right_input")

    # Initializing the layers
    layer_kwargs = {
        "kernel_size": (3, 3),
        "padding": "same",
        "activation": tf.keras.layers.Activation(tf.nn.leaky_relu),
        "use_bias": True
    }
    # Left image feature pyramid (feature extractor)
    # F1
    left_conv1 = tf.keras.layers.Conv2D(
        filters=16,
        strides=2,
        name="left_conv1",
        input_shape=(input_shape[0], input_shape[1], input_shape[2], ),
        **layer_kwargs)
    left_conv2 = tf.keras.layers.Conv2D(filters=16, strides=1, name="left_conv2", **layer_kwargs)
    # F2
    left_conv3 = tf.keras.layers.Conv2D(filters=32, strides=2, name="left_conv3", **layer_kwargs)
    left_conv4 = tf.keras.layers.Conv2D(filters=32, strides=1, name="left_conv4", **layer_kwargs)
    # F3
    left_conv5 = tf.keras.layers.Conv2D(filters=64, strides=2, name="left_conv5", **layer_kwargs)
    left_conv6 = tf.keras.layers.Conv2D(filters=64, strides=1, name="left_conv6", **layer_kwargs)
    # F4
    left_conv7 = tf.keras.layers.Conv2D(filters=96, strides=2, name="left_conv7", **layer_kwargs)
    left_conv8 = tf.keras.layers.Conv2D(filters=96, strides=1, name="left_conv8", **layer_kwargs)
    # F5
    left_conv9 = tf.keras.layers.Conv2D(filters=128, strides=2, name="left_conv9", **layer_kwargs)
    left_conv10 = tf.keras.layers.Conv2D(filters=128, strides=1, name="left_conv10", **layer_kwargs)
    # F6
    left_conv11 = tf.keras.layers.Conv2D(filters=192, strides=2, name="left_conv11", **layer_kwargs)
    left_conv12 = tf.keras.layers.Conv2D(filters=192, strides=1, name="left_conv12", **layer_kwargs)
    # Right image feature pyramid (feature extractor)
    # F1
    right_conv1 = tf.keras.layers.Conv2D(
        filters=16,
        strides=2,
        name="right_conv1",
        input_shape=(input_shape[0], input_shape[1], input_shape[2], ),
        **layer_kwargs)
    right_conv2 = tf.keras.layers.Conv2D(filters=16, strides=1, name="right_conv2", **layer_kwargs)
    # F2
    right_conv3 = tf.keras.layers.Conv2D(filters=32, strides=2, name="right_conv3", **layer_kwargs)
    right_conv4 = tf.keras.layers.Conv2D(filters=32, strides=1, name="right_conv4", **layer_kwargs)
    # F3
    right_conv5 = tf.keras.layers.Conv2D(filters=64, strides=2, name="right_conv5", **layer_kwargs)
    right_conv6 = tf.keras.layers.Conv2D(filters=64, strides=1, name="right_conv6", **layer_kwargs)
    # F4
    right_conv7 = tf.keras.layers.Conv2D(filters=96, strides=2, name="right_conv7", **layer_kwargs)
    right_conv8 = tf.keras.layers.Conv2D(filters=96, strides=1, name="right_conv8", **layer_kwargs)
    # F5
    right_conv9 = tf.keras.layers.Conv2D(filters=128, strides=2, name="right_conv9", **layer_kwargs)
    right_conv10 = tf.keras.layers.Conv2D(filters=128, strides=1, name="right_conv10", **layer_kwargs)
    # F6
    right_conv11 = tf.keras.layers.Conv2D(filters=192, strides=2, name="right_conv11", **layer_kwargs)
    right_conv12 = tf.keras.layers.Conv2D(filters=192, strides=1, name="right_conv12", **layer_kwargs)

    #############################SCALE 6#################################
    M6 = ModuleM(layer="6", search_range=search_range)
    ############################SCALE 5###################################
    M5 = ModuleM(layer="5", search_range=search_range)
    ############################SCALE 4###################################
    M4 = ModuleM(layer="4", search_range=search_range)
    ############################SCALE 3###################################
    M3 = ModuleM(layer="3", search_range=search_range)
    ############################SCALE 2###################################
    M2 = ModuleM(layer="2", search_range=search_range)

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
    final_disparity = _refinement_block(left_F2, D2, input_shape)

    model = tf.keras.Model(inputs={
                                "left_input": left_input,
                                "right_input": right_input
                            },
                           outputs=final_disparity,
                           name="MADNet")
    if weights is not None:
        model.load_weights(weights)

    return model

height = 490
width = 640
search_range = 2
batch_size = 1
left_tensor = tf.random.normal(shape=(batch_size, height, width, 3))
right_tensor = tf.random.normal(shape=(batch_size, height, width, 3))
tensor = layers.Input(shape=(height, width, 3))

model = MADNet(
    input_shape=(height, width, 3),
    input_tensor=tensor
)

model.summary()

disp_pred = model({
    "left_input": left_tensor,
    "right_input": right_tensor
})

print(disp_pred.shape)


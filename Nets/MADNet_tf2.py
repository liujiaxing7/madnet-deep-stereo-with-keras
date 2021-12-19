import tensorflow as tf
import numpy as np
import math
#from Losses.loss_factory import mean_SSIM_L1

print("\nTensorFlow Version: {}".format(tf.__version__))


# dummpy data for the images
image_height = 320
image_width = 1216
input_size = (image_height, image_width)
batch_size = 1 # Set batch size to none to have a variable batch size

search_range = 2 # maximum dispacement (ie. smallest disparity)


# height resolution is too small on M6 for this method
# def mean_SSIM_L1(reprojected, original):
#     sum_l1 = tf.reduce_sum(tf.abs(reprojected - original))
#     mean_SSIM = tf.reduce_mean(tf.image.ssim(reprojected, original, 1.0))
#     return 0.85 * mean_SSIM + 0.15 * sum_l1

# doesnt support symbolic tensors
# def mean_SSIM_L1(x, y):
#     """
#     SSIM dissimilarity measure
#     Args:
#         x: predicted image
#         y: target image
#     """
#     C1 = 0.01**2
#     C2 = 0.03**2
#     mu_x = tf.nn.avg_pool(x,[1,3,3,1],[1,1,1,1],padding='VALID')
#     mu_y = tf.nn.avg_pool(y,[1,3,3,1],[1,1,1,1],padding='VALID')

#     sigma_x = tf.nn.avg_pool(x**2, [1,3,3,1],[1,1,1,1],padding='VALID') - mu_x**2
#     sigma_y = tf.nn.avg_pool(y**2, [1,3,3,1],[1,1,1,1],padding='VALID') - mu_y**2
#     sigma_xy = tf.nn.avg_pool(x*y, [1,3,3,1],[1,1,1,1],padding='VALID') - mu_x * mu_y

#     SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
#     SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

#     SSIM = SSIM_n / SSIM_d
#     SSIM = tf.clip_by_value((1-SSIM)/2, 0 ,1)

#     mean_SSIM = tf.reduce_mean(SSIM)

#     sum_l1 = tf.reduce_sum(tf.abs(x - y))

#     return 0.85 * mean_SSIM + 0.15 * sum_l1

# def mean_SSIM_L1(x, y):
#     """
#     SSIM dissimilarity measure
#     Args:
#         x: predicted image
#         y: target image
#     """
#     print("\nInside loss")
#     # define layers
#     pool = tf.keras.layers.AveragePooling2D(pool_size=(3,3) ,strides=(1,1), padding='valid')
#     multiply = tf.keras.layers.Multiply()
#     add = tf.keras.layers.Add()
#     subtract = tf.keras.layers.Subtract()
#     divide = tf.keras.layers.Lambda(lambda x: tf.divide(x[0], x[1]))
#     clip_value = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0 ,1))
#     reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x))
#     abs = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))
#     reduce_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x))

#     C1 = tf.constant(0.01**2, dtype=tf.float32)
#     C2 = tf.constant(0.03**2, dtype=tf.float32)
#     print(f"x: {x}")
#     print(f"y: {y}")

#     # pooling
#     mu_x = pool(x)
#     mu_y = pool(y)
#     # multiplies
#     x_square = multiply([x, x])
#     y_square = multiply([y, y])
#     mu_x_square = multiply([mu_x, mu_x])
#     mu_y_square = multiply([mu_y, mu_y])
#     xy = multiply([x, y])
#     mu_x_mu_y = multiply([mu_x, mu_y])

#     sigma_x = subtract([pool(x_square), mu_x_square])
#     sigma_y = subtract([pool(y_square), mu_y_square])
#     sigma_xy = subtract([pool(xy), mu_x_mu_y])


#     n1 = multiply([tf.constant(2, dtype=tf.float32), mu_x, mu_y])
#     n2 = multiply([tf.constant(2, dtype=tf.float32), sigma_xy])

#     print(f"n1: {n1}")
#     print(f"n2: {n2}")

#     SSIM_n = multiply([add([n1, C1]), add([n2, C2])])
#     SSIM_d = multiply([add([mu_x_square, mu_y_square, C1]), add([sigma_x, sigma_y, C2])])

#     SSIM = divide([SSIM_n, SSIM_d])
#     SSIM = divide([subtract([tf.constant(1, dtype=tf.float32), SSIM]), tf.constant(2, dtype=tf.float32)])
#     SSIM = clip_value([SSIM])

#     mean_SSIM = reduce_mean(SSIM)
#     sum_l1 = reduce_sum(abs([x, y]))
#     return add([multiply([tf.constant(0.85, dtype=tf.float32), mean_SSIM]), multiply([tf.constant(0.15, dtype=tf.float32), sum_l1])])


class SSIMLoss(tf.keras.losses.Loss):
    """
    SSIM dissimilarity measure
    Args:
        x: target image
        y: predicted image
    """
    def __init__(self, name="mean_SSIM_l1"):
        super(SSIMLoss, self).__init__(name=name)

    def call(self, x, y):
        print("\nInside loss")
        # define layers
        pool = tf.keras.layers.AveragePooling2D(pool_size=(3,3) ,strides=(1,1), padding='valid')
        multiply = tf.keras.layers.Multiply()
        subtract = tf.keras.layers.Subtract()
        divide = tf.keras.layers.Lambda(lambda x: tf.divide(x[0], x[1]))
        clip_value = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0 ,1))
        reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x))
        abs = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))
        reduce_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x))

        C1 = tf.constant(0.01**2, dtype=tf.float32)
        C2 = tf.constant(0.03**2, dtype=tf.float32)

        # pooling
        mu_x = pool(x)
        mu_y = pool(y)
        # multiplies
        x_square = multiply([x, x])
        y_square = multiply([y, y])
        mu_x_square = multiply([mu_x, mu_x])
        mu_y_square = multiply([mu_y, mu_y])
        xy = multiply([x, y])
        mu_x_mu_y = multiply([mu_x, mu_y])

        sigma_x = subtract([pool(x_square), mu_x_square])
        sigma_y = subtract([pool(y_square), mu_y_square])
        sigma_xy = subtract([pool(xy), mu_x_mu_y])


        n1 = multiply([tf.constant(2, dtype=tf.float32), mu_x, mu_y])
        n2 = multiply([tf.constant(2, dtype=tf.float32), sigma_xy])

        SSIM_n = multiply([n1 + C1, n2 + C2])
        SSIM_d = multiply([mu_x_square + mu_y_square, C1, sigma_x + sigma_y, C2])

        SSIM = divide([SSIM_n, SSIM_d])
        SSIM = divide([subtract([tf.constant(1, dtype=tf.float32), SSIM]), tf.constant(2, dtype=tf.float32)])
        SSIM = clip_value([SSIM])

        mean_SSIM = reduce_mean(SSIM)
        sum_l1 = reduce_sum(abs([x, y]))
        loss = multiply([tf.constant(0.85, dtype=tf.float32), mean_SSIM]) + multiply([tf.constant(0.15, dtype=tf.float32), sum_l1])

        #loss = C1 + C2

        return loss




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

    def call(self, input, disp, final_left, final_right):
        print(f"\nStarted call context network")
        #volume = tf.keras.layers.concatenate([input, disp], axis=-1)
        volume = tf.keras.layers.Concatenate(axis=-1)([input, disp])

        print(f"input: {input.shape}")
        print(f"disp: {disp.shape}")
        print(f"volume: {volume.shape}")

        # Need to check if context was created previously,
        # so variable doesnt get created multiple times (for autograph)
        #if self.x is None:
        self.x = self.context1(volume)
        self.x = self.context2(self.x)
        self.x = self.context3(self.x)
        self.x = self.context4(self.x)
        self.x = self.context5(self.x)
        self.x = self.context6(self.x)
        self.x = self.context7(self.x)

        #context_disp = tf.keras.layers.add([disp, self.x], name="final_disp")
        context_disp = tf.keras.layers.Add(name="context_disp")([disp, self.x])


        final_disparity = tf.keras.layers.Resizing(name="final_disparity", height=final_left.shape[1], width=final_left.shape[2], interpolation='bilinear')(context_disp)

        # warp right image with final disparity to get final reprojection loss
        #final_coords = tf.keras.layers.concatenate([final_disparity, tf.zeros_like(final_disparity)], -1)
        final_coords = tf.keras.layers.Concatenate(axis=-1)([final_disparity, tf.zeros_like(final_disparity)])
        final_indices = BuildIndices(name="build_indices_final", batch_size=self.batch_size)(final_coords)
        # Warp the right image into the left using final disparity
        final_warped_left = Warp(name="warp_final")(final_right, final_indices)    

        print(f"final left image: {final_left.shape}")

        final_reprojection_loss = self.loss_fn(final_warped_left, final_left)
        self.add_loss(final_reprojection_loss) 

        print(f"final_warped_left: {final_warped_left.shape}")
        print(f"Final Reprojection Loss: {final_reprojection_loss}")       

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
        self.x = None
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
        # Need to check if disp was created previously,
        # so variable doesnt get created multiple times (for autograph)
        #if self.x is None:
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
        self.module_disparity = None
        self.final_disparity = None
        self.context_disparity = None
        self.search_range = search_range
        self.batch_size = batch_size
        self.final_reprojection_loss = None
        self.layer = layer
        self.loss_fn = SSIMLoss()
        self.cost_volume = StereoCostVolume(name=f"cost_{self.layer}")
        self.stereo_estimator = StereoEstimator(name=f"volume_filtering_{self.layer}")

    def call(self, left, right, prev_disp=None, final_left=None, final_right=None):
        print(f"\nStarted call ModuleM {self.layer}")

        print(f"left: {left.shape}")
        print(f"right: {right.shape}")

        height, width = (left.shape.as_list()[1], left.shape.as_list()[2])
        # Check if module disparity was previously calculated to prevent retracing (for autograph)
        #if self.module_disparity is None:
        # Check if layer is the bottom of the pyramid
        if prev_disp is not None:
            # Upsample disparity from previous layer
            upsampled_disp = tf.keras.layers.Resizing(name=f"upsampled_disp_{self.layer}", height=height, width=width, interpolation='bilinear')(prev_disp)
            coords = tf.keras.layers.concatenate([upsampled_disp, tf.zeros_like(upsampled_disp)], -1)
            indices = BuildIndices(name=f"build_indices_{self.layer}", batch_size=self.batch_size)(coords)
            # Warp the right image into the left using upsampled disparity
            warped_left = Warp(name=f"warp_{self.layer}")(right, indices)
        else:
            print("Inside prev disp is None")
            # No previous disparity exits, so use right image instead of warped left
            warped_left = right

        # add loss estimating the reprojection accuracy of the pyramid level (for self supervised training/MAD)
        reprojection_loss = self.loss_fn(warped_left, left)
        self.add_loss(reprojection_loss)

        print(f"warped_left: {warped_left.shape}")
        print(f"Reprojection Loss: {reprojection_loss}")

        costs = self.cost_volume(left, warped_left, self.search_range)
        # Get the disparity using cost volume between left and warped left images
        self.module_disparity = self.stereo_estimator(costs)

        # Add the residual refinement network to the final layer
        # also check if disparity was created previously (for autograph)
        if final_left is not None and self.final_disparity is None:
            self.final_disparity, self.final_reprojection_loss = StereoContextNetwork(batch_size=self.batch_size)(left, self.module_disparity, final_left, final_right) 
            self.add_loss(self.final_reprojection_loss)

        print(f"self.module_disparity: {self.module_disparity.shape}")

        disp = self.final_disparity if self.final_disparity is not None else self.module_disparity
        final_loss = None if self.final_reprojection_loss is None else self.final_reprojection_loss

        return disp, reprojection_loss, final_loss




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




    # Forward pass of the model
    def call(self, inputs):
        print("\nStarted Call MADNet")
        # Left and right image inputs
        left_input, right_input = inputs
        print(f"Left input: {left_input.shape}")
        print(f"Right input: {right_input.shape}")

        #######################PYRAMID FEATURES###############################
        # Left image feature pyramid (feature extractor)
        #if self.left_pyramid is None:
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
        #if self.right_pyramid is None:
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
        D6, losses["D6"], _ = self.M6(left_F6, right_F6)
        ############################SCALE 5###################################
        D5, losses["D5"], _ = self.M5(left_F5, right_F5, D6)       
        ############################SCALE 4###################################
        D4, losses["D4"], _ = self.M4(left_F4, right_F4, D5) 
        ############################SCALE 3###################################
        D3, losses["D3"], _ = self.M3(left_F3, right_F3, D4)
        ############################SCALE 2###################################
        D2, losses["D2"], losses["final"] = self.M2(left_F2, right_F2, D3, left_input, right_input)     

    
        print(f"Losses: \n{losses}")
        self.add_loss(losses)
        return D2

model = MADNet()

model.compile(
    optimizer='adam'   
)

#model.summary()
#tf.keras.utils.plot_model(MADNet, "./images/MADNet Model Structure.png", show_layer_names=True)

# ---------------------------------------------------------------------------
# Train the model

left_input = np.random.random((1, image_height, image_width, 3))
right_input = np.random.random((1, image_height, image_width, 3))

history = model.fit(
    x=[left_input, right_input],
    epochs=3,
    verbose=2,
    #steps_per_epoch=steps_per_epoch
)



# # --------------------------------------------------------------------------------
# # Data Preperation

# left_dir = "G:/My Drive/Data Files/2011_09_26_drive_0002_sync/left"
# right_dir = "G:/My Drive/Data Files/2011_09_26_drive_0002_sync/right"

# # Create datagenerator object for loading and preparing image data for training
# left_dataflow_kwargs = dict(
#     directory = left_dir, 
#     target_size = input_size, 
#     class_mode = None,
#     batch_size = batch_size,
#     shuffle = False,     
#     interpolation = "bilinear",
#     )

# right_dataflow_kwargs = dict(
#     directory = right_dir, 
#     target_size = input_size, 
#     class_mode = None,
#     batch_size = batch_size,
#     shuffle = False,     
#     interpolation = "bilinear",
#     )


# # Normalize pixel values
# datagen_args = dict(
#     rescale = 1./255
#         )

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args)

# left_generator = datagen.flow_from_directory(**left_dataflow_kwargs)
# right_generator = datagen.flow_from_directory(**right_dataflow_kwargs)

# def generator(left_generator, right_generator):
#     """Combines the left and right image generators into a 
#         single image generator with two inputs for training.
        
#         Make sure the left and right images have the same ID,
#         otherwise the order might change which will pair the wrong
#         left and right images."""
#     while True:
#         for left, right in zip(left_generator, right_generator):
#             yield {"left_input": left, "right_input": right}, None

# steps_per_epoch = math.ceil(left_generator.samples / batch_size)


# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # debugging

# class Model(tf.keras.Model):
#     """
#     The test model
#     """
#     def __init__(self, name="DebugModel"):
#         super(Model, self).__init__(name=name)
#         print("\nStarted Initialization")
#         self.layer = tf.keras.layers.Dense(10)

#     # Forward pass of the model
#     def call(self, inputs):
#         print("\nStarted Call")
#         left_input = inputs
#         print(f"Left input: {left_input.shape}")
#         final_res = left_input.shape[1:3]

#         if final_res[0] is None or final_res[1] is None:
#             raise RuntimeError(f"Input resolution is None, please check your inputs: height {final_res[0]} width {final_res[1]}")

#         output = self.layer(left_input)
#         return output


# model = Model()

# model.compile(
#     optimizer='adam'   
# )


# history = model.fit(
#     x=left_generator,
#     epochs=2,
#     verbose=2,
#     steps_per_epoch=1
# )

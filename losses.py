import tensorflow as tf


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

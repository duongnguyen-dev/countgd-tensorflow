import math
import tensorflow as tf

class UpScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        """
        This is an upscale layer like in the paper, the input x should be output of the last 3 stages from SwinT block. This layer
        will perform the following operation:
        - Upsample: upsampling the output of stage 2 and 3 from (H/32 x W/32 x 1024) and (H/16 x W/16 x 512) to (H/8 x H/8 x 1024) and (H/8 x W/8 x 512) respectively.
        - Concatenate: concat three layers into (H/8 x W/8 x (C1 + C2 + C3)).
        - Project: project the output channel from (C1 + C2 + C3) into 256.
        """

        out_stage_2, out_stage_3, out_stage_4 = x

        _, L4, C4 = out_stage_4.shape
        _, L3, C3 = out_stage_3.shape
        _, L2, C2 = out_stage_2.shape
        out_stage_4 = tf.reshape(out_stage_4, [-1, int(math.sqrt(L4)), int(math.sqrt(L4)), C4])
        out_stage_4 = tf.keras.layers.UpSampling2D(
            size = (4, 4),
            interpolation="bilinear"
        )(out_stage_4)

        out_stage_3 = tf.reshape(out_stage_3, [-1, int(math.sqrt(L3)), int(math.sqrt(L3)), C3])
        out_stage_3 = tf.keras.layers.UpSampling2D(
            size = (2, 2),
            interpolation="bilinear"
        )(out_stage_3)

        out_stage_2 = tf.reshape(out_stage_2, [-1, int(math.sqrt(L2)), int(math.sqrt(L2)), C2])
        concat = tf.keras.layers.Concatenate(axis=-1)([out_stage_2, out_stage_3, out_stage_4])
        proj = tf.keras.layers.Dense(256)(concat)

        return proj
import tensorflow as tf

class PatchMerging(tf.keras.layers.Layer):
    '''
    This layer merges smaller patches into the larger one and doubles its channel.
    '''
    def __init__(self, input_resolution, channels):
        super().__init__()
        self.input_resolution = input_resolution
        self.channels = channels 
        self.linear_trans = tf.keras.layers.Dense(2 * channels, use_bias=False)

    def call(self, x):
        height, width = self.input_resolution
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C)) # B, H // P, W // P, P*P*C

        x0 = x[:, 0::2, 0::2, :] # Top-left window
        x1 = x[:, 1::2, 0::2, :] # Bottom-left window
        x2 = x[:, 0::2, 1::2, :] # Top-right window
        x3 = x[:, 1::2, 1::2, :] # Bottom-right window

        # This line will merge 4 windows along the last axis, this will create a hierarchical architecture.
        x = tf.concat((x0, x1, x2, x3), axis=-1) # B, H // 2, W // 2, 4*C 
        x = tf.reshape(x, shape=(-1, (height//2) * (width//2), 4*C)) # B, (H*W)//4, 4*C
        x = self.linear_trans(x) # B, (H*W)//4, 2*C
        return x
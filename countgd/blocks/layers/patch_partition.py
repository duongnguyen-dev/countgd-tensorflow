import tensorflow as tf

class PatchParition(tf.keras.layers.Layer):
    '''
    This layer is the same as the patch parition layer from ViT model, according to part 3.1 in this paper: https://arxiv.org/pdf/2010.11929
    It takes the input of size (B, H, W, C) and turn into ( B, N, (P*P*C) ), where P is the patch size or window size and N = (H*W)//(P*P) is the number of patches in an image
    '''
    def __init__(self, window_size=4):
        super().__init__()
        self.window_size = window_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.window_size, self.window_size, 1],
            strides = [1, self.window_size, self.window_size, 1], # This parameter is used to control the overlapped between two consecutives patches
            rates = [1, 1, 1, 1], # This parameter determines which pixels are included in each patch
            padding="VALID" # VALID if you need to reduce the dimensions of the input, otherwise use SAME
        ) # (B, H, W, C) => (B, H // P, W // P, P * P * C)
        patch_dims = patches.shape[-1] # P * P * C
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) # ( B, N, (P*P*C) )
        return patches
    
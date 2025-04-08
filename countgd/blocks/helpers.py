import tensorflow as tf

def window_partition(x, window_size=7):
    '''
    This function helps to create windows from the input batch.
    It will return a list tensor of shape (B, N, W, W, C) where:
    - N: is the number of windows = Ny * Nx
    - P: window size
    '''
    _, H, W, C = x.shape # B, H, W, C
    num_patch_y = H // window_size 
    num_patch_x = W // window_size
    x = tf.reshape(x, [-1, num_patch_y, window_size, num_patch_x, window_size, C]) # B, Ny, W, Nx, W, C
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]) # B, Ny, Nx, W, W, C
    windows = tf.reshape(x, [-1, window_size, window_size, C]) # B * Ny * Nx, W, W, C
    return windows

def window_reverse(windows, window_size, H, W):
    '''
    This function helps to reverse the partition step and return the input image
    '''
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size)) if windows.shape[0] != None else 1
    x = tf.reshape(windows, [B, H//window_size, W//window_size, window_size, window_size, C])
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [-1, H, W, C])
    return x

def log2_graph(x):
    return tf.math.log(x) / tf.math.log(2.0)
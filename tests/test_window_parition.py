import tensorflow as tf
from countgd.blocks.helpers import window_partition

def window_part(batch, window_size):
    patch_part = window_partition(batch, window_size)
    return tf.shape(patch_part)

def test_window_part():
    arr = tf.zeros(((1, 56, 56, 128)))
    assert window_part(arr, 4).numpy().tolist() == [1, 196, 4, 4, 128]

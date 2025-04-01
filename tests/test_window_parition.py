import tensorflow as tf
from countgd.blocks.helpers import window_partition

def window_part(batch, window_size):
    patch_part = window_partition(batch, window_size)
    return tf.shape(patch_part)

def test_window_part():
    arr = tf.zeros((16, 224, 224, 3))
    assert window_part(arr, 4).numpy().tolist() == [16, 3136, 4, 4, 3]

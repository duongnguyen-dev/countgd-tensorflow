import tensorflow as tf
from countgd.blocks.helpers import window_partition
from countgd.config import SwintBConfig

def window_part(batch, window_size):
    patch_part = window_partition(batch, window_size)
    return tf.shape(patch_part)

def test_window_part():
    arr = tf.zeros(((1, 56, 56, 128)))
    assert window_part(arr, SwintBConfig.WINDOW_SIZE).numpy().tolist() == [64, SwintBConfig.WINDOW_SIZE, SwintBConfig.WINDOW_SIZE, 128]

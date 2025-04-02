import tensorflow as tf
from countgd.blocks.helpers import window_reverse

def window_rev(windows, window_size, H, W):
    batch = window_reverse(windows, window_size, H, W)
    return tf.shape(batch)

def test_window_rev():
    windows = tf.zeros((1, 1024, 7, 7, 3))
    assert window_rev(windows, 7, 224, 224).numpy().tolist() == [1, 224, 224, 3]
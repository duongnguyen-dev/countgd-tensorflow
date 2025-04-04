import tensorflow as tf
from countgd.blocks.helpers import window_reverse

def window_rev(windows, window_size, H, W):
    batch = window_reverse(windows, window_size, H, W)
    return tf.shape(batch)

def test_window_rev():
    windows = tf.zeros((64, 7, 7, 128))
    assert window_rev(windows, 7, 56, 56).numpy().tolist() == [1, 56, 56, 128]
import tensorflow as tf
from countgd.blocks.helpers import window_reverse

def window_rev(windows, window_size, H, W):
    batch = window_reverse(windows, window_size, H, W)
    return tf.shape(batch)

def test_window_part():
    windows = tf.zeros((16, 3136, 4, 4, 3))
    assert window_rev(windows, 4, 224, 224).numpy().tolist() == [16, 224, 224, 3]
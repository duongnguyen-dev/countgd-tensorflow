import tensorflow as tf
from countgd.blocks.upscale import UpScaleLayer

def up_scale(arr):
    upscaled = UpScaleLayer()(arr)
    
    return tf.shape(upscaled)

def test_upscale():
    arr = (tf.zeros((1, 784, 256)), tf.zeros((1, 196, 512)), tf.zeros((1, 49, 1024)))
    assert up_scale(arr).numpy().tolist() == ([1, 28, 28, 256])
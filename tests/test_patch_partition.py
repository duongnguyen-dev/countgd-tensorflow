import tensorflow as tf
from countgd.blocks.layers.patch_partition import PatchParition

def patch_partition(arr):
    patch_part = PatchParition(patch_size=4)
    return tf.shape(patch_part(arr))

def test_patch_partition():
    arr = tf.zeros((1, 224, 224, 3))
    assert patch_partition(arr).numpy().tolist() == [1, 3136, 48]
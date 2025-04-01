import tensorflow as tf
from countgd.blocks.layers.patch_merging import PatchMerging
from countgd.blocks.layers.patch_partition import PatchParition

def patch_merging(patches, input_resolution, channels):
    merge = PatchMerging(input_resolution, channels)
    return tf.shape(merge(patches))

def test_patch_merging():
    part = PatchParition(4)
    arr = tf.zeros((1, 224, 224, 3))
    patches = part(arr)
    channels = 96
    num_patch_x = 224 // 4
    num_patch_y = 224 // 4
    assert patch_merging(patches, (num_patch_x, num_patch_y), channels).numpy().tolist() == [1, 784, 192]
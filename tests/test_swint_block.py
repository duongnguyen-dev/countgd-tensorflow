import tensorflow as tf
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.blocks.layers.patch_partition import PatchParition
from countgd.blocks.swint import SwinTransformerBlock

def swint(arr):
    patches = PatchParition(window_size=4)(arr)
    embeddings = LinearEmbedding(num_patches=3136, projection_dim=96)(patches)
    block = SwinTransformerBlock(96, (56, 56), 8, window_size=4)
    return tf.shape(block(embeddings))

def test_swint():
    arr = tf.zeros((1, 224, 224, 3))
    assert swint(arr).numpy().tolist() == [1, 3136, 96]
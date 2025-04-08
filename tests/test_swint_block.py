import tensorflow as tf
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.blocks.layers.patch_partition import PatchParition
from countgd.blocks.layers.patch_merging import PatchMerging
from countgd.blocks.swint import SwinTransformerBlock

def swint(arr):
    num_patch_x = 56
    num_patch_y = 56
    embed_dim = 128
    patches = PatchParition(patch_size=4)(arr)
    embeddings = LinearEmbedding(num_patches=num_patch_x * num_patch_y, embed_dim=embed_dim)(patches)
    block1 = SwinTransformerBlock((num_patch_x, num_patch_y), num_heads=2, embed_dim=embed_dim, window_size=7)(embeddings)
 
    return tf.shape(block1)

def test_swint():
    arr = tf.zeros((1, 224, 224, 3))
    assert swint(arr).numpy().tolist() == [1, 3136, 128]
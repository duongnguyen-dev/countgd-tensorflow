import tensorflow as tf
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.blocks.layers.patch_partition import PatchParition
from countgd.blocks.layers.patch_merging import PatchMerging
from countgd.blocks.swint import SwinTransformerBlock

def swint(arr):
    patches = PatchParition(patch_size=4)(arr)
    embeddings = LinearEmbedding(num_patches=3136, embed_dim=128)(patches)
    block = SwinTransformerBlock(128, (56, 56), 2, window_size=4)(embeddings)
    
    merge = PatchMerging((56, 56), channels=128)(block)
    num_patch_x = int(tf.sqrt(tf.cast(tf.shape(merge)[1], tf.float32)).numpy())
    num_patch_y = int(tf.sqrt(tf.cast(tf.shape(merge)[1], tf.float32)).numpy())
    embed_dim = tf.shape(merge)[-1].numpy().item()
    block = SwinTransformerBlock(embed_dim, (num_patch_x, num_patch_y), 2, window_size=4)(merge)
    return tf.shape(block)

def test_swint():
    arr = tf.zeros((1, 224, 224, 3))
    assert swint(arr).numpy().tolist() == [1, 784, 256]
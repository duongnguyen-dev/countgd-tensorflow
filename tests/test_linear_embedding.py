import tensorflow as tf
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.config import SwintBConfig

def linear_embedding(patch):
    patch_part = LinearEmbedding(num_patches=3136, embed_dim=SwintBConfig.EMBED_DIM)
    return tf.shape(patch_part(patch))

def test_linear_embedding():
    arr = tf.zeros((1, 3136, 48))
    assert linear_embedding(arr).numpy().tolist() == [1, 3136, 96]